"""Heston characteristic-function pricer — the ground truth the surrogate learns.

Numerics adapted from the companion `heston_vol` project, whose FFT machinery
was validated against an independent Gil-Pelaez quadrature pricer:
- Albrecher et al. "little Heston trap" branching of the characteristic function
- Carr-Madan FFT with Simpson weights 1/3, 4/3, 2/3, 4/3, ... = (3 - (-1)^j)/3, w0 = 1/3
- CubicSpline interpolation off the log-strike grid (linear leaves O(1e-3) errors)
"""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import ndtr

from .black_scholes import implied_vol_otm

_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


@dataclass
class HestonParams:
    """The five Heston parameters.

    kappa : mean-reversion speed of variance
    theta : long-run variance level
    xi    : volatility of variance (vol-of-vol)
    rho   : correlation between spot and variance Brownian motions
    v0    : initial instantaneous variance
    """
    kappa: float
    theta: float
    xi: float
    rho: float
    v0: float

    def as_array(self) -> np.ndarray:
        return np.array([self.kappa, self.theta, self.xi, self.rho, self.v0])

    @classmethod
    def from_array(cls, x: np.ndarray) -> "HestonParams":
        return cls(*(float(v) for v in x))


PARAM_NAMES = ("kappa", "theta", "xi", "rho", "v0")


def feller_condition(params: HestonParams) -> tuple[float, bool]:
    """Feller condition 2*kappa*theta >= xi^2 keeps the variance strictly positive.

    Returns (2*kappa*theta - xi^2, satisfied).
    """
    margin = 2.0 * params.kappa * params.theta - params.xi**2
    return float(margin), bool(margin >= 0.0)


def heston_charfn(
    u: np.ndarray | complex,
    T: float,
    params: HestonParams,
    S0: float,
    r: float,
) -> np.ndarray | complex:
    """Characteristic function E[exp(i u ln S_T)] under the risk-neutral measure.

    Albrecher et al. "little Heston trap" branching — numerically stable for
    long maturities (Heston's original formulation crosses the negative real
    axis of the complex log and produces discontinuities).
    """
    kappa, theta, xi, rho, v0 = (
        params.kappa, params.theta, params.xi, params.rho, params.v0,
    )
    iu = 1j * np.asarray(u)
    x0 = np.log(S0)

    d = np.sqrt((rho * xi * iu - kappa) ** 2 + xi**2 * (iu + np.asarray(u) ** 2))
    g = (kappa - rho * xi * iu - d) / (kappa - rho * xi * iu + d)

    exp_dT = np.exp(-d * T)
    C = (kappa * theta / xi**2) * (
        (kappa - rho * xi * iu - d) * T - 2.0 * np.log((1.0 - g * exp_dT) / (1.0 - g))
    )
    D = ((kappa - rho * xi * iu - d) / xi**2) * ((1.0 - exp_dT) / (1.0 - g * exp_dT))

    return np.exp(iu * (x0 + r * T) + C + D * v0)


def carr_madan_fft(
    S0: float,
    r: float,
    T: float,
    params: HestonParams,
    alpha: float = 1.5,
    N: int = 4096,
    eta: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Price European calls on a log-strike grid via the Carr-Madan FFT.

    Returns (strikes, call_prices). One transform prices the whole strike
    slice — the property that makes direct calibration feasible at all, and
    that the surrogate then beats by three orders of magnitude.
    """
    lam = 2.0 * np.pi / (N * eta)
    b = 0.5 * N * lam
    v = np.arange(N) * eta
    k = -b + lam * np.arange(N)

    phi = heston_charfn(v - (alpha + 1.0) * 1j, T, params, S0, r)
    psi = np.exp(-r * T) * phi / (alpha**2 + alpha - v**2 + 1j * (2.0 * alpha + 1.0) * v)

    # Simpson's rule weights 1/3, 4/3, 2/3, 4/3, ...: (3 - (-1)^j) / 3 with w0 = 1/3
    simpson = (3.0 - (-1.0) ** np.arange(N)) / 3.0
    simpson[0] = 1.0 / 3.0
    integrand = np.exp(1j * b * v) * psi * eta * simpson

    fft_vals = np.fft.fft(integrand).real
    calls = np.exp(-alpha * k) / np.pi * fft_vals
    strikes = np.exp(k)
    return strikes, calls


def heston_call(
    S0: float,
    K: float | np.ndarray,
    r: float,
    T: float,
    params: HestonParams,
    alpha: float = 1.5,
    N: int = 4096,
    eta: float = 0.25,
) -> float | np.ndarray:
    """European call price under Heston, cubic-spline interpolated off the FFT grid."""
    strikes, calls = carr_madan_fft(S0, r, T, params, alpha=alpha, N=N, eta=eta)
    log_k_grid = np.log(strikes)
    log_k = np.log(np.asarray(K, dtype=float))

    lo = np.searchsorted(log_k_grid, np.min(log_k) - 0.5)
    hi = np.searchsorted(log_k_grid, np.max(log_k) + 0.5)
    spline = CubicSpline(log_k_grid[lo:hi], calls[lo:hi])
    out = spline(log_k)
    if np.isscalar(K):
        return float(out)
    return out


def heston_put(S0: float, K: float, r: float, T: float, params: HestonParams) -> float:
    """European put via put-call parity: P = C - S0 + K e^{-rT}."""
    return float(heston_call(S0, K, r, T, params) - S0 + K * np.exp(-r * T))


def _bs_otm_price_vec(
    sigma: np.ndarray, S0: float, K: np.ndarray, r: float, T: float, is_put: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised Black-Scholes OTM-side price and vega for the Newton inverter."""
    sqrtT = np.sqrt(T)
    sig = np.maximum(sigma, 1e-8)
    d1 = (np.log(S0 / K) + (r + 0.5 * sig**2) * T) / (sig * sqrtT)
    d2 = d1 - sig * sqrtT
    # scipy.special.ndtr is ~20x cheaper than norm.cdf (no dist-object overhead);
    # this loop runs millions of times during dataset generation.
    call = S0 * ndtr(d1) - K * np.exp(-r * T) * ndtr(d2)
    price = np.where(is_put, call - S0 + K * np.exp(-r * T), call)
    vega = S0 * _INV_SQRT_2PI * np.exp(-0.5 * d1**2) * sqrtT
    return price, vega


def implied_vol_grid_newton(
    call_prices: np.ndarray,
    S0: float,
    strikes: np.ndarray,
    r: float,
    T: float,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> np.ndarray:
    """Vectorised OTM-side implied vol for one maturity slice.

    Newton iteration with vega derivative across the whole strike slice at
    once (the per-point Brent loop is ~50x slower, which matters when the
    dataset generator inverts millions of quotes). Non-converged points fall
    back to the scalar Brent inverter; sub-intrinsic prices return NaN.
    """
    K = np.asarray(strikes, dtype=float)
    C = np.asarray(call_prices, dtype=float)
    is_put = K < S0
    target = np.where(is_put, C - S0 + K * np.exp(-r * T), C)
    intrinsic = np.where(
        is_put, np.maximum(K * np.exp(-r * T) - S0, 0.0), np.maximum(S0 - K * np.exp(-r * T), 0.0)
    )
    bad = target <= intrinsic + 1e-12

    # Brenner-Subrahmanyam-style starting point, clipped to a sane band
    sigma = np.clip(np.sqrt(2.0 * np.pi / T) * target / S0, 0.05, 2.0)
    converged = np.zeros_like(sigma, dtype=bool)
    for _ in range(max_iter):
        price, vega = _bs_otm_price_vec(sigma, S0, K, r, T, is_put)
        diff = price - target
        step = diff / np.maximum(vega, 1e-12)
        # Converged on price error OR on step size: deep wings have tiny vega,
        # where the vol is pinned long before the price tolerance is met.
        converged |= (np.abs(diff) < tol) | (np.abs(step) < 1e-9)
        if np.all(converged | bad):
            break
        sigma = np.clip(sigma - np.where(converged, 0.0, step), 1e-4, 5.0)

    iv = np.where(bad, np.nan, sigma)
    # Brent fallback for any straggler the Newton step did not converge
    for idx in np.flatnonzero(~converged & ~bad):
        iv[idx] = implied_vol_otm(float(C[idx]), S0, float(K[idx]), r, T)
    return iv


def iv_grid(
    params: HestonParams,
    moneyness: np.ndarray,
    maturities: np.ndarray,
    S0: float = 1.0,
    r: float = 0.0,
    N: int = 4096,
) -> np.ndarray:
    """Implied-vol grid over (maturity, moneyness) — the surrogate's target.

    Strikes are forward moneyness K/S0 with S0 = 1 and r = 0 (rates are
    stripped before fitting, as in Horvath-Muguruza-Tomas). One FFT prices
    each maturity slice; the slice is inverted to BS implied vols at once.
    Returns an array of shape (len(maturities), len(moneyness)); unrecoverable
    far-wing entries are NaN.
    """
    strikes = S0 * np.asarray(moneyness, dtype=float)
    out = np.empty((len(maturities), len(strikes)))
    for i, T in enumerate(maturities):
        calls = heston_call(S0, strikes, r, float(T), params, N=N)
        out[i] = implied_vol_grid_newton(np.atleast_1d(calls), S0, strikes, r, float(T))
    return out


def heston_call_quad(
    S0: float, K: float, r: float, T: float, params: HestonParams
) -> float:
    """Independent Gil-Pelaez quadrature pricer: C = S0 P1 - K e^{-rT} P2.

    Slow but shares no code with the FFT path — the cross-check that catches
    transposed Simpson weights (a ~1e-2 * S0 bias on every price) and sloppy
    off-grid interpolation.
    """
    from scipy.integrate import quad

    log_K = np.log(K)
    fwd = S0 * np.exp(r * T)

    def integrand_p1(u: float) -> float:
        phi = heston_charfn(u - 1j, T, params, S0, r)
        return (np.exp(-1j * u * log_K) * phi / (1j * u * fwd)).real

    def integrand_p2(u: float) -> float:
        phi = heston_charfn(u, T, params, S0, r)
        return (np.exp(-1j * u * log_K) * phi / (1j * u)).real

    P1 = 0.5 + quad(integrand_p1, 1e-8, 200.0, limit=200)[0] / np.pi
    P2 = 0.5 + quad(integrand_p2, 1e-8, 200.0, limit=200)[0] / np.pi
    return float(S0 * P1 - K * np.exp(-r * T) * P2)
