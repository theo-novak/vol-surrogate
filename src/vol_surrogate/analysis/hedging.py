"""Delta-hedged P&L backtest: Heston minimum-variance Greeks vs Black-Scholes.

The experiment: simulate spot/variance paths under true Heston dynamics, sell
an ATM call on each path, and rebalance a delta hedge daily. The Black-Scholes
hedger keeps the initial implied vol frozen and sees only spot. The model
hedger holds the *minimum-variance* Heston delta,

    delta_mv = dC/dS |_v  +  (rho * xi / S) * dC/dv,

where the first term uses the model's true smile dynamics (Heston is
homogeneous of degree one in (S, K), so at fixed variance the smile rides
moneyness K/S) and the second term hedges the part of the vol move that is
correlated with the spot move — cov(dv, dS)/var(dS) = rho*xi/S. With rho < 0
this *cuts* the delta below Black-Scholes: when spot falls, vol rises and
lifts the short call's value, so less stock is needed to offset the move.

Both Greeks are bump-and-reprice through an implied-vol cube over
(v0, maturity, moneyness) built once before the run — by default from the
analytic FFT pricer, optionally from the surrogate (one batched forward pass
over the v0 grid, which is how a production desk would do per-path Greeks).
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.special import ndtr

from ..pricing.heston import HestonParams, iv_grid


def simulate_heston_paths(
    S0: float,
    r: float,
    params: HestonParams,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Full-truncation Euler (Lord et al. 2010): v+ = max(v, 0) in drift and
    diffusion — the smallest-bias Euler variant for the CIR variance."""
    rng = np.random.default_rng(seed)
    kappa, theta, xi, rho, v0 = params.as_array()
    dt = T / n_steps

    S = np.empty((n_paths, n_steps + 1))
    v = np.empty((n_paths, n_steps + 1))
    S[:, 0], v[:, 0] = S0, v0
    log_s = np.full(n_paths, np.log(S0))

    for t in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = rho * z1 + np.sqrt(1.0 - rho**2) * rng.standard_normal(n_paths)
        v_plus = np.maximum(v[:, t], 0.0)
        sqrt_v_dt = np.sqrt(v_plus * dt)

        log_s = log_s + (r - 0.5 * v_plus) * dt + sqrt_v_dt * z1
        v[:, t + 1] = v[:, t] + kappa * (theta - v_plus) * dt + xi * sqrt_v_dt * z2
        S[:, t + 1] = np.exp(log_s)
    return S, v


def _bs_call_unit(m: np.ndarray, tau: float, sig: np.ndarray) -> np.ndarray:
    """Black-Scholes call with S = 1, K = m, r = 0 (forward convention)."""
    sig = np.maximum(sig, 1e-4)
    sqrtT = np.sqrt(tau)
    d1 = (-np.log(m) + 0.5 * sig**2 * tau) / (sig * sqrtT)
    return ndtr(d1) - m * ndtr(d1 - sig * sqrtT)


def _bs_delta_vec(S: np.ndarray, K: float, tau: float, sig: float) -> np.ndarray:
    d1 = (np.log(S / K) + 0.5 * sig**2 * tau) / (sig * np.sqrt(tau))
    return ndtr(d1)


def build_iv_cube(
    params: HestonParams,
    moneyness: np.ndarray,
    maturities: np.ndarray,
    v_grid: np.ndarray,
    surface_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    fft_n: int = 2048,
) -> RegularGridInterpolator:
    """Implied-vol cube iv(v0, T, m): one surface per v0 level.

    surface_fn maps a 5-vector of Heston parameters to an (n_mat, n_mon) IV
    grid; default is the analytic FFT pricer. Passing the surrogate's predict
    here turns every per-path Greek below into a batched network evaluation.
    """
    cube = np.empty((len(v_grid), len(maturities), len(moneyness)))
    for i, v in enumerate(v_grid):
        p = HestonParams(params.kappa, params.theta, params.xi, params.rho, float(v))
        if surface_fn is not None:
            cube[i] = surface_fn(p.as_array())
        else:
            cube[i] = iv_grid(p, moneyness, maturities, N=fft_n)
    return RegularGridInterpolator(
        (v_grid, maturities, moneyness), cube, bounds_error=False, fill_value=None
    )


@dataclass
class HedgeResult:
    pnl_model: np.ndarray
    pnl_bs: np.ndarray
    pnl_unhedged: np.ndarray

    def summary(self) -> dict:
        return {
            "std_model": float(self.pnl_model.std()),
            "std_bs": float(self.pnl_bs.std()),
            "std_unhedged": float(self.pnl_unhedged.std()),
            "mean_model": float(self.pnl_model.mean()),
            "mean_bs": float(self.pnl_bs.mean()),
            "std_reduction_vs_bs": float(1.0 - self.pnl_model.std() / self.pnl_bs.std()),
        }


def delta_hedge_backtest(
    params: HestonParams,
    moneyness: np.ndarray,
    maturities: np.ndarray,
    S0: float = 100.0,
    T: float = 0.5,
    n_steps: int = 126,
    n_paths: int = 2000,
    bump: float = 0.01,
    v_bump: float = 0.005,
    n_v_grid: int = 9,
    surface_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    seed: int = 42,
) -> HedgeResult:
    """Daily delta hedge of a short ATM call on simulated Heston paths (r = 0).

    The model hedger re-prices through the IV cube at the path's *current*
    variance v_t (the model state a desk would carry from its last
    calibration) and holds delta_mv; the BS hedger holds N(d1) at the frozen
    initial ATM vol. P&L is terminal wealth per option.
    """
    K = S0
    m_grid = np.asarray(moneyness, dtype=float)
    t_grid = np.asarray(maturities, dtype=float)

    # v0-grid spans the variance band the paths realistically visit
    v_lo = max(0.25 * min(params.v0, params.theta), 5e-3)
    v_hi = 3.0 * max(params.v0, params.theta)
    v_grid = np.linspace(v_lo, v_hi, n_v_grid)
    cube = build_iv_cube(params, m_grid, t_grid, v_grid, surface_fn=surface_fn)

    def iv_at(v: np.ndarray, tau: float, m: np.ndarray) -> np.ndarray:
        pts = np.column_stack([
            np.clip(v, v_grid[0], v_grid[-1]),
            np.full_like(m, np.clip(tau, t_grid[0], t_grid[-1])),
            np.clip(m, m_grid[0], m_grid[-1]),
        ])
        return cube(pts)

    def call_price(spot: np.ndarray, v: np.ndarray, tau: float) -> np.ndarray:
        m = K / spot
        return spot * _bs_call_unit(m, tau, iv_at(v, tau, m))

    S, v = simulate_heston_paths(S0, 0.0, params, T, n_steps, n_paths, seed=seed)
    dt = T / n_steps

    sigma0 = float(iv_at(np.array([params.v0]), T, np.array([1.0]))[0])
    premium = float(call_price(np.array([S0]), np.array([params.v0]), T)[0])

    cash_mv = np.full(n_paths, premium)
    cash_bs = np.full(n_paths, premium)
    pos_mv = np.zeros(n_paths)
    pos_bs = np.zeros(n_paths)

    for t in range(n_steps):
        tau = T - t * dt
        spot = S[:, t]
        var = np.maximum(v[:, t], 1e-6)

        # dC/dS at fixed variance (central bump; smile rides K/S by homogeneity)
        up = call_price(spot * (1.0 + bump), var, tau)
        dn = call_price(spot * (1.0 - bump), var, tau)
        delta_s = (up - dn) / (2.0 * bump * spot)

        # dC/dv (central bump in the variance state)
        cv_up = call_price(spot, var + v_bump, tau)
        cv_dn = call_price(spot, np.maximum(var - v_bump, 1e-6), tau)
        dC_dv = (cv_up - cv_dn) / (2.0 * v_bump)

        delta_mv = delta_s + (params.rho * params.xi / spot) * dC_dv
        delta_bs = _bs_delta_vec(spot, K, tau, sigma0)

        cash_mv -= (delta_mv - pos_mv) * spot
        cash_bs -= (delta_bs - pos_bs) * spot
        pos_mv, pos_bs = delta_mv, delta_bs

    ST = S[:, -1]
    payoff = np.maximum(ST - K, 0.0)
    pnl_mv = cash_mv + pos_mv * ST - payoff
    pnl_bs = cash_bs + pos_bs * ST - payoff
    pnl_unhedged = premium - payoff

    return HedgeResult(pnl_model=pnl_mv, pnl_bs=pnl_bs, pnl_unhedged=pnl_unhedged)
