import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def bs_price(
    S0: float, K: float, r: float, T: float, sigma: float, option_type: str = "call"
) -> float:
    """Black-Scholes European option price (no dividends)."""
    if T <= 0 or sigma <= 0:
        intrinsic = S0 - K if option_type == "call" else K - S0
        return float(max(intrinsic, 0.0))
    sqrtT = np.sqrt(T)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    if option_type == "call":
        return float(S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1))


def bs_vega(S0: float, K: float, r: float, T: float, sigma: float) -> float:
    """Black-Scholes vega dC/dsigma."""
    sqrtT = np.sqrt(T)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    return float(S0 * norm.pdf(d1) * sqrtT)


def bs_delta(
    S0: float, K: float, r: float, T: float, sigma: float, option_type: str = "call"
) -> float:
    """Black-Scholes delta dC/dS."""
    sqrtT = np.sqrt(T)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    if option_type == "call":
        return float(norm.cdf(d1))
    return float(norm.cdf(d1) - 1.0)


def implied_vol(
    price: float, S0: float, K: float, r: float, T: float, option_type: str = "call"
) -> float:
    """Invert Black-Scholes for the implied volatility via Brent's method."""
    intrinsic = max(S0 - K * np.exp(-r * T), 0.0) if option_type == "call" \
        else max(K * np.exp(-r * T) - S0, 0.0)
    if price <= intrinsic + 1e-12:
        return float("nan")

    def objective(sigma: float) -> float:
        return bs_price(S0, K, r, T, sigma, option_type) - price

    try:
        return float(brentq(objective, 1e-6, 5.0, xtol=1e-10))
    except ValueError:
        return float("nan")


def implied_vol_otm(
    call_price: float, S0: float, K: float, r: float, T: float
) -> float:
    """Implied vol inverted from the OTM side.

    Deep-ITM calls sit on intrinsic value and the inversion is numerically
    hopeless; below spot the parity put P = C - S0 + K e^{-rT} carries the
    same time value on a price that is small but well-resolved.
    """
    if K < S0:
        put = call_price - S0 + K * np.exp(-r * T)
        return implied_vol(put, S0, K, r, T, option_type="put")
    return implied_vol(call_price, S0, K, r, T, option_type="call")
