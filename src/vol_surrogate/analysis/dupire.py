"""Dupire local volatility from the fitted implied-vol surface.

Gatheral's formulation in total implied variance w(y, T) = iv^2 * T at
log-moneyness y = ln(K/F) avoids differencing raw option prices (whose second
strike derivative is numerically vicious):

    sigma_loc^2(y, T) =                  dw/dT
                        -------------------------------------------------
                        1 - (y/w) w_y + 1/4 (-1/4 - 1/w + y^2/w^2) w_y^2
                          + 1/2 w_yy

On a flat Black-Scholes surface every smile term vanishes and the formula
returns the constant vol — the validation test. On the fitted Heston smile it
exposes the structural difference in smile dynamics: local vol freezes
today's smile into the spot dimension (the smile moves opposite to spot),
while stochastic vol moves the smile with spot.
"""

import numpy as np

from ..pricing.heston import HestonParams, iv_grid


def dupire_local_vol(
    iv: np.ndarray,
    moneyness: np.ndarray,
    maturities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Local-vol grid by central finite differences on total implied variance.

    iv: (n_mat, n_mon) implied vols on (maturities x moneyness), forward
    moneyness convention (S0 = 1, r = 0, F = S0).
    Returns (local_vol, interior_moneyness, interior_maturities) — the
    interior of the grid, where central differences exist. Nodes where the
    denominator is non-positive (numerical noise or genuine butterfly
    violation) come back NaN.
    """
    iv = np.asarray(iv, dtype=float)
    m = np.asarray(moneyness, dtype=float)
    T = np.asarray(maturities, dtype=float)
    y = np.log(m)
    w = iv**2 * T[:, None]

    # d/dy along strikes (non-uniform-safe via np.gradient), d/dT along maturities
    w_y = np.gradient(w, y, axis=1)
    w_yy = np.gradient(w_y, y, axis=1)
    w_T = np.gradient(w, T, axis=0)

    yy = y[None, :]
    denom = (
        1.0
        - (yy / w) * w_y
        + 0.25 * (-0.25 - 1.0 / w + yy**2 / w**2) * w_y**2
        + 0.5 * w_yy
    )
    local_var = np.where((denom > 1e-8) & (w_T > 0.0), w_T / np.maximum(denom, 1e-8), np.nan)

    inner = (slice(1, -1), slice(1, -1))
    return np.sqrt(local_var[inner]), m[1:-1], T[1:-1]


def dupire_from_heston(
    params: HestonParams,
    n_moneyness: int = 41,
    n_maturities: int = 25,
    m_lo: float = 0.8,
    m_hi: float = 1.2,
    t_lo: float = 0.1,
    t_hi: float = 2.0,
    fft_n: int = 2048,
) -> dict[str, np.ndarray]:
    """Dense Heston IV surface -> Dupire local vol, plus the inputs for plotting."""
    m = np.linspace(m_lo, m_hi, n_moneyness)
    T = np.linspace(t_lo, t_hi, n_maturities)
    iv = iv_grid(params, m, T, N=fft_n)
    local, m_in, T_in = dupire_local_vol(iv, m, T)
    return {
        "moneyness": m, "maturities": T, "iv": iv,
        "local_vol": local, "moneyness_inner": m_in, "maturities_inner": T_in,
    }
