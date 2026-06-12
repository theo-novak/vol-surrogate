"""Hagan et al. (2002) SABR lognormal implied-vol approximation.

The second ground-truth model: where Heston needs a Fourier transform per
maturity slice, SABR maps parameters to implied vol algebraically — which is
exactly the role a neural surrogate plays for Heston. Including it gives the
dataset generator a second analytic check and the page a like-for-like
comparison between an asymptotic-formula 'surrogate' and a learned one.
"""

import numpy as np


def sabr_lognormal_iv(
    F: float | np.ndarray,
    K: float | np.ndarray,
    T: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> float | np.ndarray:
    """Hagan lognormal (Black) implied vol, with the standard ATM limit.

    F : forward, K : strike, T : maturity
    alpha : initial vol level, beta : CEV exponent in [0, 1],
    rho : spot-vol correlation, nu : vol-of-vol.
    """
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    one_m_beta = 1.0 - beta

    FK = F * K
    logFK = np.log(F / K)
    FK_pow = FK ** (0.5 * one_m_beta)

    # z / x(z) factor; z -> 0 (ATM) limit is 1
    z = (nu / max(alpha, 1e-12)) * FK_pow * logFK
    sqrt_term = np.sqrt(1.0 - 2.0 * rho * z + z**2)
    x_z = np.log((sqrt_term + z - rho) / (1.0 - rho))
    z_over_x = np.where(np.abs(z) < 1e-8, 1.0, z / np.where(np.abs(x_z) < 1e-14, 1.0, x_z))

    denom = FK_pow * (
        1.0
        + (one_m_beta**2 / 24.0) * logFK**2
        + (one_m_beta**4 / 1920.0) * logFK**4
    )
    correction = 1.0 + T * (
        (one_m_beta**2 / 24.0) * alpha**2 / FK**one_m_beta
        + 0.25 * rho * beta * nu * alpha / FK_pow
        + ((2.0 - 3.0 * rho**2) / 24.0) * nu**2
    )

    iv = (alpha / denom) * z_over_x * correction
    if np.isscalar(F) or (iv.ndim == 0):
        return float(iv)
    return iv


def sabr_smile(
    F: float,
    moneyness: np.ndarray,
    T: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> np.ndarray:
    """SABR implied-vol slice on a moneyness grid (K = m * F)."""
    K = F * np.asarray(moneyness, dtype=float)
    return np.asarray(sabr_lognormal_iv(F, K, T, alpha, beta, rho, nu))
