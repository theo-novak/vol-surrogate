"""Parameter time-series tracking: daily recalibration as a regime monitor.

Once calibration costs milliseconds, refitting every day (or every quote
update) is free, and the fitted parameter path itself becomes a signal:
kappa spikes when the term structure inverts, xi rises when the wings bid up
— both classic stress signatures. This module provides the scaffold: a
synthetic two-regime parameter path, surface generation, and rolling
warm-started calibration.
"""

import numpy as np
import pandas as pd

from ..pricing.heston import PARAM_NAMES, HestonParams
from ..surrogate.calibrate import SurrogateCalibrator


def synthetic_param_path(n_days: int = 120, seed: int = 42) -> pd.DataFrame:
    """Two-regime daily Heston parameter path (calm -> stressed).

    Calm: slow mean reversion, modest vol-of-vol. Stress (last third):
    kappa and xi jump, v0 spikes above theta, rho deepens — the standard
    crisis signature. AR(1) noise keeps the path wandering inside the
    training box.
    """
    rng = np.random.default_rng(seed)
    calm = np.array([1.8, 0.045, 0.35, -0.55, 0.04])
    stress = np.array([4.5, 0.08, 0.85, -0.80, 0.13])
    switch = int(n_days * 2 / 3)

    levels = np.vstack([calm if d < switch else stress for d in range(n_days)])
    noise = np.zeros((n_days, 5))
    scale = np.array([0.15, 0.004, 0.03, 0.02, 0.006])
    for d in range(1, n_days):
        noise[d] = 0.9 * noise[d - 1] + rng.normal(0.0, scale)
    path = levels + noise

    lo = np.array([0.6, 0.022, 0.12, -0.93, 0.022])
    hi = np.array([7.8, 0.24, 1.15, -0.05, 0.24])
    path = np.clip(path, lo, hi)
    # keep the path Feller-respecting like the training set
    path[:, 2] = np.minimum(path[:, 2], np.sqrt(2.0 * path[:, 0] * path[:, 1]) - 1e-3)
    return pd.DataFrame(path, columns=list(PARAM_NAMES)).assign(day=np.arange(n_days))


def track_parameters(
    calibrator: SurrogateCalibrator,
    param_path: pd.DataFrame | None = None,
    noise_bp: float = 20.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Rolling daily calibration along a parameter path.

    Each day's 'market' surface is the surrogate surface at the true
    parameters plus quote noise (noise_bp of implied vol, mimicking bid-ask
    mid jitter). Calibration warm-starts at the previous day's fit — the
    production pattern. Returns a tidy frame with true and fitted parameters,
    fit RMSE, and wall time per day.
    """
    rng = np.random.default_rng(seed)
    path = param_path if param_path is not None else synthetic_param_path(seed=seed)

    rows = []
    x_prev: np.ndarray | None = None
    for _, row in path.iterrows():
        true_x = row[list(PARAM_NAMES)].to_numpy(dtype=float)
        surface = calibrator.predict(true_x)
        surface = surface + rng.normal(0.0, noise_bp * 1e-4, surface.shape)
        res = calibrator.calibrate(surface, x0=x_prev)
        x_prev = res.params.as_array()
        rec = {"day": int(row["day"])}
        rec |= {f"true_{k}": float(v) for k, v in zip(PARAM_NAMES, true_x)}
        rec |= {f"fit_{k}": float(v) for k, v in zip(PARAM_NAMES, x_prev)}
        rec |= {"rmse_iv": res.rmse_iv, "wall_time_s": res.wall_time_s}
        rows.append(rec)
    return pd.DataFrame(rows)
