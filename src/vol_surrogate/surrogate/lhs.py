"""Latin-hypercube sampling of the Heston parameter box.

LHS stratifies every marginal, so a 30k-sample design covers the 5-dimensional
box far more evenly than i.i.d. uniform draws — important because the
surrogate is only trustworthy inside the convex hull of its training set.
"""

import numpy as np
import pandas as pd
from scipy.stats import qmc

from ..pricing.heston import PARAM_NAMES

# (lower, upper) per parameter. The box brackets published index calibrations
# with margin while keeping every grid node's implied vol numerically
# recoverable (sqrt(v0) >= 14% keeps the 0.8/1.2 wings within ~5 sd at the
# shortest lattice maturity).
PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "kappa": (0.5, 8.0),
    "theta": (0.02, 0.25),
    "xi": (0.1, 1.2),
    "rho": (-0.95, 0.0),
    "v0": (0.02, 0.25),
}


def sample_params(
    n: int,
    seed: int = 42,
    feller_only: bool = True,
    oversample: float = 3.0,
) -> pd.DataFrame:
    """Latin-hypercube sample of HestonParams within PARAM_BOUNDS.

    With feller_only=True (default) the sample is filtered to the
    Feller-respecting region 2*kappa*theta >= xi^2, where the variance stays
    strictly positive and the surface map is smoothest. A single oversized LHS
    is drawn and filtered so the result is deterministic in (n, seed).
    """
    lower = np.array([PARAM_BOUNDS[k][0] for k in PARAM_NAMES])
    upper = np.array([PARAM_BOUNDS[k][1] for k in PARAM_NAMES])

    n_draw = int(np.ceil(n * oversample)) if feller_only else n
    sampler = qmc.LatinHypercube(d=len(PARAM_NAMES), seed=seed)
    unit = sampler.random(n_draw)
    samples = qmc.scale(unit, lower, upper)
    df = pd.DataFrame(samples, columns=list(PARAM_NAMES))

    if feller_only:
        ok = 2.0 * df["kappa"] * df["theta"] >= df["xi"] ** 2
        df = df[ok].reset_index(drop=True)
        if len(df) < n:
            raise ValueError(
                f"Only {len(df)} Feller-respecting samples from {n_draw} draws; "
                f"increase oversample (acceptance was {len(df) / n_draw:.1%})."
            )
        df = df.iloc[:n].reset_index(drop=True)
    return df
