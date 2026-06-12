"""Generate (Heston params -> implied-vol grid) training pairs.

Each sample is one FFT per maturity plus a vectorised Newton inversion of the
whole strike slice. The fixed lattice is 8 moneyness levels x 6 maturities =
48 outputs; samples whose far wings are numerically unrecoverable (NaN implied
vol) are dropped and counted. Designed to scale to 500k+ samples; the CLI
defaults to a tractable size.
"""

import time
from pathlib import Path

import numpy as np

from ..pricing.heston import HestonParams, iv_grid
from .lhs import PARAM_BOUNDS, sample_params

# Fixed output lattice. Short end starts at 0.1y: at the box's minimum vol
# (sqrt(0.02) ~ 14%) the 0.8/1.2 wings are ~5 sd at T=0.1 — still invertible —
# whereas at T=0.05 they fall below FFT resolution (~1e-9 S0) and the lattice
# would silently bias the sample toward high-vol parameter draws.
GRID_MONEYNESS = np.linspace(0.80, 1.20, 8)
GRID_MATURITIES = np.array([0.1, 0.25, 0.5, 1.0, 1.5, 2.0])


def generate_dataset(
    n: int = 30_000,
    seed: int = 42,
    fft_n: int = 2048,
    progress_every: int = 2_000,
) -> dict[str, np.ndarray]:
    """LHS-sample the parameter box and price the IV grid for each sample.

    fft_n=2048 halves the characteristic-function cost; on this lattice it
    agrees with N=4096 to <1e-6 in implied vol (checked in tests).
    Returns dict with X (n, 5), Y (n, 48), the lattice, bounds, and the count
    of dropped samples.
    """
    params_df = sample_params(n, seed=seed)
    X = params_df.to_numpy()
    Y = np.empty((len(X), len(GRID_MATURITIES) * len(GRID_MONEYNESS)))
    keep = np.ones(len(X), dtype=bool)

    t0 = time.perf_counter()
    for i, row in enumerate(X):
        grid = iv_grid(HestonParams.from_array(row), GRID_MONEYNESS, GRID_MATURITIES, N=fft_n)
        if np.isnan(grid).any():
            keep[i] = False
            continue
        Y[i] = grid.ravel()
        if progress_every and (i + 1) % progress_every == 0:
            rate = (i + 1) / (time.perf_counter() - t0)
            print(f"  {i + 1}/{len(X)} samples ({rate:.0f}/s)", flush=True)

    dropped = int((~keep).sum())
    bounds = np.array([PARAM_BOUNDS[k] for k in params_df.columns])
    return {
        "X": X[keep],
        "Y": Y[keep],
        "moneyness": GRID_MONEYNESS,
        "maturities": GRID_MATURITIES,
        "bounds": bounds,
        "dropped": np.array(dropped),
        "seed": np.array(seed),
    }


def save_dataset(data: dict[str, np.ndarray], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    return path


def load_dataset(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path)) as z:
        return {k: z[k] for k in z.files}
