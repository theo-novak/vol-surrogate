from pathlib import Path

import numpy as np
import pytest

from vol_surrogate.surrogate.dataset import GRID_MATURITIES, GRID_MONEYNESS
from vol_surrogate.pricing.heston import HestonParams

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "models" / "mlp_surrogate.pt"

SEED = 42


@pytest.fixture(scope="session")
def grid() -> dict:
    return {"moneyness": GRID_MONEYNESS, "maturities": GRID_MATURITIES}


@pytest.fixture(scope="session")
def true_params() -> HestonParams:
    """Equity-index-like, Feller-satisfying, well inside the training box."""
    return HestonParams(kappa=3.0, theta=0.05, xi=0.45, rho=-0.65, v0=0.07)


@pytest.fixture(scope="session")
def tiny_dataset() -> dict:
    """Small real (params -> IV grid) dataset for fast training smoke tests."""
    from vol_surrogate.surrogate.dataset import generate_dataset

    data = generate_dataset(n=200, seed=SEED, progress_every=0)
    assert data["X"].shape[0] >= 190  # at most a few wing-dropped samples
    return data


@pytest.fixture(scope="session")
def checkpoint() -> tuple:
    """The committed demo checkpoint (real trained weights)."""
    from vol_surrogate.surrogate.mlp import load_checkpoint

    if not CHECKPOINT.exists():
        pytest.skip("demo checkpoint not trained yet")
    return load_checkpoint(CHECKPOINT)


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(SEED)
