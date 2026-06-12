import numpy as np
import pandas as pd

from vol_surrogate.surrogate.lhs import PARAM_BOUNDS, sample_params


class TestLHS:
    def test_bounds_respected(self):
        df = sample_params(500, seed=42)
        for name, (lo, hi) in PARAM_BOUNDS.items():
            assert df[name].min() >= lo
            assert df[name].max() <= hi

    def test_feller_respected(self):
        df = sample_params(500, seed=42)
        assert bool((2.0 * df["kappa"] * df["theta"] >= df["xi"] ** 2).all())

    def test_deterministic_in_seed(self):
        a = sample_params(100, seed=42)
        b = sample_params(100, seed=42)
        pd.testing.assert_frame_equal(a, b)
        c = sample_params(100, seed=7)
        assert not a.equals(c)

    def test_requested_count(self):
        assert len(sample_params(250, seed=42)) == 250


class TestGenerate:
    def test_dataset_shapes_and_sanity(self, tiny_dataset):
        X, Y = tiny_dataset["X"], tiny_dataset["Y"]
        assert X.shape[1] == 5
        assert Y.shape == (X.shape[0], 48)
        # implied vols in a sane band and no NaNs survived the filter
        assert np.isfinite(Y).all()
        assert Y.min() > 0.05 and Y.max() < 1.5

    def test_lattice_matches_spec(self, tiny_dataset):
        assert len(tiny_dataset["moneyness"]) == 8
        assert len(tiny_dataset["maturities"]) == 6
        assert tiny_dataset["moneyness"][0] == 0.8
        assert tiny_dataset["moneyness"][-1] == 1.2
