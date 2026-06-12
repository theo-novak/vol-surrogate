import numpy as np
import pytest

from vol_surrogate.analysis.hedging import delta_hedge_backtest, simulate_heston_paths


class TestSimulation:
    def test_martingale_property(self, true_params):
        S0, r, T = 100.0, 0.03, 1.0
        S, _ = simulate_heston_paths(S0, r, true_params, T, n_steps=200,
                                     n_paths=20_000, seed=42)
        discounted_mean = S[:, -1].mean() * np.exp(-r * T)
        se = S[:, -1].std() / np.sqrt(len(S))
        assert abs(discounted_mean - S0) < 4 * se

    def test_variance_mean_reverts(self, true_params):
        _, v = simulate_heston_paths(100.0, 0.0, true_params, 5.0, n_steps=500,
                                     n_paths=10_000, seed=42)
        assert np.maximum(v[:, -1], 0.0).mean() == pytest.approx(true_params.theta, rel=0.1)

    def test_paths_deterministic_in_seed(self, true_params):
        a, _ = simulate_heston_paths(100.0, 0.0, true_params, 0.5, 50, 100, seed=42)
        b, _ = simulate_heston_paths(100.0, 0.0, true_params, 0.5, 50, 100, seed=42)
        assert np.array_equal(a, b)


class TestDeltaHedge:
    @pytest.fixture(scope="class")
    def result(self, true_params, grid):
        return delta_hedge_backtest(true_params, grid["moneyness"], grid["maturities"],
                                    T=0.5, n_steps=63, n_paths=800, seed=42)

    def test_hedging_reduces_variance(self, result):
        s = result.summary()
        assert s["std_model"] < 0.5 * s["std_unhedged"]
        assert s["std_bs"] < 0.5 * s["std_unhedged"]

    def test_min_variance_delta_beats_bs(self, result):
        """Under negative-rho Heston dynamics the minimum-variance delta
        (smile-consistent dC/dS plus the rho-xi vol-correlation correction)
        must hedge strictly better than the frozen-vol BS delta."""
        s = result.summary()
        assert s["std_model"] < s["std_bs"]
        assert s["std_reduction_vs_bs"] > 0.03

    def test_pnl_centred_near_zero(self, result):
        s = result.summary()
        assert abs(s["mean_model"]) < 1.0  # per option on S0 = 100

    def test_backtest_deterministic_in_seed(self, true_params, grid):
        a = delta_hedge_backtest(true_params, grid["moneyness"], grid["maturities"],
                                 T=0.25, n_steps=21, n_paths=200, seed=42)
        b = delta_hedge_backtest(true_params, grid["moneyness"], grid["maturities"],
                                 T=0.25, n_steps=21, n_paths=200, seed=42)
        assert np.array_equal(a.pnl_model, b.pnl_model)
        assert np.array_equal(a.pnl_bs, b.pnl_bs)
