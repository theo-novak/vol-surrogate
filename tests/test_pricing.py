import numpy as np
import pytest

from vol_surrogate.pricing.black_scholes import bs_price, implied_vol_otm
from vol_surrogate.pricing.heston import (
    HestonParams,
    feller_condition,
    heston_call,
    heston_charfn,
    heston_put,
    implied_vol_grid_newton,
    iv_grid,
)
from vol_surrogate.pricing.sabr import sabr_lognormal_iv, sabr_smile


class TestCharacteristicFunction:
    def test_cf_at_zero_is_one(self, true_params):
        assert heston_charfn(0.0, 1.0, true_params, 100.0, 0.03) == pytest.approx(1.0)

    def test_martingale_property(self, true_params):
        """phi(-i) = E[S_T] = S0 e^{rT}."""
        S0, r, T = 100.0, 0.03, 1.5
        val = heston_charfn(-1j, T, true_params, S0, r)
        assert complex(val).real == pytest.approx(S0 * np.exp(r * T), rel=1e-8)

    def test_cf_modulus_bounded(self, true_params):
        u = np.linspace(0.1, 50.0, 100)
        assert np.all(np.abs(heston_charfn(u, 0.5, true_params, 100.0, 0.0)) <= 1.0 + 1e-12)


class TestHestonPricer:
    def test_put_call_parity(self, true_params):
        S0, r = 100.0, 0.03
        for K in [85.0, 100.0, 115.0]:
            for T in [0.25, 1.0]:
                c = heston_call(S0, K, r, T, true_params)
                p = heston_put(S0, K, r, T, true_params)
                assert c - p == pytest.approx(S0 - K * np.exp(-r * T), abs=1e-8)

    def test_price_within_no_arbitrage_bounds(self, true_params):
        S0, r, T = 100.0, 0.02, 0.75
        strikes = np.linspace(80.0, 120.0, 9)
        calls = heston_call(S0, strikes, r, T, true_params)
        lower = np.maximum(S0 - strikes * np.exp(-r * T), 0.0)
        assert np.all(calls > lower - 1e-9)
        assert np.all(calls < S0)
        # decreasing and convex in strike
        assert np.all(np.diff(calls) < 0)
        assert np.all(np.diff(calls, 2) > -1e-9)

    def test_bs_limit_vanishing_vol_of_vol(self):
        """xi -> 0 with v0 = theta collapses Heston to BS at sigma = sqrt(theta)."""
        params = HestonParams(kappa=5.0, theta=0.04, xi=1e-4, rho=0.0, v0=0.04)
        S0, r = 100.0, 0.03
        for K in [85.0, 100.0, 115.0]:
            c_h = heston_call(S0, K, r, 1.0, params)
            c_bs = bs_price(S0, K, r, 1.0, 0.2)
            assert c_h == pytest.approx(c_bs, rel=1e-4)

    def test_bs_limit_gives_flat_smile(self):
        params = HestonParams(kappa=5.0, theta=0.04, xi=1e-4, rho=0.0, v0=0.04)
        grid = iv_grid(params, np.linspace(0.85, 1.15, 7), np.array([0.5, 1.0]))
        assert np.allclose(grid, 0.2, atol=2e-4)


class TestIVGrid:
    def test_negative_rho_produces_skew(self, true_params, grid):
        g = iv_grid(true_params, grid["moneyness"], grid["maturities"])
        # low strikes carry higher implied vol at every maturity
        assert np.all(g[:, 0] > g[:, -1])

    def test_newton_matches_brent(self, true_params, grid):
        S0, r, T = 1.0, 0.0, 0.5
        strikes = grid["moneyness"]
        calls = np.atleast_1d(heston_call(S0, strikes, r, T, true_params))
        newton = implied_vol_grid_newton(calls, S0, strikes, r, T)
        brent = np.array([
            implied_vol_otm(float(c), S0, float(k), r, T) for c, k in zip(calls, strikes)
        ])
        assert np.allclose(newton, brent, atol=1e-8)

    def test_feller_condition(self):
        margin, ok = feller_condition(HestonParams(2.0, 0.04, 0.3, -0.7, 0.04))
        assert ok and margin == pytest.approx(2 * 2.0 * 0.04 - 0.09)
        _, bad = feller_condition(HestonParams(0.5, 0.02, 1.0, -0.7, 0.04))
        assert not bad


class TestSABR:
    def test_lognormal_limit(self):
        """beta = 1, nu -> 0 is exactly Black with sigma = alpha."""
        iv = sabr_lognormal_iv(100.0, 100.0, 1.0, alpha=0.25, beta=1.0, rho=0.0, nu=1e-10)
        assert iv == pytest.approx(0.25, rel=1e-6)

    def test_atm_continuity(self):
        """The z/x(z) ATM limit must join the wings smoothly."""
        args = dict(T=1.0, alpha=0.3, beta=0.7, rho=-0.3, nu=0.5)
        atm = sabr_lognormal_iv(100.0, 100.0, **args)
        near = sabr_lognormal_iv(100.0, 100.0001, **args)
        assert atm == pytest.approx(near, rel=1e-5)

    def test_negative_rho_skew(self):
        smile = sabr_smile(100.0, np.array([0.9, 1.0, 1.1]), 1.0,
                           alpha=0.25, beta=1.0, rho=-0.6, nu=0.6)
        assert smile[0] > smile[2]


class TestCrossValidation:
    def test_fft_matches_independent_quad_pricer(self, true_params):
        """The FFT and Gil-Pelaez quadrature pricers share no code; agreement
        validates the Simpson weights, damping, and spline interpolation."""
        from vol_surrogate.pricing.heston import heston_call_quad

        S0, r = 100.0, 0.03
        for T in [0.25, 1.0, 2.0]:
            for K in [80.0, 90.0, 100.0, 110.0, 120.0]:
                c_fft = heston_call(S0, K, r, T, true_params)
                c_quad = heston_call_quad(S0, K, r, T, true_params)
                assert c_fft == pytest.approx(c_quad, rel=2e-4, abs=2e-4)

    def test_bs_round_trip_iv_inversion(self):
        """price -> implied vol -> price recovers sigma to 1e-8."""
        from vol_surrogate.pricing.black_scholes import implied_vol

        S0, r, T = 100.0, 0.02, 0.75
        for sigma in [0.12, 0.25, 0.60]:
            for K in [85.0, 100.0, 115.0]:
                price = bs_price(S0, K, r, T, sigma)
                assert implied_vol(price, S0, K, r, T) == pytest.approx(sigma, abs=1e-8)
