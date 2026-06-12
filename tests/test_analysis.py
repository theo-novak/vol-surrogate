import numpy as np

from vol_surrogate.analysis.dupire import dupire_from_heston, dupire_local_vol


class TestDupire:
    def test_flat_bs_surface_returns_constant_vol(self):
        """On sigma_imp = const the denominator is 1 and dw/dT = sigma^2."""
        sigma = 0.23
        m = np.linspace(0.8, 1.2, 21)
        T = np.linspace(0.1, 2.0, 15)
        iv = np.full((len(T), len(m)), sigma)
        local, m_in, T_in = dupire_local_vol(iv, m, T)
        assert local.shape == (len(T) - 2, len(m) - 2)
        assert np.allclose(local, sigma, atol=1e-10)

    def test_heston_local_vol_sane(self, true_params):
        res = dupire_from_heston(true_params, n_moneyness=25, n_maturities=15)
        lv = res["local_vol"]
        finite = np.isfinite(lv)
        assert finite.mean() > 0.95  # almost all interior nodes valid
        assert 0.05 < np.nanmedian(lv) < 1.0
        # short-dated ATM local vol sits near the instantaneous vol sqrt(v0)
        j_atm = np.argmin(np.abs(res["moneyness_inner"] - 1.0))
        assert abs(lv[0, j_atm] - np.sqrt(true_params.v0)) < 0.05

    def test_local_vol_skew_steeper_than_implied(self, true_params):
        """2x rule of thumb: local vol skew is roughly twice the implied skew."""
        res = dupire_from_heston(true_params, n_moneyness=25, n_maturities=15)
        iv = res["iv"]
        lv = res["local_vol"]
        # implied skew at the first interior maturity row vs local skew
        iv_skew = iv[1, 2] - iv[1, -3]
        lv_skew = lv[0, 1] - lv[0, -2]
        assert lv_skew > iv_skew > 0
