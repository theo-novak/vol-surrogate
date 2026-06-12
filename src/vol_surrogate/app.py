"""Server-side Streamlit dashboard (the browser demo lives in dashboard.html).

Run via `volsur dashboard` or `streamlit run src/vol_surrogate/app.py` from the
project root. Loads the committed demo checkpoint from models/.
"""

import os
import time

import numpy as np
import streamlit as st

from vol_surrogate.surrogate.calibrate import SurrogateCalibrator
from vol_surrogate.analysis.hedging import delta_hedge_backtest
from vol_surrogate.pricing.heston import HestonParams, feller_condition, iv_grid
from vol_surrogate.report.plots import (
    fig_error_heatmap,
    fig_pnl_hist,
    fig_smile_overlay,
    fig_surface3d,
)
from vol_surrogate.surrogate.mlp import load_checkpoint, predict_iv

st.set_page_config(page_title="Vol Surface Surrogate", layout="wide")
st.title("Neural Surrogate for Implied-Vol Surface Calibration")

_MODEL_PATH = os.path.join(os.getenv("MODELS_DIR", "models"), "mlp_surrogate.pt")


@st.cache_resource
def _load():
    model, normalizer, meta = load_checkpoint(_MODEL_PATH)
    return model, normalizer, meta


try:
    model, normalizer, meta = _load()
except FileNotFoundError:
    st.error(f"No checkpoint at {_MODEL_PATH} — run `volsur generate` then `volsur train`.")
    st.stop()

MONEYNESS = np.asarray(meta["moneyness"])
MATURITIES = np.asarray(meta["maturities"])
GRID_SHAPE = (len(MATURITIES), len(MONEYNESS))

with st.sidebar:
    st.header("Heston parameters")
    kappa = st.slider("kappa — mean reversion", 0.5, 8.0, 2.0, 0.1)
    theta = st.slider("theta — long-run variance", 0.02, 0.25, 0.04, 0.005)
    xi = st.slider("xi — vol-of-vol", 0.10, 1.20, 0.50, 0.05)
    rho = st.slider("rho — spot-vol correlation", -0.95, 0.00, -0.70, 0.01)
    v0 = st.slider("v0 — initial variance", 0.02, 0.25, 0.04, 0.005)
    params = HestonParams(kappa=kappa, theta=theta, xi=xi, rho=rho, v0=v0)
    margin, ok = feller_condition(params)
    (st.success if ok else st.warning)(
        f"Feller 2kt − xi² = {margin:+.3f} ({'satisfied' if ok else 'violated'})"
    )

x = params.as_array()
t0 = time.perf_counter()
iv_sur = predict_iv(model, normalizer, x, grid_shape=GRID_SHAPE)
surrogate_ms = (time.perf_counter() - t0) * 1e3

tab1, tab2, tab3, tab4 = st.tabs(
    ["Surface", "Accuracy vs FFT", "Calibration", "Hedging P&L"]
)

with tab1:
    st.caption(f"Surrogate forward pass: {surrogate_ms:.2f} ms")
    st.plotly_chart(fig_surface3d(MONEYNESS, MATURITIES, iv_sur), width="stretch")

with tab2:
    with st.spinner("Pricing the analytic surface (6 FFTs + inversion)…"):
        t0 = time.perf_counter()
        iv_fft = iv_grid(params, MONEYNESS, MATURITIES)
        fft_ms = (time.perf_counter() - t0) * 1e3
    err_bp = (iv_sur - iv_fft) * 1e4
    c1, c2, c3 = st.columns(3)
    c1.metric("Surrogate", f"{surrogate_ms:.2f} ms")
    c2.metric("FFT + inversion", f"{fft_ms:.0f} ms")
    c3.metric("Max abs error", f"{np.nanmax(np.abs(err_bp)):.1f} bp")
    st.plotly_chart(fig_smile_overlay(MONEYNESS, MATURITIES, iv_fft, iv_sur),
                    width="stretch")
    st.plotly_chart(fig_error_heatmap(MONEYNESS, MATURITIES, err_bp), width="stretch")

with tab3:
    st.markdown("Synthetic market = analytic FFT surface at the slider parameters "
                "+ optional quote noise; the surrogate calibrates from a deliberately "
                "wrong start.")
    noise_bp = st.slider("Quote noise (bp of IV)", 0, 100, 20, 5)
    if st.button("Calibrate"):
        cal = SurrogateCalibrator(model, normalizer, MONEYNESS, MATURITIES)
        market = iv_grid(params, MONEYNESS, MATURITIES)
        market += np.random.default_rng(42).normal(0, noise_bp * 1e-4, market.shape)
        res = cal.calibrate(market)
        st.success(f"Calibrated in {res.wall_time_s * 1e3:.1f} ms "
                   f"({res.n_evals} residual evaluations)")
        rows = {
            "true": dict(zip(("kappa", "theta", "xi", "rho", "v0"), x)),
            "fitted": res.params.__dict__,
        }
        st.dataframe(rows, width="stretch")
        resid_bp = (cal.predict(res.params.as_array()) - market) * 1e4
        st.plotly_chart(
            fig_error_heatmap(MONEYNESS, MATURITIES, resid_bp,
                              title="Calibration residuals (bp)"),
            width="stretch",
        )

with tab4:
    n_paths = st.slider("Heston paths", 500, 5000, 2000, 500)
    if st.button("Run hedging backtest"):
        with st.spinner("Simulating and hedging…"):
            result = delta_hedge_backtest(
                params, MONEYNESS, MATURITIES, n_paths=n_paths,
                surface_fn=lambda x: predict_iv(model, normalizer, x,
                                                grid_shape=GRID_SHAPE),
            )
        s = result.summary()
        c1, c2, c3 = st.columns(3)
        c1.metric("BS-delta P&L std", f"{s['std_bs']:.3f}")
        c2.metric("Min-variance delta P&L std", f"{s['std_model']:.3f}")
        c3.metric("Std reduction", f"{s['std_reduction_vs_bs']:.1%}")
        st.plotly_chart(fig_pnl_hist(result.pnl_model, result.pnl_bs),
                        width="stretch")
