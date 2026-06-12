"""Plotly figures shared by the CLI and the Streamlit app (plotly_dark theme)."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_TEMPLATE = "plotly_dark"


def fig_surface3d(
    moneyness: np.ndarray, maturities: np.ndarray, iv: np.ndarray,
    title: str = "Implied volatility surface",
) -> go.Figure:
    fig = go.Figure(go.Surface(
        x=moneyness, y=maturities, z=iv * 100.0, colorscale="Viridis",
        colorbar=dict(title="IV (%)"),
    ))
    fig.update_layout(
        template=_TEMPLATE, title=title, height=620,
        scene=dict(
            xaxis_title="Moneyness K/S0", yaxis_title="Maturity (y)",
            zaxis_title="Implied vol (%)",
        ),
    )
    return fig


def fig_error_heatmap(
    moneyness: np.ndarray, maturities: np.ndarray, err_bp: np.ndarray,
    title: str = "Surrogate error (bp of implied vol)",
) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        x=np.round(moneyness, 3), y=np.round(maturities, 3), z=err_bp,
        colorscale="RdBu", zmid=0.0, colorbar=dict(title="bp"),
    ))
    fig.update_layout(
        template=_TEMPLATE, title=title, height=480,
        xaxis_title="Moneyness K/S0", yaxis_title="Maturity (y)",
    )
    return fig


def fig_smile_overlay(
    moneyness: np.ndarray, maturities: np.ndarray,
    iv_true: np.ndarray, iv_pred: np.ndarray,
) -> go.Figure:
    """Analytic smiles as lines, surrogate predictions as markers."""
    fig = go.Figure()
    palette = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3"]
    for i, T in enumerate(maturities):
        c = palette[i % len(palette)]
        fig.add_trace(go.Scatter(
            x=moneyness, y=iv_true[i] * 100, mode="lines",
            line=dict(color=c), name=f"FFT T={T:.2f}y",
        ))
        fig.add_trace(go.Scatter(
            x=moneyness, y=iv_pred[i] * 100, mode="markers",
            marker=dict(color=c, symbol="circle-open", size=9),
            name=f"MLP T={T:.2f}y", showlegend=False,
        ))
    fig.update_layout(
        template=_TEMPLATE, height=520, title="Analytic Heston (lines) vs surrogate (markers)",
        xaxis_title="Moneyness K/S0", yaxis_title="Implied vol (%)",
    )
    return fig


def fig_training_curve(history: dict) -> go.Figure:
    fig = go.Figure()
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig.add_trace(go.Scatter(x=epochs, y=history["train_loss"], name="train loss",
                             line=dict(color="#636EFA")))
    fig.add_trace(go.Scatter(x=epochs, y=history["val_loss"], name="val loss",
                             line=dict(color="#EF553B")))
    fig.update_layout(
        template=_TEMPLATE, height=420, title="Training history",
        xaxis_title="Epoch", yaxis_title="Loss", yaxis_type="log",
    )
    return fig


def fig_param_timeseries(track: pd.DataFrame) -> go.Figure:
    """kappa and xi fitted vs true over the synthetic regime path."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("kappa (mean reversion)", "xi (vol-of-vol)"))
    for i, p in enumerate(["kappa", "xi"], start=1):
        fig.add_trace(go.Scatter(x=track["day"], y=track[f"true_{p}"], name=f"true {p}",
                                 line=dict(color="#00CC96", dash="dot")), row=i, col=1)
        fig.add_trace(go.Scatter(x=track["day"], y=track[f"fit_{p}"], name=f"fitted {p}",
                                 line=dict(color="#636EFA")), row=i, col=1)
    fig.update_layout(template=_TEMPLATE, height=560,
                      title="Daily recalibrated parameters as regime indicators")
    fig.update_xaxes(title_text="Day", row=2, col=1)
    return fig


def fig_pnl_hist(pnl_model: np.ndarray, pnl_bs: np.ndarray) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=pnl_bs, name="BS delta (frozen IV)",
                               marker_color="#EF553B", opacity=0.6, nbinsx=60))
    fig.add_trace(go.Histogram(x=pnl_model, name="Min-variance Heston delta",
                               marker_color="#636EFA", opacity=0.6, nbinsx=60))
    fig.update_layout(
        template=_TEMPLATE, barmode="overlay", height=460,
        title="Delta-hedged P&L per option (short ATM call, daily rebalance)",
        xaxis_title="Terminal hedging error", yaxis_title="Paths",
    )
    return fig


def fig_localvol_heatmap(
    moneyness: np.ndarray, maturities: np.ndarray, local_vol: np.ndarray
) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        x=np.round(moneyness, 3), y=np.round(maturities, 3), z=local_vol * 100,
        colorscale="Viridis", colorbar=dict(title="Local vol (%)"),
    ))
    fig.update_layout(
        template=_TEMPLATE, height=480, title="Dupire local volatility",
        xaxis_title="Moneyness K/S0", yaxis_title="Maturity (y)",
    )
    return fig
