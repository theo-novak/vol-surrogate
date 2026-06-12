import os
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich import print as rprint
from rich.table import Table

load_dotenv()
app = typer.Typer(help="Neural surrogate for implied-vol surface calibration.")

_DEFAULT = dict(kappa=2.0, theta=0.04, xi=0.5, rho=-0.7, v0=0.04)
_MODEL_PATH = os.getenv("MODELS_DIR", "models") + "/mlp_surrogate.pt"
_DATASET_PATH = os.getenv("DATASET_PATH", "data/heston_dataset.npz")


def _load_calibrator():
    from .surrogate.calibrate import SurrogateCalibrator
    from .surrogate.mlp import load_checkpoint

    model, normalizer, meta = load_checkpoint(_MODEL_PATH)
    return SurrogateCalibrator(model, normalizer, meta["moneyness"], meta["maturities"])


@app.command()
def fetch(
    ticker: str = typer.Option("SPY", help="Underlying ticker (SPY proxies SPX)"),
    max_expiries: int = typer.Option(8, help="Number of expiries to pull"),
    db: str = typer.Option(os.getenv("DB_PATH", "data/vol_surrogate.duckdb")),
) -> None:
    """Download spot, the live option chain, and the FRED 3m T-bill to DuckDB."""
    from .data.fetchers import fetch_option_chain, fetch_riskfree, fetch_spot
    from .data.store import init_db, upsert_options, upsert_rates, upsert_spot

    Path(db).parent.mkdir(parents=True, exist_ok=True)
    con = init_db(db)
    rprint(f"[cyan]Fetching {ticker} spot + option chain…[/cyan]")
    n_spot = upsert_spot(con, fetch_spot(ticker))
    n_opt = upsert_options(con, fetch_option_chain(ticker, max_expiries=max_expiries))
    rates = fetch_riskfree()
    n_rate = upsert_rates(con, rates)
    rprint(f"[green]{n_spot} spot rows, {n_opt} option quotes, {n_rate} rate rows "
           f"(latest 3m T-bill {rates.iloc[-1]['rate']:.2%})[/green]")
    con.close()


@app.command()
def generate(
    n: int = typer.Option(30_000, help="Number of LHS samples (design scales to 500k+)"),
    seed: int = typer.Option(42),
    out: str = typer.Option(_DATASET_PATH, help="Output .npz path"),
) -> None:
    """LHS-sample the Heston box and price the training IV grids."""
    from .surrogate.dataset import generate_dataset, save_dataset

    rprint(f"[cyan]Generating {n} (params → IV grid) pairs (seed {seed})…[/cyan]")
    data = generate_dataset(n=n, seed=seed)
    path = save_dataset(data, out)
    rprint(f"[green]Saved {data['X'].shape[0]} samples to {path} "
           f"({int(data['dropped'])} dropped for unrecoverable wings)[/green]")


@app.command()
def train(
    dataset: str = typer.Option(_DATASET_PATH),
    out: str = typer.Option(_MODEL_PATH),
    hidden: int = typer.Option(256),
    depth: int = typer.Option(4),
    max_epochs: int = typer.Option(60),
    lambda_arb: float = typer.Option(1.0, help="No-arbitrage penalty weight"),
    seed: int = typer.Option(42),
) -> None:
    """Train the MLP surrogate with early stopping; saves weights + norm stats."""
    from .surrogate.dataset import load_dataset
    from .surrogate.mlp import save_checkpoint
    from .surrogate.train import TrainConfig, train_surrogate

    data = load_dataset(dataset)
    cfg = TrainConfig(hidden=hidden, depth=depth, max_epochs=max_epochs,
                      lambda_arb=lambda_arb, seed=seed)
    model, normalizer, history = train_surrogate(
        data["X"], data["Y"], data["moneyness"], data["maturities"], cfg
    )
    path = save_checkpoint(out, model, normalizer, data["moneyness"], data["maturities"],
                           history={k: history[k] for k in ("train_loss", "val_loss")},
                           metrics=history["best"])
    b = history["best"]
    rprint(f"[green]Saved {path} — val rel-MSE {b['val_rel_mse']:.3e}, "
           f"val MAE {b['val_mae_bp']:.2f} bp, max abs err {b['val_maxae_bp']:.1f} bp[/green]")


@app.command(name="export-weights")
def export_weights(
    checkpoint: str = typer.Option(_MODEL_PATH),
    out: str = typer.Option("models/mlp_weights.json"),
) -> None:
    """Export the trained MLP to compact JSON for the browser dashboard."""
    from .surrogate.dataset import load_dataset
    from .surrogate.mlp import export_weights_json, load_checkpoint

    model, normalizer, meta = load_checkpoint(checkpoint)
    bounds = None
    try:
        bounds = load_dataset(_DATASET_PATH)["bounds"]
    except FileNotFoundError:
        pass
    path = export_weights_json(out, model, normalizer, meta["moneyness"],
                               meta["maturities"], bounds=bounds, metrics=meta["metrics"])
    size_kb = path.stat().st_size / 1024
    rprint(f"[green]Exported {path} ({size_kb:.0f} KB)[/green]")


@app.command()
def compare(
    dataset: str = typer.Option(_DATASET_PATH),
    out: str = typer.Option("models/compare.json"),
    max_epochs: int = typer.Option(30),
) -> None:
    """Train MLP vs transformer surrogate on the same budget and report."""
    from .surrogate.dataset import load_dataset
    from .surrogate.compare import compare_models
    from .surrogate.train import TrainConfig

    data = load_dataset(dataset)
    results = compare_models(data["X"], data["Y"], data["moneyness"], data["maturities"],
                             TrainConfig(max_epochs=max_epochs), out_path=out)
    table = Table(title="MLP vs transformer surrogate")
    table.add_column("Model", style="cyan")
    for col in ("val rel-MSE", "val MAE (bp)", "params", "latency (ms)"):
        table.add_column(col, justify="right")
    for name, res in results.items():
        table.add_row(name, f"{res['val_rel_mse']:.3e}", f"{res['val_mae_bp']:.2f}",
                      f"{res['n_params']:,}", f"{res['latency_ms']:.3f}")
    rprint(table)


@app.command()
def calibrate(
    kappa: float = typer.Option(_DEFAULT["kappa"]),
    theta: float = typer.Option(_DEFAULT["theta"]),
    xi: float = typer.Option(_DEFAULT["xi"]),
    rho: float = typer.Option(_DEFAULT["rho"]),
    v0: float = typer.Option(_DEFAULT["v0"]),
    noise_bp: float = typer.Option(0.0, help="Quote noise added to the synthetic surface"),
) -> None:
    """Recover known parameters from an FFT-generated synthetic market surface."""
    import numpy as np

    from .pricing.heston import PARAM_NAMES, HestonParams, iv_grid

    cal = _load_calibrator()
    true = HestonParams(kappa=kappa, theta=theta, xi=xi, rho=rho, v0=v0)
    market = iv_grid(true, cal.moneyness, cal.maturities)
    if noise_bp > 0:
        market += np.random.default_rng(42).normal(0, noise_bp * 1e-4, market.shape)
    res = cal.calibrate(market)

    table = Table(title=f"Surrogate calibration ({res.wall_time_s * 1e3:.1f} ms, "
                        f"{res.n_evals} evals)")
    table.add_column("Parameter", style="cyan")
    table.add_column("True", justify="right")
    table.add_column("Fitted", justify="right")
    for name, t, f in zip(PARAM_NAMES, true.as_array(), res.params.as_array()):
        table.add_row(name, f"{t:.4f}", f"{f:.4f}")
    table.add_row("RMSE (IV)", "—", f"{res.rmse_iv * 1e4:.2f} bp")
    table.add_row("Feller", "—", "satisfied" if res.feller_satisfied else "VIOLATED")
    rprint(table)


@app.command()
def benchmark() -> None:
    """Wall-time: surrogate calibration vs direct FFT calibration."""
    from .surrogate.calibrate import benchmark_calibration

    cal = _load_calibrator()
    rprint("[cyan]Benchmarking (direct FFT calibration takes a moment)…[/cyan]")
    res = benchmark_calibration(cal)
    table = Table(title="Calibration wall time")
    table.add_column("Engine", style="cyan")
    table.add_column("Time", justify="right")
    table.add_column("Evals", justify="right")
    table.add_column("RMSE (bp)", justify="right")
    table.add_row("Surrogate (torch jac)", f"{res['surrogate']['wall_time_s'] * 1e3:.1f} ms",
                  str(res['surrogate']['n_evals']), f"{res['surrogate']['rmse_iv'] * 1e4:.2f}")
    table.add_row("Direct FFT", f"{res['direct']['wall_time_s']:.2f} s",
                  str(res['direct']['n_evals']), f"{res['direct']['rmse_iv'] * 1e4:.2f}")
    table.add_row("Speed-up", f"[bold]{res['speedup']:.0f}x[/bold]", "—", "—")
    rprint(table)


@app.command()
def hedge(
    n_paths: int = typer.Option(2000),
    maturity: float = typer.Option(0.5),
    kappa: float = typer.Option(_DEFAULT["kappa"]),
    theta: float = typer.Option(_DEFAULT["theta"]),
    xi: float = typer.Option(_DEFAULT["xi"]),
    rho: float = typer.Option(_DEFAULT["rho"]),
    v0: float = typer.Option(_DEFAULT["v0"]),
) -> None:
    """Delta-hedged P&L: Heston minimum-variance deltas vs frozen-IV BS deltas."""
    from pathlib import Path as _Path

    from .analysis.hedging import delta_hedge_backtest
    from .pricing.heston import HestonParams
    from .surrogate.dataset import GRID_MATURITIES, GRID_MONEYNESS

    true = HestonParams(kappa=kappa, theta=theta, xi=xi, rho=rho, v0=v0)
    surface_fn = None
    if _Path(_MODEL_PATH).exists():
        # Greeks through the learned surface: the IV cube is built by batched
        # surrogate forward passes instead of one FFT per v0 level.
        from .surrogate.mlp import load_checkpoint, predict_iv

        model, normalizer, meta = load_checkpoint(_MODEL_PATH)
        shape = (len(meta["maturities"]), len(meta["moneyness"]))
        surface_fn = lambda x: predict_iv(model, normalizer, x, grid_shape=shape)  # noqa: E731
        rprint("[cyan]Using the surrogate for the model hedger's IV cube[/cyan]")
    result = delta_hedge_backtest(true, GRID_MONEYNESS, GRID_MATURITIES,
                                  T=maturity, n_paths=n_paths, surface_fn=surface_fn)
    s = result.summary()
    table = Table(title=f"Delta-hedged P&L, short ATM call, {n_paths} Heston paths")
    table.add_column("Hedger", style="cyan")
    table.add_column("P&L std", justify="right")
    table.add_column("P&L mean", justify="right")
    table.add_row("Unhedged", f"{s['std_unhedged']:.3f}", "—")
    table.add_row("BS delta (frozen IV)", f"{s['std_bs']:.3f}", f"{s['mean_bs']:+.3f}")
    table.add_row("Min-variance Heston delta", f"{s['std_model']:.3f}", f"{s['mean_model']:+.3f}")
    table.add_row("Std reduction vs BS", f"[bold]{s['std_reduction_vs_bs']:.1%}[/bold]", "—")
    rprint(table)


@app.command()
def localvol(
    kappa: float = typer.Option(_DEFAULT["kappa"]),
    theta: float = typer.Option(_DEFAULT["theta"]),
    xi: float = typer.Option(_DEFAULT["xi"]),
    rho: float = typer.Option(_DEFAULT["rho"]),
    v0: float = typer.Option(_DEFAULT["v0"]),
) -> None:
    """Dupire local-vol surface from the Heston smile (finite differences on w)."""
    import numpy as np

    from .analysis.dupire import dupire_from_heston
    from .pricing.heston import HestonParams

    res = dupire_from_heston(HestonParams(kappa=kappa, theta=theta, xi=xi, rho=rho, v0=v0))
    lv, m, T = res["local_vol"], res["moneyness_inner"], res["maturities_inner"]
    table = Table(title="Dupire local vol (selected nodes)")
    table.add_column("T \\ m", style="cyan")
    cols = np.linspace(0, len(m) - 1, 7).astype(int)
    for j in cols:
        table.add_column(f"{m[j]:.2f}", justify="right")
    for i in np.linspace(0, len(T) - 1, 6).astype(int):
        table.add_row(f"{T[i]:.2f}y", *[
            f"{lv[i, j] * 100:.1f}%" if np.isfinite(lv[i, j]) else "—" for j in cols
        ])
    rprint(table)


@app.command()
def dashboard() -> None:
    """Launch the Streamlit server-side dashboard."""
    import subprocess
    import sys

    subprocess.run([sys.executable, "-m", "streamlit", "run",
                    str(Path(__file__).parent / "app.py")])


if __name__ == "__main__":
    app()
