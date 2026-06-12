# vol-surrogate

Neural surrogate calibration of the Heston stochastic-volatility model: a
feedforward network learns the map from the five Heston parameters to the
full implied-volatility surface, replacing the Carr-Madan FFT pricer inside
the calibration loop and cutting fit time from seconds to milliseconds. The
sequel to the classical FFT pipeline built in the companion `heston_vol`
project.

## What it does

- **Ground-truth pricers** — Albrecher "little trap" Heston characteristic
  function, Carr-Madan FFT (Simpson weights `(3 - (-1)^j)/3`, `w0 = 1/3`,
  cubic-spline off-grid interpolation), an independent Gil-Pelaez quadrature
  cross-check, Black-Scholes with OTM-side implied-vol inversion, and the
  Hagan et al. (2002) SABR lognormal expansion.
- **Training-set generator** — Latin-hypercube sample of the
  (kappa, theta, xi, rho, v0) box, one FFT per maturity slice, vectorised
  Newton implied-vol inversion; 48-node lattice (8 moneyness x 6 maturities).
  Scales to 500k+ samples; the committed demo uses ~30k.
- **Surrogates** — a SiLU MLP (params -> flattened IV grid) and a
  transformer variant (lattice nodes as tokens), trained with relative MSE
  plus soft static no-arbitrage penalties (call-spread, butterfly, calendar).
- **Calibration as inversion** — bounded least squares through the frozen
  network with an exact autograd Jacobian; benchmarked wall-clock against
  direct FFT calibration on identical synthetic surfaces.
- **Analysis** — minimum-variance Heston deltas vs Black-Scholes in a
  delta-hedged P&L backtest on simulated Heston paths; Dupire local vol via
  finite differences on total implied variance; daily-recalibration parameter
  time series as regime indicators (kappa, xi).

## Layout

```
pyproject.toml
scripts/download_data.py        # SPY chain + spot + FRED 3m T-bill -> DuckDB
models/                         # committed demo checkpoint + JSON export
src/vol_surrogate/
  data/                         # Pydantic v2 schemas, yfinance/FRED fetchers, DuckDB store
  pricing/                      # heston.py (CF + FFT + IV grid), black_scholes.py, sabr.py
  surrogate/                    # lhs.py, dataset.py, mlp.py, transformer.py,
                                # losses.py, train.py, compare.py, calibrate.py
  analysis/                     # hedging.py, dupire.py, params_ts.py
  report/plots.py               # shared plotly figures
  cli.py                        # Typer CLI (volsur)
  app.py                        # server-side Streamlit dashboard
tests/                          # offline, deterministic (seed 42)
```

## Quick start

```bash
pip install -e ".[dev]"

volsur generate --n 30000            # LHS sample -> data/heston_dataset.npz
volsur train --hidden 80 --depth 3   # -> models/mlp_surrogate.pt
volsur export-weights                # -> models/mlp_weights.json (for the browser demo)

volsur calibrate --xi 0.45 --rho -0.65   # recover known params in milliseconds
volsur benchmark                          # surrogate vs direct FFT wall time
volsur hedge                              # min-variance vs BS delta-hedged P&L
volsur localvol                           # Dupire local vol from the Heston smile
volsur compare                            # MLP vs transformer study
volsur fetch --ticker SPY                 # live chain + rates -> DuckDB
volsur dashboard                          # server-side Streamlit app
```

## Tests

```bash
py -3.12 -m pytest
```

Fully offline and deterministic (seed 42): FFT vs independent quadrature,
put-call parity, BS round-trip inversion, SABR limits, no-arbitrage penalty
sign tests, overfit-50 sanity check, JSON-export parity with the torch
forward pass, parameter recovery through the committed checkpoint, Dupire
flat-surface recovery, and hedging determinism.

## Notes

- Python 3.11+; torch CPU is sufficient for everything here.
- `data/` (DuckDB, datasets) is git-ignored; `models/` commits the small
  demo checkpoint so the dashboard and tests work out of the box.
- The browser dashboard (`dashboard.html`) runs on stlite/Pyodide where torch
  does not exist: it re-implements the MLP forward pass in numpy from the
  exported JSON weights.
