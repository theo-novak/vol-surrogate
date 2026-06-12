"""MLP vs transformer-style surrogate, trained on identical data and budget.

Reports validation relative MSE, parameter counts, and single-surface
inference latency — the three numbers that decide which architecture earns
the calibration loop.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .mlp import MLPSurrogate, Normalizer
from .train import TrainConfig, train_surrogate
from .transformer import GridTransformer


def _latency_ms(model: nn.Module, normalizer: Normalizer, n_repeat: int = 200) -> float:
    """Median wall time for a single params -> surface forward pass."""
    x = torch.as_tensor(
        normalizer.norm_x(np.array([[2.0, 0.04, 0.5, -0.7, 0.04]])), dtype=torch.float32
    )
    with torch.no_grad():
        for _ in range(10):
            model(x)  # warm-up
        times = []
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            model(x)
            times.append(time.perf_counter() - t0)
    return float(np.median(times) * 1e3)


def compare_models(
    X: np.ndarray,
    Y: np.ndarray,
    moneyness: np.ndarray,
    maturities: np.ndarray,
    config: TrainConfig | None = None,
    out_path: str | Path | None = None,
) -> dict:
    """Train both architectures with the same data, budget, and seed."""
    cfg = config or TrainConfig()
    results: dict = {}

    candidates: dict[str, nn.Module] = {
        "mlp": MLPSurrogate(in_dim=X.shape[1], hidden=cfg.hidden, depth=cfg.depth,
                            out_dim=Y.shape[1]),
        "transformer": GridTransformer(in_dim=X.shape[1], n_tokens=Y.shape[1]),
    }
    for name, net in candidates.items():
        print(f"--- training {name} ---", flush=True)
        model, normalizer, history = train_surrogate(
            X, Y, moneyness, maturities, config=cfg, model=net, verbose=True
        )
        results[name] = {
            "val_rel_mse": history["best"]["val_rel_mse"],
            "val_mae_bp": history["best"]["val_mae_bp"],
            "val_maxae_bp": history["best"]["val_maxae_bp"],
            "n_params": sum(p.numel() for p in model.parameters()),
            "latency_ms": _latency_ms(model, normalizer),
            "epochs": len(history["train_loss"]),
            "wall_time_s": history["wall_time_s"],
        }

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
    return results
