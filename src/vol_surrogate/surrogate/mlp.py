"""MLP surrogate: five Heston parameters -> 48-node implied-vol grid.

The network learns the pricing map offline; online it replaces the FFT
pricer inside the calibration loop. Checkpoints store float16 weights plus
the normalisation statistics and the output lattice, so a checkpoint is fully
self-describing (a few hundred KB, small enough to commit).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


@dataclass
class Normalizer:
    """Z-score normalisation statistics for inputs and outputs."""
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray, Y: np.ndarray) -> "Normalizer":
        return cls(
            x_mean=X.mean(axis=0), x_std=X.std(axis=0) + 1e-12,
            y_mean=Y.mean(axis=0), y_std=Y.std(axis=0) + 1e-12,
        )

    def norm_x(self, X: np.ndarray) -> np.ndarray:
        return (X - self.x_mean) / self.x_std

    def denorm_x(self, Xn: np.ndarray) -> np.ndarray:
        return Xn * self.x_std + self.x_mean

    def norm_y(self, Y: np.ndarray) -> np.ndarray:
        return (Y - self.y_mean) / self.y_std

    def denorm_y(self, Yn: np.ndarray) -> np.ndarray:
        return Yn * self.y_std + self.y_mean

    def torch_y_stats(self, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.as_tensor(self.y_mean, dtype=torch.float32, device=device),
            torch.as_tensor(self.y_std, dtype=torch.float32, device=device),
        )


class MLPSurrogate(nn.Module):
    """Feedforward surrogate pricer: params (5) -> flattened IV grid (48).

    SiLU activations: smooth (the calibrator differentiates through the net)
    and empirically better than ReLU on this smooth regression target.
    """

    def __init__(
        self, in_dim: int = 5, hidden: int = 256, depth: int = 4, out_dim: int = 48
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)
        self.in_dim, self.hidden, self.depth, self.out_dim = in_dim, hidden, depth, out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def save_checkpoint(
    path: str | Path,
    model: MLPSurrogate,
    normalizer: Normalizer,
    moneyness: np.ndarray,
    maturities: np.ndarray,
    history: dict | None = None,
    metrics: dict | None = None,
) -> Path:
    """Save a self-describing checkpoint with float16 weights (halves the file)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state_fp16 = {k: v.half() for k, v in model.state_dict().items()}
    torch.save({
        "state_dict": state_fp16,
        "arch": {
            "in_dim": model.in_dim, "hidden": model.hidden,
            "depth": model.depth, "out_dim": model.out_dim,
        },
        "x_mean": normalizer.x_mean, "x_std": normalizer.x_std,
        "y_mean": normalizer.y_mean, "y_std": normalizer.y_std,
        "moneyness": np.asarray(moneyness), "maturities": np.asarray(maturities),
        "history": history or {}, "metrics": metrics or {},
    }, path)
    return path


def load_checkpoint(path: str | Path) -> tuple[MLPSurrogate, Normalizer, dict]:
    """Load a checkpoint; weights are cast back to float32 for inference."""
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
    model = MLPSurrogate(**ckpt["arch"])
    model.load_state_dict({k: v.float() for k, v in ckpt["state_dict"].items()})
    model.eval()
    normalizer = Normalizer(
        x_mean=ckpt["x_mean"], x_std=ckpt["x_std"],
        y_mean=ckpt["y_mean"], y_std=ckpt["y_std"],
    )
    meta = {
        "moneyness": ckpt["moneyness"], "maturities": ckpt["maturities"],
        "history": ckpt["history"], "metrics": ckpt["metrics"],
    }
    return model, normalizer, meta


def predict_iv(
    model: nn.Module,
    normalizer: Normalizer,
    params: np.ndarray,
    grid_shape: tuple[int, int] = (6, 8),
) -> np.ndarray:
    """Surrogate IV prediction for one parameter vector or a batch.

    params shape (5,) -> returns (n_mat, n_mon); shape (B, 5) -> (B, n_mat, n_mon).
    """
    arr = np.atleast_2d(np.asarray(params, dtype=float))
    xn = torch.as_tensor(normalizer.norm_x(arr), dtype=torch.float32)
    with torch.no_grad():
        yn = model(xn).numpy()
    iv = normalizer.denorm_y(yn).reshape(-1, *grid_shape)
    if np.asarray(params).ndim == 1:
        return iv[0]
    return iv


def _round_sig(x: np.ndarray, sig: int = 5) -> list:
    """Round to `sig` significant digits for compact JSON export."""
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore"):
        mags = np.where(x == 0.0, 1.0, 10.0 ** (sig - 1 - np.floor(np.log10(np.abs(x)))))
    return (np.round(x * mags) / mags).tolist()


def export_weights_json(
    path: str | Path,
    model: MLPSurrogate,
    normalizer: Normalizer,
    moneyness: np.ndarray,
    maturities: np.ndarray,
    bounds: np.ndarray | None = None,
    metrics: dict | None = None,
) -> Path:
    """Export the MLP to compact JSON (5 significant digits) for the stlite
    dashboard, where torch does not exist and the forward pass is plain numpy."""
    import json

    layers = []
    linears = [m for m in model.net if isinstance(m, nn.Linear)]
    for lin in linears:
        layers.append({
            "W": _round_sig(lin.weight.detach().numpy()),
            "b": _round_sig(lin.bias.detach().numpy()),
        })
    payload = {
        "activation": "silu",
        "layers": layers,
        "x_mean": _round_sig(normalizer.x_mean, 7),
        "x_std": _round_sig(normalizer.x_std, 7),
        "y_mean": _round_sig(normalizer.y_mean, 7),
        "y_std": _round_sig(normalizer.y_std, 7),
        "moneyness": np.asarray(moneyness).tolist(),
        "maturities": np.asarray(maturities).tolist(),
        "bounds": np.asarray(bounds).tolist() if bounds is not None else None,
        "metrics": metrics or {},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")))
    return path


def numpy_forward(weights: dict, params: np.ndarray) -> np.ndarray:
    """Pure-numpy replica of the exported MLP — the dashboard's pricer.

    params: (5,) or (B, 5) raw Heston parameters; returns denormalised IVs.
    """
    x = np.atleast_2d(np.asarray(params, dtype=float))
    h = (x - np.asarray(weights["x_mean"])) / np.asarray(weights["x_std"])
    n_layers = len(weights["layers"])
    for i, layer in enumerate(weights["layers"]):
        h = h @ np.asarray(layer["W"]).T + np.asarray(layer["b"])
        if i < n_layers - 1:
            h = h * (1.0 / (1.0 + np.exp(-h)))  # SiLU = x * sigmoid(x)
    iv = h * np.asarray(weights["y_std"]) + np.asarray(weights["y_mean"])
    return iv if np.asarray(params).ndim > 1 else iv[0]
