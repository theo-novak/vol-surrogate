"""Training loop: validation split, early stopping, checkpoint with norm stats."""

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from .losses import surrogate_loss
from .mlp import MLPSurrogate, Normalizer


@dataclass
class TrainConfig:
    hidden: int = 256
    depth: int = 4
    lr: float = 1e-3
    batch_size: int = 512
    max_epochs: int = 60
    patience: int = 8
    val_frac: float = 0.1
    lambda_arb: float = 1.0
    seed: int = 42
    extra: dict = field(default_factory=dict)


def _val_metrics(
    model: nn.Module,
    Xn_val: torch.Tensor,
    iv_val: torch.Tensor,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    m_t: torch.Tensor,
    T_t: torch.Tensor,
    lambda_arb: float,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        iv_pred = model(Xn_val) * y_std + y_mean
        n_mat, n_mon = len(T_t), len(m_t)
        loss, parts = surrogate_loss(
            iv_pred.view(-1, n_mat, n_mon), iv_val.view(-1, n_mat, n_mon),
            m_t, T_t, lambda_arb,
        )
        abs_err = torch.abs(iv_pred - iv_val)
    model.train()
    return {
        "val_loss": float(loss),
        "val_rel_mse": parts["rel_mse"],
        "val_arb": parts["arb_total"],
        "val_mae_bp": float(abs_err.mean()) * 1e4,
        "val_maxae_bp": float(abs_err.max()) * 1e4,
    }


def train_surrogate(
    X: np.ndarray,
    Y: np.ndarray,
    moneyness: np.ndarray,
    maturities: np.ndarray,
    config: TrainConfig | None = None,
    model: nn.Module | None = None,
    verbose: bool = True,
) -> tuple[nn.Module, Normalizer, dict]:
    """Train a surrogate on (params, IV grid) pairs.

    Returns (best model, normalizer, history). history carries per-epoch
    train/val losses, the best-epoch validation metrics, and wall time.
    The relative-MSE part of the loss is computed on *denormalised* implied
    vols so the metric is interpretable; the arbitrage penalty acts on the
    predicted grid reshaped to (n_mat, n_mon).
    """
    cfg = config or TrainConfig()
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    n = len(X)
    idx = rng.permutation(n)
    n_val = max(1, int(n * cfg.val_frac))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    normalizer = Normalizer.fit(X[train_idx], Y[train_idx])
    Xn = torch.as_tensor(normalizer.norm_x(X), dtype=torch.float32)
    iv = torch.as_tensor(Y, dtype=torch.float32)
    y_mean, y_std = normalizer.torch_y_stats()
    m_t = torch.as_tensor(moneyness, dtype=torch.float32)
    T_t = torch.as_tensor(maturities, dtype=torch.float32)
    n_mat, n_mon = len(maturities), len(moneyness)

    if model is None:
        model = MLPSurrogate(
            in_dim=X.shape[1], hidden=cfg.hidden, depth=cfg.depth, out_dim=Y.shape[1]
        )
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)

    Xn_tr, iv_tr = Xn[train_idx], iv[train_idx]
    Xn_val, iv_val = Xn[val_idx], iv[val_idx]

    history: dict = {"train_loss": [], "val_loss": [], "val_rel_mse": []}
    best_val = float("inf")
    best_state = None
    best_metrics: dict[str, float] = {}
    epochs_no_improve = 0
    t0 = time.perf_counter()

    for epoch in range(cfg.max_epochs):
        perm = torch.randperm(len(train_idx))
        epoch_loss, n_batches = 0.0, 0
        for start in range(0, len(perm), cfg.batch_size):
            batch = perm[start:start + cfg.batch_size]
            opt.zero_grad()
            iv_pred = model(Xn_tr[batch]) * y_std + y_mean
            loss, _ = surrogate_loss(
                iv_pred.view(-1, n_mat, n_mon),
                iv_tr[batch].view(-1, n_mat, n_mon),
                m_t, T_t, cfg.lambda_arb,
            )
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach())
            n_batches += 1

        metrics = _val_metrics(model, Xn_val, iv_val, y_mean, y_std, m_t, T_t, cfg.lambda_arb)
        sched.step(metrics["val_loss"])
        history["train_loss"].append(epoch_loss / max(n_batches, 1))
        history["val_loss"].append(metrics["val_loss"])
        history["val_rel_mse"].append(metrics["val_rel_mse"])

        if metrics["val_loss"] < best_val - 1e-9:
            best_val = metrics["val_loss"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_metrics = metrics | {"best_epoch": epoch}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose:
            print(
                f"epoch {epoch + 1:3d}  train {history['train_loss'][-1]:.3e}  "
                f"val_rel_mse {metrics['val_rel_mse']:.3e}  "
                f"val MAE {metrics['val_mae_bp']:.2f} bp", flush=True,
            )
        if epochs_no_improve >= cfg.patience:
            if verbose:
                print(f"early stop at epoch {epoch + 1}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    history["best"] = best_metrics
    history["wall_time_s"] = time.perf_counter() - t0
    history["n_train"], history["n_val"] = len(train_idx), len(val_idx)
    return model, normalizer, history
