"""Training losses: relative MSE on implied vol + static no-arbitrage penalties.

The surrogate predicts implied vols, but arbitrage lives in *prices*: a
surface is arbitrage-free iff call prices are decreasing and convex in strike
(call-spread and butterfly) and total implied variance w = iv^2 * T is
non-decreasing in maturity at fixed moneyness (calendar). The penalties are
soft squared hinges on those three conditions, evaluated through a
differentiable Black-Scholes reprice of the predicted grid — zero on any
clean surface, positive exactly where a static arbitrage appears.
"""

import numpy as np
import torch

_SQRT2 = float(np.sqrt(2.0))


def _phi(x: torch.Tensor) -> torch.Tensor:
    """Standard normal CDF via erf (differentiable in torch)."""
    return 0.5 * (1.0 + torch.erf(x / _SQRT2))


def torch_bs_call(
    iv: torch.Tensor, moneyness: torch.Tensor, maturities: torch.Tensor
) -> torch.Tensor:
    """Black-Scholes call prices on the lattice, S0 = 1, r = 0 (forward terms).

    iv: (..., n_mat, n_mon); moneyness: (n_mon,); maturities: (n_mat,).
    """
    K = moneyness.view(1, -1)
    sqrtT = torch.sqrt(maturities).view(-1, 1)
    sig = torch.clamp(iv, min=1e-4)
    d1 = (-torch.log(K) + 0.5 * sig**2 * sqrtT**2) / (sig * sqrtT)
    d2 = d1 - sig * sqrtT
    return _phi(d1) - K * _phi(d2)


def relative_mse(iv_pred: torch.Tensor, iv_true: torch.Tensor) -> torch.Tensor:
    """Mean squared *relative* IV error: errors at 12 vol count like errors at 40 vol."""
    rel = (iv_pred - iv_true) / torch.clamp(iv_true, min=1e-3)
    return torch.mean(rel**2)


def arbitrage_penalty(
    iv: torch.Tensor, moneyness: torch.Tensor, maturities: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Soft static-arbitrage penalties on a predicted IV grid.

    iv: (B, n_mat, n_mon) implied vols.
    Returns dict with 'call_spread', 'butterfly', 'calendar', 'total'
    (each a scalar mean of squared hinge violations).
    """
    C = torch_bs_call(iv, moneyness, maturities)

    # Call-spread monotonicity: C(K) non-increasing in strike
    spread = torch.relu(C[..., 1:] - C[..., :-1])
    call_spread = torch.mean(spread**2)

    # Butterfly convexity in strike (uniform moneyness spacing on the lattice)
    fly = C[..., :-2] - 2.0 * C[..., 1:-1] + C[..., 2:]
    butterfly = torch.mean(torch.relu(-fly) ** 2)

    # Calendar: total implied variance non-decreasing in maturity
    w = iv**2 * maturities.view(1, -1, 1)
    cal = torch.relu(w[:, :-1, :] - w[:, 1:, :])
    calendar = torch.mean(cal**2)

    total = call_spread + butterfly + calendar
    return {
        "call_spread": call_spread,
        "butterfly": butterfly,
        "calendar": calendar,
        "total": total,
    }


def surrogate_loss(
    iv_pred: torch.Tensor,
    iv_true: torch.Tensor,
    moneyness: torch.Tensor,
    maturities: torch.Tensor,
    lambda_arb: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Relative MSE + lambda * no-arbitrage penalty on the *predicted* grid."""
    rmse = relative_mse(iv_pred, iv_true)
    pen = arbitrage_penalty(iv_pred, moneyness, maturities)
    loss = rmse + lambda_arb * pen["total"]
    parts = {
        "rel_mse": float(rmse.detach()),
        "arb_total": float(pen["total"].detach()),
    }
    return loss, parts
