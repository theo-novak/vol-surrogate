"""Transformer-style alternative surrogate for the MLP-vs-attention comparison.

Each of the 48 lattice nodes becomes a token: a learned positional embedding
(which node am I?) plus a projection of the five Heston parameters (what
world am I in?). Two pre-norm self-attention blocks let nodes share
information across the surface before a per-token head reads out implied vol.
Smile shapes are strongly coupled across strikes and maturities, so attention
is a natural prior — whether it beats a plain MLP at this size is exactly
what compare.py measures.
"""

import torch
from torch import nn


class GridTransformer(nn.Module):
    def __init__(
        self,
        in_dim: int = 5,
        n_tokens: int = 48,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
    ) -> None:
        super().__init__()
        self.n_tokens = n_tokens
        self.param_proj = nn.Linear(in_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, n_tokens, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)
        self.in_dim, self.out_dim = in_dim, n_tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, 5) -> (B, n_tokens, d_model): every token sees the same params
        tokens = self.pos_emb + self.param_proj(x).unsqueeze(1)
        encoded = self.encoder(tokens)
        return self.head(encoded).squeeze(-1)  # (B, n_tokens)
