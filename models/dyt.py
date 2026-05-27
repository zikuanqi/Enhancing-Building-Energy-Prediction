"""Dynamic Transformer (DyT) layer — Section 2.4.

Replaces the conventional `Add & Norm` residual with a learned weighted
combination:

    Y_t = α_t * X_t + β_t * F_t(X_t)                                  (4)

    α_t = σ(W_α · LayerNorm(X_t) + b_α)                                (5)
    β_t = σ(W_β · LayerNorm(F_t(X_t)) + b_β)                           (6)

where F_t is the wrapped sub-layer (self-attention or feed-forward). The
gates α_t and β_t are produced per token and per channel.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DyTLayer(nn.Module):
    """Wrap a Transformer sub-layer with a dynamic (gated) residual."""

    def __init__(self, sublayer: nn.Module, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.sublayer = sublayer
        self.ln_x = nn.LayerNorm(d_model)
        self.ln_fx = nn.LayerNorm(d_model)
        self.gate_alpha = nn.Linear(d_model, d_model)
        self.gate_beta = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """x: (B, T, d_model). Extra positional args / kwargs are forwarded
        to the wrapped sub-layer (e.g. attention masks)."""
        fx = self.sublayer(x, *args, **kwargs)
        if isinstance(fx, tuple):
            fx = fx[0]
        alpha = torch.sigmoid(self.gate_alpha(self.ln_x(x)))
        beta = torch.sigmoid(self.gate_beta(self.ln_fx(fx)))
        return self.dropout(alpha * x + beta * fx)
