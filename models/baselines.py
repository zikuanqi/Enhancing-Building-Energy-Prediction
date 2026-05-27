"""Comparative baseline models — corresponds to Section 3.1 method list:

    Transformer
    Transformer-KAN
    Transformer-DyT
    Transformer-MatMul-free
    Transformer-KAN-MatMul-free
    Transformer-KAN-DyT
    Transformer-DyT-MatMul-free
    CNN-LSTM
    LSTM-Attention

The Transformer variants share a common encoder skeleton; each ablation flips
on / off the KAN feed-forward, the DyT residual, and the MatMul-free output
head. This makes the contribution of each component directly attributable in
Table 1 of the paper.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .dyt import DyTLayer
from .kan import HierarchicalKAN
from .matmul_free import MatMulFreeDense


# -------- shared utilities --------------------------------------------------


class _PosEnc(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class _AttnSub(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

    def forward(self, x):
        out, _ = self.mha(x, x, x, need_weights=False)
        return out


class _MLPSub(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class _KANFFN(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.ffn = HierarchicalKAN(d_model, d_model, dropout=dropout)

    def forward(self, x):
        return self.ffn(x)


class _AddNorm(nn.Module):
    """Standard Add & Norm residual — used when DyT is disabled."""

    def __init__(self, sublayer: nn.Module, d_model: int, dropout: float):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, *args, **kwargs):
        return self.norm(x + self.dropout(self.sublayer(x, *args, **kwargs)))


def _wrap(sublayer, d_model, dropout, use_dyt: bool):
    return DyTLayer(sublayer, d_model, dropout) if use_dyt else _AddNorm(sublayer, d_model, dropout)


# -------- configurable Transformer encoder ----------------------------------


class _ConfigurableTransformer(nn.Module):
    """One backbone, four feature flags.

    - ``use_kan``        : KAN feed-forward instead of MLP
    - ``use_dyt``        : DyT residual instead of Add & Norm
    - ``use_matmul_free``: ternary MatMul-free output head
    """

    def __init__(
        self,
        input_dim: int,
        num_targets: int,
        horizon: int,
        d_model: int = 128,
        n_heads: int = 8,
        d_ff: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_kan: bool = False,
        use_dyt: bool = False,
        use_matmul_free: bool = False,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.num_targets = num_targets
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = _PosEnc(d_model)

        layers = []
        for _ in range(num_layers):
            attn = _wrap(_AttnSub(d_model, n_heads, dropout), d_model, dropout, use_dyt)
            ffn_sub = _KANFFN(d_model, dropout) if use_kan else _MLPSub(d_model, d_ff, dropout)
            ffn = _wrap(ffn_sub, d_model, dropout, use_dyt)
            layers.append(nn.ModuleList([attn, ffn]))
        self.layers = nn.ModuleList(layers)

        head_dim = horizon * num_targets
        self.head = MatMulFreeDense(d_model, head_dim) if use_matmul_free else nn.Linear(d_model, head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pos_enc(self.input_proj(x))
        for attn, ffn in self.layers:
            h = attn(h)
            h = ffn(h)
        h = h.mean(dim=1)
        return self.head(h).view(-1, self.horizon, self.num_targets)


# -------- named ablations exported to compare.py ---------------------------


class PlainTransformer(_ConfigurableTransformer):
    def __init__(self, input_dim, num_targets=4, horizon=24, **kw):
        super().__init__(input_dim, num_targets, horizon, **kw)


class TransformerKAN(_ConfigurableTransformer):
    def __init__(self, input_dim, num_targets=4, horizon=24, **kw):
        super().__init__(input_dim, num_targets, horizon, use_kan=True, **kw)


class TransformerDyT(_ConfigurableTransformer):
    def __init__(self, input_dim, num_targets=4, horizon=24, **kw):
        super().__init__(input_dim, num_targets, horizon, use_dyt=True, **kw)


class TransformerMatMulFree(_ConfigurableTransformer):
    def __init__(self, input_dim, num_targets=4, horizon=24, **kw):
        super().__init__(input_dim, num_targets, horizon, use_matmul_free=True, **kw)


class TransformerKANMatMulFree(_ConfigurableTransformer):
    def __init__(self, input_dim, num_targets=4, horizon=24, **kw):
        super().__init__(
            input_dim, num_targets, horizon, use_kan=True, use_matmul_free=True, **kw
        )


class TransformerKANDyT(_ConfigurableTransformer):
    def __init__(self, input_dim, num_targets=4, horizon=24, **kw):
        super().__init__(
            input_dim, num_targets, horizon, use_kan=True, use_dyt=True, **kw
        )


class TransformerDyTMatMulFree(_ConfigurableTransformer):
    def __init__(self, input_dim, num_targets=4, horizon=24, **kw):
        super().__init__(
            input_dim, num_targets, horizon, use_dyt=True, use_matmul_free=True, **kw
        )


# -------- recurrent baselines ----------------------------------------------


class CNNLSTM(nn.Module):
    """Conv1D feature extractor → stacked LSTM → projection head."""

    def __init__(
        self,
        input_dim: int,
        num_targets: int = 4,
        horizon: int = 24,
        conv_channels: int = 64,
        kernel_size: int = 5,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.num_targets = num_targets
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, conv_channels, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.head = nn.Linear(lstm_hidden, horizon * num_targets)

    def forward(self, x):
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        out, _ = self.lstm(h)
        last = out[:, -1, :]
        return self.head(last).view(-1, self.horizon, self.num_targets)


class LSTMAttention(nn.Module):
    """LSTM encoder with additive attention over hidden states."""

    def __init__(
        self,
        input_dim: int,
        num_targets: int = 4,
        horizon: int = 24,
        hidden: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.num_targets = num_targets
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn_w = nn.Linear(hidden, hidden)
        self.attn_v = nn.Linear(hidden, 1, bias=False)
        self.head = nn.Linear(hidden, horizon * num_targets)

    def forward(self, x):
        out, _ = self.lstm(x)              # (B, T, H)
        scores = self.attn_v(torch.tanh(self.attn_w(out)))  # (B, T, 1)
        weights = torch.softmax(scores, dim=1)
        context = (weights * out).sum(dim=1)               # (B, H)
        return self.head(context).view(-1, self.horizon, self.num_targets)
