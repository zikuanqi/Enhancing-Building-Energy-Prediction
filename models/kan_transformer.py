"""KAN-Transformer — the proposed network (Section 2.2).

Architecture:
  1. Multi-scale temporal decomposition (utils.decomposition) — input is
     decomposed into trend / seasonal / weekly / daily / short-term streams.
  2. Per-component embedding into d_model + sinusoidal positional encoding.
  3. Component-aware fusion via cross-component attention.
  4. N × KAN-Transformer encoder block:
        - DyT-wrapped multi-head self-attention
        - DyT-wrapped Hierarchical KAN feed-forward
  5. MatMul-free dense projection head producing (horizon × num_targets).

Personalized learning strategy (Section 1 contribution #3): a shared trunk
followed by light building-specific heads — exposed via
``building_id`` argument so the same model can be quickly fine-tuned for a
new building with minimal data.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from utils.decomposition import MultiScaleDecomposition

from .dyt import DyTLayer
from .kan import HierarchicalKAN
from .matmul_free import MatMulFreeDense


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class _SelfAttnSubLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.mha(x, x, x, need_weights=False)
        return out


class _KANFFNSubLayer(nn.Module):
    def __init__(self, d_model: int, hidden: tuple[int, int, int], num_splines: int, num_basis: int, dropout: float):
        super().__init__()
        self.ffn = HierarchicalKAN(
            in_features=d_model,
            out_features=d_model,
            hidden=hidden,
            num_splines=num_splines,
            num_basis=num_basis,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


class KANTransformerBlock(nn.Module):
    """One encoder block: DyT(self-attn) → DyT(KAN-FFN)."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        kan_hidden: tuple[int, int, int],
        num_splines: int,
        num_basis: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attn = DyTLayer(
            _SelfAttnSubLayer(d_model, n_heads, dropout), d_model, dropout
        )
        self.ffn = DyTLayer(
            _KANFFNSubLayer(d_model, kan_hidden, num_splines, num_basis, dropout),
            d_model,
            dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(x)
        x = self.ffn(x)
        return x


class ComponentEmbedding(nn.Module):
    """Embed each multi-scale component separately and fuse along the
    component axis via attention so the network can re-weight scales."""

    COMPONENTS = ("trend", "seasonal", "weekly", "daily", "short_term")

    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        self.embeds = nn.ModuleDict(
            {c: nn.Linear(input_dim, d_model) for c in self.COMPONENTS}
        )
        self.scale_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, components: dict[str, torch.Tensor]) -> torch.Tensor:
        # Each component: (B, T, input_dim). Embed → (B, T, d_model).
        # Stack along a "scale" axis then attend over it for each token.
        embedded = [self.embeds[c](components[c]) for c in self.COMPONENTS]
        stacked = torch.stack(embedded, dim=2)  # (B, T, S, d_model)
        B, T, S, D = stacked.shape
        flat = stacked.reshape(B * T, S, D)
        fused, _ = self.scale_attn(flat, flat, flat, need_weights=False)
        fused = fused.mean(dim=1).reshape(B, T, D)
        return self.proj(fused)


class KANTransformer(nn.Module):
    """The proposed KAN-Transformer network.

    Parameters
    ----------
    input_dim
        Feature dimension of each timestep in the input window (loads + exog
        + engineered features).
    num_targets
        Number of building categories to predict (default 4).
    horizon
        Number of future timesteps to predict.
    d_model, n_heads, num_layers
        Standard Transformer hyper-parameters.
    kan_hidden, num_splines, num_basis
        KAN feed-forward configuration (default matches Section 2.6).
    num_buildings
        If > 1, enable personalized per-building output heads.
    """

    def __init__(
        self,
        input_dim: int,
        num_targets: int = 4,
        horizon: int = 24,
        d_model: int = 128,
        n_heads: int = 8,
        num_layers: int = 3,
        kan_hidden: tuple[int, int, int] = (128, 256, 128),
        num_splines: int = 8,
        num_basis: int = 16,
        dropout: float = 0.1,
        num_buildings: int = 1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_targets = num_targets
        self.horizon = horizon
        self.num_buildings = num_buildings

        self.decomp = MultiScaleDecomposition()
        self.embedding = ComponentEmbedding(input_dim, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model)
        self.blocks = nn.ModuleList([
            KANTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                kan_hidden=kan_hidden,
                num_splines=num_splines,
                num_basis=num_basis,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.shared_head = MatMulFreeDense(d_model, horizon * num_targets)
        if num_buildings > 1:
            # Personalized: small per-building residual heads.
            self.personal_heads = nn.ModuleList([
                MatMulFreeDense(d_model, horizon * num_targets)
                for _ in range(num_buildings)
            ])
        else:
            self.personal_heads = None

    def forward(
        self,
        x: torch.Tensor,
        building_id: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """x: (B, T, input_dim). Returns (B, horizon, num_targets)."""
        components = self.decomp(x)
        h = self.embedding(components)
        h = self.pos_enc(h)
        for blk in self.blocks:
            h = blk(h)
        # Aggregate temporal dim → (B, d_model). Mean pool emphasizes
        # multi-scale patterns; the decomposition has already disentangled
        # scales, so a simple mean works well.
        h = h.mean(dim=1)
        y = self.shared_head(h)
        if self.personal_heads is not None and building_id is not None:
            if isinstance(building_id, int):
                y = y + self.personal_heads[building_id](h)
            else:
                # Per-sample building ids.
                stacked = torch.stack(
                    [head(h) for head in self.personal_heads], dim=1
                )  # (B, num_buildings, horizon*num_targets)
                pick = building_id.long().view(-1, 1, 1).expand(-1, 1, stacked.size(-1))
                y = y + stacked.gather(1, pick).squeeze(1)
        return y.view(-1, self.horizon, self.num_targets)
