"""Kolmogorov-Arnold Network (KAN) layer — Section 2.6.

KAN replaces fixed activations with learned spline activations:

    z_i = Σ_{j=1..k} g_{i,j}(w_{i,j}^T x + b_{i,j})                   (12)
    g_{i,j}(s) = Σ_{l=1..L} c_{i,j,l} · B_l(s)                        (13)

Each KAN unit i applies k spline functions g_{i,j} to k linear projections of
x. Each spline is a learned linear combination of L fixed B-spline basis
functions evaluated on a uniform knot grid plus a SiLU residual to ease
optimisation, following Liu et al. (2024).

Paper config: hierarchical 128 → 256 → 128, k=8 splines/unit, L=16 basis.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _b_spline_basis(
    x: torch.Tensor,
    knots: torch.Tensor,
    degree: int = 3,
) -> torch.Tensor:
    """Evaluate the B-spline basis B_l(x) for all l on a 1D knot vector.

    x:     (..., 1) input scalars (any leading shape)
    knots: (n_knots,) sorted knot vector; output is (..., n_knots - degree - 1)
    """
    x = x.unsqueeze(-1)  # (..., 1)
    k = knots
    # Order-1 basis: indicator over [k_i, k_{i+1})
    basis = ((x >= k[:-1]) & (x < k[1:])).to(x.dtype)
    for d in range(1, degree + 1):
        left_den = (k[d:-1] - k[:-d - 1])
        right_den = (k[d + 1:] - k[1:-d])
        # Avoid division by zero at duplicate knots
        left = torch.where(
            left_den == 0,
            torch.zeros_like(left_den),
            (x.squeeze(-1).unsqueeze(-1) - k[:-d - 1]) / left_den,
        ) * basis[..., :-1]
        right = torch.where(
            right_den == 0,
            torch.zeros_like(right_den),
            (k[d + 1:] - x.squeeze(-1).unsqueeze(-1)) / right_den,
        ) * basis[..., 1:]
        basis = left + right
    return basis


class KANLayer(nn.Module):
    """One KAN layer: in_features → out_features, each output is a sum of
    ``num_splines`` learned spline activations applied to linear projections.

    The total basis dimensionality per output unit is
    ``num_splines * num_basis``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_splines: int = 8,
        num_basis: int = 16,
        spline_degree: int = 3,
        grid_range: tuple[float, float] = (-1.0, 1.0),
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_splines = num_splines
        self.num_basis = num_basis
        self.spline_degree = spline_degree

        # Linear projection w_{i,j}^T x + b_{i,j} producing K=num_splines
        # pre-activations per output unit i, so total = out_features * K.
        self.proj = nn.Linear(in_features, out_features * num_splines)

        # Knot grid for the B-spline basis (shared across units).
        lo, hi = grid_range
        n_knots = num_basis + spline_degree + 1
        knots = torch.linspace(lo, hi, n_knots)
        self.register_buffer("knots", knots)

        # Learnable coefficients c_{i,j,l}: shape (out, K, L).
        self.coef = nn.Parameter(
            torch.empty(out_features, num_splines, num_basis)
        )
        nn.init.normal_(self.coef, std=1.0 / math.sqrt(num_basis))

        self.residual = residual
        if residual:
            self.res_lin = nn.Linear(in_features, out_features)

        self.grid_lo, self.grid_hi = grid_range

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features)
        s = self.proj(x)  # (..., out * K)
        out_shape = s.shape[:-1] + (self.out_features, self.num_splines)
        s = s.view(*out_shape)
        # Normalise s into the grid range so the B-spline basis is well-defined.
        s = torch.tanh(s) * (self.grid_hi - 1e-3)
        # Evaluate basis: result shape (..., out, K, L)
        basis = _b_spline_basis(s, self.knots, self.spline_degree)
        # Weighted sum across basis dim → spline output g_{i,j}(s)
        # coef: (out, K, L); broadcast against basis (..., out, K, L)
        spline = (basis * self.coef).sum(dim=-1)  # (..., out, K)
        # Sum the K splines per output unit  → z_i
        z = spline.sum(dim=-1)  # (..., out)
        if self.residual:
            z = z + F.silu(self.res_lin(x))
        return z


class HierarchicalKAN(nn.Module):
    """Three-layer hierarchical KAN with sizes 128 → 256 → 128 (Section 2.6)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden: tuple[int, int, int] = (128, 256, 128),
        num_splines: int = 8,
        num_basis: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        h1, h2, h3 = hidden
        self.l1 = KANLayer(in_features, h1, num_splines, num_basis)
        self.l2 = KANLayer(h1, h2, num_splines, num_basis)
        self.l3 = KANLayer(h2, h3, num_splines, num_basis)
        self.out = nn.Linear(h3, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.l1(x))
        x = self.dropout(self.l2(x))
        x = self.dropout(self.l3(x))
        return self.out(x)
