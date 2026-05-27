"""Unit tests for the KAN layer."""

from __future__ import annotations

import torch

from models.kan import HierarchicalKAN, KANLayer, _b_spline_basis


def test_b_spline_basis_partition_of_unity():
    """B-spline basis functions sum to 1 inside the inner grid range."""
    knots = torch.linspace(-1.0, 1.0, 20)
    x = torch.linspace(-0.5, 0.5, 50)
    basis = _b_spline_basis(x.unsqueeze(-1), knots, degree=3).squeeze(-2)
    sums = basis.sum(dim=-1)
    # Inside the inner region, the partition-of-unity should be very close to 1.
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


def test_kan_layer_output_shape():
    layer = KANLayer(in_features=12, out_features=24, num_splines=4, num_basis=8)
    x = torch.randn(3, 7, 12)
    y = layer(x)
    assert y.shape == (3, 7, 24)


def test_kan_backward():
    layer = KANLayer(in_features=8, out_features=4, num_splines=4, num_basis=8)
    x = torch.randn(2, 8, requires_grad=True)
    layer(x).sum().backward()
    assert x.grad is not None
    assert layer.coef.grad is not None


def test_hierarchical_kan():
    model = HierarchicalKAN(in_features=16, out_features=4, dropout=0.0)
    x = torch.randn(5, 16)
    y = model(x)
    assert y.shape == (5, 4)
