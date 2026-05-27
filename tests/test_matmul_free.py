"""Unit tests for the MatMul-free dense layer."""

from __future__ import annotations

import torch

from models.matmul_free import MatMulFreeDense, ternary_quantize


def test_ternary_values_in_set():
    w = torch.randn(64, 32)
    q = ternary_quantize(w, alpha=0.7)
    unique = q.unique().tolist()
    assert set(unique).issubset({-1.0, 0.0, 1.0})


def test_dense_output_shape():
    layer = MatMulFreeDense(16, 32)
    x = torch.randn(8, 16)
    y = layer(x)
    assert y.shape == (8, 32)


def test_ternary_accumulate_matches_forward_numerically():
    """Eq. (10) — additive form must agree with the standard matmul form
    when both use the same quantized weight."""
    layer = MatMulFreeDense(12, 6)
    layer.eval()
    x = torch.randn(4, 12)
    a = layer(x)
    b = layer.ternary_accumulate(x)
    assert torch.allclose(a, b, atol=1e-5)


def test_straight_through_gradient_exists():
    layer = MatMulFreeDense(10, 5)
    x = torch.randn(3, 10, requires_grad=True)
    layer(x).sum().backward()
    assert x.grad is not None
    assert layer.weight.grad is not None
