"""Unit tests for the DyT layer."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.dyt import DyTLayer


def test_dyt_output_shape():
    d = 32
    sub = nn.Linear(d, d)
    layer = DyTLayer(sub, d_model=d, dropout=0.0)
    x = torch.randn(2, 10, d)
    y = layer(x)
    assert y.shape == x.shape


def test_dyt_gates_are_bounded():
    d = 16
    layer = DyTLayer(nn.Linear(d, d), d_model=d, dropout=0.0)
    x = torch.randn(4, 7, d) * 100
    with torch.no_grad():
        alpha = torch.sigmoid(layer.gate_alpha(layer.ln_x(x)))
        beta = torch.sigmoid(layer.gate_beta(layer.ln_fx(layer.sublayer(x))))
    assert ((alpha >= 0) & (alpha <= 1)).all()
    assert ((beta >= 0) & (beta <= 1)).all()


def test_dyt_backward_flows():
    d = 8
    layer = DyTLayer(nn.Linear(d, d), d_model=d, dropout=0.0)
    x = torch.randn(2, 5, d, requires_grad=True)
    y = layer(x).sum()
    y.backward()
    assert x.grad is not None
    assert all(p.grad is not None for p in layer.parameters())
