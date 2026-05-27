"""Unit tests for the multi-scale temporal decomposition."""

from __future__ import annotations

import torch

from utils.decomposition import MultiScaleDecomposition


def test_components_reconstruct_input():
    """trend + seasonal + weekly + daily + short_term should equal input."""
    decomp = MultiScaleDecomposition()
    x = torch.randn(2, 256, 4)
    parts = decomp(x)
    recon = sum(parts.values())
    assert torch.allclose(recon, x, atol=1e-4)


def test_components_have_same_shape_as_input():
    decomp = MultiScaleDecomposition()
    x = torch.randn(3, 100, 6)
    parts = decomp(x)
    for v in parts.values():
        assert v.shape == x.shape


def test_short_window_does_not_crash():
    """The decomposition kernel sizes are clamped so the smallest sample
    in compare.py / quick.yaml still produces valid output."""
    decomp = MultiScaleDecomposition()
    x = torch.randn(1, 8, 4)
    parts = decomp(x)
    assert all(v.shape == x.shape for v in parts.values())
