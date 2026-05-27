"""Unit tests for utils.metrics."""

from __future__ import annotations

import numpy as np

from utils.metrics import all_metrics, mae, mape, r2_score, rmse


def test_perfect_prediction_zero_error():
    rng = np.random.default_rng(0)
    y = rng.standard_normal((10, 5))
    assert mape(y, y) < 1e-6
    assert rmse(y, y) < 1e-6
    assert mae(y, y) < 1e-6
    assert r2_score(y, y) > 0.999


def test_constant_offset_is_mae():
    y = np.ones((4, 3))
    assert mae(y, y + 0.5) == 0.5
    assert rmse(y, y + 0.5) == 0.5


def test_r2_negative_when_worse_than_mean():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    pred_worse_than_mean = np.full_like(y, 100.0)
    assert r2_score(y, pred_worse_than_mean) < 0


def test_all_metrics_returns_all_keys():
    rng = np.random.default_rng(0)
    y = rng.standard_normal((8, 4))
    out = all_metrics(y, y + 0.1)
    assert set(out.keys()) == {"MAPE", "RMSE", "MAE", "R2"}
