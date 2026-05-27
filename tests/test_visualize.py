"""Unit tests for utils.visualize plotting functions."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from utils.visualize import (
    plot_method_comparison,
    plot_metric_bar,
    plot_predictions_vs_true,
)


def _toy_data(rng_seed: int = 0):
    rng = np.random.default_rng(rng_seed)
    # (samples, horizon, num_categories)
    y_true = rng.standard_normal((4, 12, 4))
    preds = {
        "transformer": y_true + rng.standard_normal(y_true.shape) * 0.4,
        "transformer_kan": y_true + rng.standard_normal(y_true.shape) * 0.3,
        "proposed": y_true + rng.standard_normal(y_true.shape) * 0.1,
    }
    return y_true, preds


def test_plot_predictions_vs_true_writes_png(tmp_path: Path):
    y_true, preds = _toy_data()
    out = plot_predictions_vs_true(
        y_true, preds["proposed"], out_path=tmp_path / "fig2.png"
    )
    assert out.exists()
    assert out.stat().st_size > 1000  # something real was written


def test_plot_predictions_vs_true_creates_parent_dir(tmp_path: Path):
    y_true, preds = _toy_data()
    target = tmp_path / "sub" / "nested" / "fig.png"
    out = plot_predictions_vs_true(y_true, preds["proposed"], out_path=target)
    assert out.exists()


def test_plot_method_comparison_overlays_all_methods(tmp_path: Path):
    y_true, preds = _toy_data()
    out = plot_method_comparison(
        y_true, preds, out_path=tmp_path / "fig3.png", category=2,
    )
    assert out.exists()


def test_plot_method_comparison_with_missing_proposed_key(tmp_path: Path):
    """Should still draw all baselines even if no 'proposed' entry exists."""
    y_true, preds = _toy_data()
    preds.pop("proposed")
    out = plot_method_comparison(y_true, preds, out_path=tmp_path / "fig3.png")
    assert out.exists()


def test_plot_metric_bar_each_metric(tmp_path: Path):
    results = {
        "transformer": {"MAPE": 0.082, "RMSE": 39.0, "MAE": 16.8, "R2": 0.789},
        "transformer_kan": {"MAPE": 0.076, "RMSE": 31.8, "MAE": 15.2, "R2": 0.796},
        "proposed": {"MAPE": 0.042, "RMSE": 12.7, "MAE": 8.7, "R2": 0.956},
    }
    for metric in ("MAPE", "RMSE", "MAE", "R2"):
        out = plot_metric_bar(results, metric=metric, out_path=tmp_path / f"{metric}.png")
        assert out.exists()


def test_plot_metric_bar_skips_methods_without_metric(tmp_path: Path):
    """A method dict containing 'error' (compare.py uses that on failure)
    should be silently skipped rather than crashing."""
    results = {
        "transformer": {"RMSE": 30.0},
        "broken": {"error": "OOM"},
        "proposed": {"RMSE": 12.7},
    }
    out = plot_metric_bar(results, metric="RMSE", out_path=tmp_path / "bar.png")
    assert out.exists()
