"""Evaluation metrics — MAPE, RMSE, MAE, R²."""

from __future__ import annotations

import numpy as np


def _flatten(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    return y_true, y_pred


def mape(y_true, y_pred, eps: float = 1e-6) -> float:
    y_true, y_pred = _flatten(y_true, y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / np.where(np.abs(y_true) < eps, eps, y_true))))


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = _flatten(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = _flatten(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true, y_pred) -> float:
    y_true, y_pred = _flatten(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def all_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "MAPE": mape(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }
