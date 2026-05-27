"""Unit tests for utils.preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.preprocessing import (
    detect_and_clean_anomalies,
    linear_interpolate_missing,
    minmax_normalize,
    preprocess_energy_dataframe,
    temporal_encoding,
)
from utils.data import synthetic_energy_dataframe


def test_linear_interpolate_fills_nan():
    s = pd.Series([1.0, np.nan, 3.0, np.nan, np.nan, 6.0])
    out = linear_interpolate_missing(s)
    assert not out.isna().any()
    # Eq. (2): linear → midpoint between 1 and 3 = 2
    assert abs(out.iloc[1] - 2.0) < 1e-6


def test_minmax_normalize_in_unit_range():
    df = pd.DataFrame({"x": np.linspace(10.0, 50.0, 20)})
    out, ranges = minmax_normalize(df, ["x"])
    assert out["x"].min() == 0.0
    assert out["x"].max() == 1.0
    assert ranges["x"] == (10.0, 50.0)


def test_temporal_encoding_shape_and_range():
    idx = pd.date_range("2024-01-01", periods=24 * 7, freq="h")
    enc = temporal_encoding(idx)
    assert len(enc) == len(idx)
    assert (enc.values >= -1.0).all() and (enc.values <= 1.0).all()


def test_anomaly_replacement():
    s = pd.Series(np.r_[np.zeros(50), [10_000.0], np.zeros(50)])
    out = detect_and_clean_anomalies(s)
    # Isolated spike should be neutralised (back near zero).
    assert abs(out.iloc[50]) < 1.0


def test_preprocess_split_ratios():
    df = synthetic_energy_dataframe(periods=24 * 30)
    splits = preprocess_energy_dataframe(df)
    n = len(df)
    assert abs(len(splits.train) - int(n * 0.7)) <= 1
    assert abs(len(splits.val) - int(n * 0.1)) <= 1
    # Normalized columns should sit roughly in [0, 1] on train.
    for c in splits.load_cols:
        assert splits.train[c].between(-0.01, 1.01).all()
