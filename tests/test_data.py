"""Unit tests for utils.data — Dataset, DataLoader, CSV/synthetic loader."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from utils.data import (
    EnergyDataset,
    build_dataloaders,
    load_or_synthesize,
    prepare,
    synthetic_energy_dataframe,
)
from utils.preprocessing import EXOG_COLS_DEFAULT, LOAD_COLS_DEFAULT, preprocess_energy_dataframe


def _small_synth(periods: int = 24 * 30):
    df = synthetic_energy_dataframe(periods=periods)
    return preprocess_energy_dataframe(df)


def test_synthetic_energy_dataframe_columns_and_dtypes():
    df = synthetic_energy_dataframe(periods=48)
    for c in LOAD_COLS_DEFAULT + EXOG_COLS_DEFAULT:
        assert c in df.columns
    # Hourly index, monotonic.
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.is_monotonic_increasing


def test_energy_dataset_window_and_horizon_shapes():
    splits = _small_synth()
    ds = EnergyDataset(
        splits.train,
        load_cols=splits.load_cols,
        feature_cols=splits.feature_cols,
        window=24,
        horizon=6,
    )
    assert len(ds) == len(splits.train) - 24 - 6 + 1
    x, y = ds[0]
    assert isinstance(x, torch.Tensor)
    assert x.shape == (24, len(splits.load_cols) + len(splits.feature_cols))
    assert y.shape == (6, len(splits.load_cols))


def test_energy_dataset_raises_when_too_small():
    splits = _small_synth(periods=48)
    with pytest.raises(ValueError, match="Not enough samples"):
        EnergyDataset(
            splits.train,
            load_cols=splits.load_cols,
            feature_cols=splits.feature_cols,
            window=200,
            horizon=50,
        )


def test_build_dataloaders_returns_three_loaders():
    splits = _small_synth()
    train, val, test = build_dataloaders(splits, window=24, horizon=6, batch_size=8)
    assert train.batch_size == val.batch_size == test.batch_size == 8
    # train shuffles + drops last; val / test do not.
    xb, yb = next(iter(train))
    assert xb.dim() == 3 and yb.dim() == 3


def test_load_or_synthesize_falls_back_when_no_csv(tmp_path):
    df = load_or_synthesize(tmp_path / "does-not-exist.csv")
    assert "office" in df.columns


def test_load_or_synthesize_reads_csv(tmp_path):
    df = synthetic_energy_dataframe(periods=72)
    csv = tmp_path / "energy.csv"
    df.reset_index().to_csv(csv, index=False)
    loaded = load_or_synthesize(csv)
    assert loaded.shape == df.shape
    assert (loaded.columns == df.columns).all()


def test_load_or_synthesize_raises_on_missing_column(tmp_path):
    df = synthetic_energy_dataframe(periods=24)
    df = df.drop(columns=["occupancy"])  # corrupt the file
    csv = tmp_path / "bad.csv"
    df.reset_index().to_csv(csv, index=False)
    with pytest.raises(ValueError, match="Column missing"):
        load_or_synthesize(csv)


def test_prepare_end_to_end(tmp_path):
    """Smoke check: prepare() with no CSV path goes through synthesis +
    preprocessing + loader construction and returns 3 loaders + splits."""
    train_loader, val_loader, test_loader, splits = prepare(
        csv_path=tmp_path / "missing.csv",
        window=24,
        horizon=6,
        batch_size=8,
    )
    assert splits.load_cols == LOAD_COLS_DEFAULT
    # Each batch must have at least one element.
    for loader in (train_loader, val_loader, test_loader):
        xb, yb = next(iter(loader))
        assert xb.shape[0] > 0
        assert yb.shape[0] > 0


def test_synthetic_dataframe_is_deterministic_for_fixed_seed():
    a = synthetic_energy_dataframe(periods=48, seed=7)
    b = synthetic_energy_dataframe(periods=48, seed=7)
    assert np.allclose(a["office"].values, b["office"].values)
