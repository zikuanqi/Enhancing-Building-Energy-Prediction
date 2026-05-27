"""Dataset / DataLoader utilities and a synthetic data generator.

The paper uses multivariate hourly time series with four load categories
(office, residential, commercial, industrial) plus five exogenous features
(temperature, humidity, precipitation, wind speed, occupancy). When no real
file is available, ``synthetic_energy_dataframe`` builds a plausible mock.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .preprocessing import (
    EXOG_COLS_DEFAULT,
    LOAD_COLS_DEFAULT,
    PreprocessedSplits,
    preprocess_energy_dataframe,
)


class EnergyDataset(Dataset):
    """Sliding-window dataset yielding (X, y) pairs.

    X: (window, 4 + num_features) — load + exogenous + engineered features
    y: (horizon, 4) — future loads
    """

    def __init__(
        self,
        df: pd.DataFrame,
        load_cols: list[str],
        feature_cols: list[str],
        window: int = 168,
        horizon: int = 24,
    ) -> None:
        super().__init__()
        self.window = window
        self.horizon = horizon
        self.load_cols = load_cols
        self.feature_cols = feature_cols
        self.loads = df[load_cols].values.astype(np.float32)
        self.feats = df[feature_cols].values.astype(np.float32)
        self.length = len(df) - window - horizon + 1
        if self.length <= 0:
            raise ValueError(
                f"Not enough samples ({len(df)}) for window={window}, horizon={horizon}"
            )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        s = idx
        e = idx + self.window
        x = np.concatenate(
            [self.loads[s:e], self.feats[s:e]], axis=-1
        )
        y = self.loads[e:e + self.horizon]
        return torch.from_numpy(x), torch.from_numpy(y)


def build_dataloaders(
    splits: PreprocessedSplits,
    window: int = 168,
    horizon: int = 24,
    batch_size: int = 64,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    common = dict(
        load_cols=splits.load_cols,
        feature_cols=splits.feature_cols,
        window=window,
        horizon=horizon,
    )
    train_ds = EnergyDataset(splits.train, **common)
    val_ds = EnergyDataset(splits.val, **common)
    test_ds = EnergyDataset(splits.test, **common)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )


def synthetic_energy_dataframe(
    start: str = "2022-01-01",
    periods: int = 24 * 365 * 2,
    freq: str = "h",
    seed: int = 42,
) -> pd.DataFrame:
    """Synthesize a 2-year hourly dataset with daily / weekly / seasonal
    energy patterns and correlated weather + occupancy features."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, periods=periods, freq=freq)
    n = len(idx)

    hours = idx.hour.values
    dow = idx.dayofweek.values
    day_of_year = idx.dayofyear.values

    # Seasonal temperature (°C): warmer mid-year (NH)
    temperature = 12 + 12 * np.sin(2 * np.pi * (day_of_year - 81) / 365) \
        + 5 * np.sin(2 * np.pi * hours / 24 - np.pi / 2) + rng.normal(0, 1.5, n)
    humidity = 60 + 15 * np.sin(2 * np.pi * (day_of_year + 30) / 365) - 0.4 * (temperature - 12) \
        + rng.normal(0, 4, n)
    humidity = np.clip(humidity, 5, 100)
    precipitation = np.clip(rng.gamma(0.4, 1.0, n) - 0.1, 0, None)
    wind_speed = np.clip(3 + 2 * rng.standard_normal(n) + 1.5 * np.sin(2 * np.pi * day_of_year / 365), 0, None)
    occupancy = 0.2 + 0.7 * ((dow < 5) & (hours >= 8) & (hours <= 18)).astype(float) \
        + rng.normal(0, 0.05, n)
    occupancy = np.clip(occupancy, 0, 1)

    def load(base, day_amp, week_amp, season_amp, t_coef, occ_coef, noise):
        daily = day_amp * np.sin(2 * np.pi * hours / 24 - np.pi / 3)
        weekly = week_amp * np.cos(2 * np.pi * dow / 7)
        seasonal = season_amp * np.sin(2 * np.pi * (day_of_year - 30) / 365)
        return (
            base
            + daily
            + weekly
            + seasonal
            + t_coef * np.maximum(temperature - 22, 0)        # cooling
            + 0.5 * t_coef * np.maximum(18 - temperature, 0)  # heating
            + occ_coef * occupancy
            + rng.normal(0, noise, n)
        )

    office = load(180, 60, 20, 40, 3.5, 80, 6)
    residential = load(140, 30, 5, 50, 2.0, 30, 5)
    commercial = load(220, 70, 25, 45, 4.0, 90, 8)
    industrial = load(260, 25, 15, 30, 2.5, 40, 7)

    df = pd.DataFrame(
        {
            "office": office,
            "residential": residential,
            "commercial": commercial,
            "industrial": industrial,
            "temperature": temperature,
            "humidity": humidity,
            "precipitation": precipitation,
            "wind_speed": wind_speed,
            "occupancy": occupancy,
        },
        index=idx,
    )
    df.index.name = "timestamp"
    return df


def load_or_synthesize(
    csv_path: str | Path | None = None,
    **synth_kwargs,
) -> pd.DataFrame:
    """Load ``csv_path`` if it exists (timestamp index expected) else synthesize."""
    if csv_path is not None and Path(csv_path).exists():
        df = pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp")
        for c in LOAD_COLS_DEFAULT + EXOG_COLS_DEFAULT:
            if c not in df.columns:
                raise ValueError(f"Column missing in {csv_path}: {c}")
        return df
    return synthetic_energy_dataframe(**synth_kwargs)


def prepare(
    csv_path: str | Path | None = None,
    window: int = 168,
    horizon: int = 24,
    batch_size: int = 64,
) -> tuple[DataLoader, DataLoader, DataLoader, PreprocessedSplits]:
    df = load_or_synthesize(csv_path)
    splits = preprocess_energy_dataframe(df)
    loaders = build_dataloaders(splits, window=window, horizon=horizon, batch_size=batch_size)
    return (*loaders, splits)
