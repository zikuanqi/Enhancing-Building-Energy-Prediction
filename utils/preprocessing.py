"""Data preprocessing for the KAN-Transformer building energy prediction.

Implements Section 2.3 of the paper:
  - Missing value handling via linear interpolation (Eq. 2)
  - Anomaly detection and replacement
  - Min-max normalization (Eq. 3)
  - Feature engineering (temporal encoding, rolling stats, HDD/CDD,
    occupancy-weighted demand)
  - Temporal 70 / 10 / 20 train / val / test split
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

LOAD_COLS_DEFAULT = ["office", "residential", "commercial", "industrial"]
EXOG_COLS_DEFAULT = ["temperature", "humidity", "precipitation", "wind_speed", "occupancy"]


def linear_interpolate_missing(series: pd.Series) -> pd.Series:
    """Equation (2): x_t = x_{t-k} + (t-(t-k))/((t+m)-(t-k)) * (x_{t+m} - x_{t-k}).

    pandas's `interpolate(method="linear")` implements exactly this formula
    using positional indices, treating ``NaN`` as the missing markers.
    """
    return series.interpolate(method="linear", limit_direction="both")


def detect_and_clean_anomalies(
    series: pd.Series,
    z_threshold: float = 3.0,
    consecutive_window: int = 3,
) -> pd.Series:
    """Mark anomalies via robust z-score then either interpolate (consecutive
    runs >= ``consecutive_window``) or replace by neighbour mean (isolated).

    A point is anomalous if |x - median| / (1.4826 * MAD) > z_threshold.
    """
    x = series.astype(float).copy()
    med = np.nanmedian(x.values)
    mad = np.nanmedian(np.abs(x.values - med))
    scale = 1.4826 * mad if mad > 0 else (np.nanstd(x.values) + 1e-9)
    z = np.abs((x.values - med) / scale)
    mask = z > z_threshold

    # Group consecutive anomalies
    run_id = (mask != np.r_[False, mask[:-1]]).cumsum()
    run_lengths = pd.Series(mask).groupby(run_id).transform("sum").values

    interp_mask = mask & (run_lengths >= consecutive_window)
    isolated_mask = mask & ~interp_mask

    # Consecutive runs → set to NaN then linear-interpolate
    x[interp_mask] = np.nan
    x = linear_interpolate_missing(x)

    # Isolated anomalies → replace with mean of left/right neighbour
    idx = np.where(isolated_mask)[0]
    arr = x.values.copy()
    for i in idx:
        left = arr[i - 1] if i - 1 >= 0 else arr[i]
        right = arr[i + 1] if i + 1 < len(arr) else arr[i]
        arr[i] = 0.5 * (left + right)
    return pd.Series(arr, index=series.index, name=series.name)


def minmax_normalize(
    df: pd.DataFrame,
    cols: Iterable[str],
    ranges: dict[str, tuple[float, float]] | None = None,
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    """Equation (3): x_norm = (x - x_min) / (x_max - x_min) ∈ [0, 1].

    If ``ranges`` is provided, those (min, max) are reused (e.g. apply train
    ranges to validation / test). Otherwise computed per column.
    """
    df = df.copy()
    out_ranges: dict[str, tuple[float, float]] = {}
    for c in cols:
        if ranges is not None and c in ranges:
            lo, hi = ranges[c]
        else:
            lo = float(df[c].min())
            hi = float(df[c].max())
        denom = hi - lo if hi > lo else 1.0
        df[c] = (df[c] - lo) / denom
        out_ranges[c] = (lo, hi)
    return df, out_ranges


def temporal_encoding(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Hour-of-day, day-of-week, month, season encoded with sin/cos pairs
    so the cyclic boundary (e.g. 23h → 0h) is smooth."""
    hour = idx.hour.values
    dow = idx.dayofweek.values
    month = idx.month.values
    season = ((idx.month.values % 12) // 3)  # 0 winter, 1 spring, 2 summer, 3 autumn

    def cyc(values: np.ndarray, period: int) -> tuple[np.ndarray, np.ndarray]:
        radians = 2 * np.pi * values / period
        return np.sin(radians), np.cos(radians)

    hsin, hcos = cyc(hour, 24)
    dsin, dcos = cyc(dow, 7)
    msin, mcos = cyc(month, 12)
    ssin, scos = cyc(season, 4)
    return pd.DataFrame(
        {
            "hour_sin": hsin, "hour_cos": hcos,
            "dow_sin": dsin, "dow_cos": dcos,
            "month_sin": msin, "month_cos": mcos,
            "season_sin": ssin, "season_cos": scos,
        },
        index=idx,
    )


def rolling_statistics(
    df: pd.DataFrame,
    cols: Iterable[str],
    windows: Iterable[int] = (24, 168),
) -> pd.DataFrame:
    """Per-column rolling mean, variance and max over given window sizes
    (default: 24h and 168h = 1 week)."""
    out = {}
    for c in cols:
        s = df[c]
        for w in windows:
            roll = s.rolling(window=w, min_periods=1)
            out[f"{c}_rmean_{w}"] = roll.mean().values
            out[f"{c}_rvar_{w}"] = roll.var(ddof=0).fillna(0.0).values
            out[f"{c}_rmax_{w}"] = roll.max().values
    return pd.DataFrame(out, index=df.index)


def degree_days_and_demand(
    df: pd.DataFrame,
    base_temp: float = 18.0,
    temp_col: str = "temperature",
    occupancy_col: str = "occupancy",
    load_cols: Iterable[str] = LOAD_COLS_DEFAULT,
) -> pd.DataFrame:
    """Heating / cooling degree days plus occupancy-weighted demand.

    HDD = max(base - T, 0), CDD = max(T - base, 0).
    occupancy_weighted_demand_k = occupancy * load_k (interaction feature).
    """
    t = df[temp_col].values
    hdd = np.maximum(base_temp - t, 0.0)
    cdd = np.maximum(t - base_temp, 0.0)
    out = {"hdd": hdd, "cdd": cdd}
    occ = df[occupancy_col].values
    for c in load_cols:
        if c in df.columns:
            out[f"{c}_owd"] = occ * df[c].values
    return pd.DataFrame(out, index=df.index)


@dataclass
class PreprocessedSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    feature_cols: list[str]
    load_cols: list[str]
    exog_cols: list[str]
    norm_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)


def preprocess_energy_dataframe(
    df: pd.DataFrame,
    load_cols: list[str] | None = None,
    exog_cols: list[str] | None = None,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    rolling_windows: tuple[int, ...] = (24, 168),
) -> PreprocessedSplits:
    """End-to-end preprocessing producing temporally ordered 70 / 10 / 20 splits.

    Normalization ranges are fitted on the training portion only and reused
    for validation / test to avoid leakage.
    """
    load_cols = load_cols or LOAD_COLS_DEFAULT
    exog_cols = exog_cols or EXOG_COLS_DEFAULT

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Input dataframe must have a DatetimeIndex")
    df = df.sort_index().copy()

    base_cols = list(load_cols) + list(exog_cols)
    for c in base_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
        df[c] = linear_interpolate_missing(df[c])
        df[c] = detect_and_clean_anomalies(df[c])

    df = pd.concat(
        [
            df,
            temporal_encoding(df.index),
            rolling_statistics(df, load_cols, rolling_windows),
            degree_days_and_demand(df, load_cols=load_cols),
        ],
        axis=1,
    )

    feature_cols = [c for c in df.columns if c not in load_cols]

    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val:].copy()

    train_df, ranges = minmax_normalize(train_df, load_cols + feature_cols)
    val_df, _ = minmax_normalize(val_df, load_cols + feature_cols, ranges=ranges)
    test_df, _ = minmax_normalize(test_df, load_cols + feature_cols, ranges=ranges)

    return PreprocessedSplits(
        train=train_df,
        val=val_df,
        test=test_df,
        feature_cols=feature_cols,
        load_cols=list(load_cols),
        exog_cols=list(exog_cols),
        norm_ranges=ranges,
    )
