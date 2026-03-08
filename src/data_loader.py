from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class DataConfig:
    """Configuration for the demand data pipeline."""

    raw_dir: Path
    processed_dir: Path
    processed_file: str = "demand_features.parquet"
    date_column: str = "date"
    demand_column: str = "sales"
    sku_column: str = "item_id"
    store_column: Optional[str] = "store_id"
    promo_column: Optional[str] = "onpromotion"
    frequency: str = "D"


def load_raw_data(csv_paths: Sequence[Path]) -> pd.DataFrame:
    """Load one or more raw CSV files containing transactional or daily sales data.

    Expected minimal columns:
    - date
    - item_id (SKU)
    - sales (units sold)

    Additional columns such as store_id or onpromotion are optional and will be
    preserved if present.
    """
    frames: list[pd.DataFrame] = []
    for path in csv_paths:
        df = pd.read_csv(path)
        frames.append(df)
    if not frames:
        raise ValueError("No raw data files provided.")
    raw = pd.concat(frames, ignore_index=True)
    return raw


def clean_and_aggregate(
    raw: pd.DataFrame, config: DataConfig, freq: Optional[str] = None
) -> pd.DataFrame:
    """Clean raw sales data and aggregate to time series by SKU (and optionally store).

    Steps:
    1. Parse dates and sort.
    2. Handle missing values in demand.
    3. Aggregate to the requested frequency (daily by default).
    """
    freq = freq or config.frequency

    df = raw.copy()
    df[config.date_column] = pd.to_datetime(df[config.date_column])

    # Coerce demand to numeric and impute missing/invalid with zero.
    df[config.demand_column] = pd.to_numeric(df[config.demand_column], errors="coerce")
    df[config.demand_column] = df[config.demand_column].fillna(0.0)
    df.loc[df[config.demand_column] < 0, config.demand_column] = 0.0

    group_cols = [config.date_column, config.sku_column]
    if config.store_column and config.store_column in df.columns:
        group_cols.append(config.store_column)

    agg = (
        df.groupby(group_cols, as_index=False)[config.demand_column]
        .sum()
        .sort_values(group_cols)
    )

    # Reindex to a complete date range per SKU/(store, SKU) to make time series continuous.
    all_series: list[pd.DataFrame] = []
    key_cols = [config.sku_column]
    if config.store_column and config.store_column in agg.columns:
        key_cols = [config.store_column, config.sku_column]

    for key_values, slice_df in agg.groupby(key_cols):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        min_date = slice_df[config.date_column].min()
        max_date = slice_df[config.date_column].max()
        date_index = pd.date_range(min_date, max_date, freq=freq)

        slice_df = slice_df.set_index(config.date_column).reindex(date_index)
        slice_df.index.name = config.date_column
        # Forward-fill optional categorical columns, fill demand with 0 where missing.
        for col, val in zip(key_cols, key_values):
            slice_df[col] = val
        slice_df[config.demand_column] = slice_df[config.demand_column].fillna(0.0)
        all_series.append(slice_df.reset_index())

    ts = pd.concat(all_series, ignore_index=True)
    ts = ts.sort_values([config.sku_column, config.date_column])
    return ts


def add_time_features(df: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    """Create calendar and lag-based features for forecasting.

    Features:
    - day_of_week, week_of_year, month, year
    - is_weekend
    - lags: 1, 7, 14, 28
    - rolling means: 7, 14, 28
    """
    df = df.copy()
    df[config.date_column] = pd.to_datetime(df[config.date_column])

    df["day_of_week"] = df[config.date_column].dt.dayofweek
    df["week_of_year"] = df[config.date_column].dt.isocalendar().week.astype(int)
    df["month"] = df[config.date_column].dt.month
    df["year"] = df[config.date_column].dt.year
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    lag_windows = [1, 7, 14, 28]
    rolling_windows = [7, 14, 28]

    key_cols = [config.sku_column]
    if config.store_column and config.store_column in df.columns:
        key_cols = [config.store_column, config.sku_column]

    df = df.sort_values(key_cols + [config.date_column])

    for lag in lag_windows:
        df[f"lag_{lag}"] = (
            df.groupby(key_cols)[config.demand_column].shift(lag).fillna(0.0)
        )

    for window in rolling_windows:
        df[f"roll_mean_{window}"] = (
            df.groupby(key_cols)[config.demand_column]
            .shift(1)
            .rolling(window=window, min_periods=1)
            .mean()
            .fillna(0.0)
        )

    return df


def add_promotion_features(df: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    """Create promotion-related features if a promotion column is available."""
    if not (config.promo_column and config.promo_column in df.columns):
        return df

    df = df.copy()
    promo_col = config.promo_column
    df[promo_col] = df[promo_col].fillna(0).astype(int)

    key_cols = [config.sku_column]
    if config.store_column and config.store_column in df.columns:
        key_cols = [config.store_column, config.sku_column]

    df["promo_lag_7"] = (
        df.groupby(key_cols)[promo_col].shift(7).fillna(0).astype(int)
    )
    df["promo_rolling_28"] = (
        df.groupby(key_cols)[promo_col]
        .shift(1)
        .rolling(window=28, min_periods=1)
        .mean()
        .fillna(0.0)
    )
    return df


def build_demand_dataset(
    csv_paths: Sequence[str | Path],
    config: DataConfig,
    save: bool = True,
) -> pd.DataFrame:
    """Run the full data pipeline and optionally persist the processed dataset.

    Parameters
    ----------
    csv_paths:
        One or more raw CSV file paths (e.g. M5, Store Item Demand, DataCo).
    config:
        DataConfig describing column names and directories.
    save:
        If True, save the processed dataset to `processed_dir/processed_file`.
    """
    csv_paths = [Path(p) for p in csv_paths]
    raw = load_raw_data(csv_paths)
    aggregated = clean_and_aggregate(raw, config=config)
    with_time = add_time_features(aggregated, config=config)
    enriched = add_promotion_features(with_time, config=config)

    if save:
        processed_path = config.processed_dir / config.processed_file
        config.processed_dir.mkdir(parents=True, exist_ok=True)
        if processed_path.suffix == ".parquet":
            enriched.to_parquet(processed_path, index=False)
        else:
            enriched.to_csv(processed_path, index=False)

    return enriched

