from __future__ import annotations

"""Dataset adapter layer.

Maps raw dataset column names to the standardized column names expected by
the data pipeline (DataConfig).  Add new dataset presets here as needed.
"""

from pathlib import Path
from typing import Dict, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Column-rename presets
# ---------------------------------------------------------------------------

COLUMN_MAPS: Dict[str, Dict[str, str]] = {
    # Kaggle – Store Item Demand Forecasting Challenge
    # train.csv columns: date, store, item, sales
    "store_item": {
        "store": "store_id",
        "item": "item_id",
        # "date" and "sales" already match DataConfig defaults
    },
    # Kaggle – M5 Forecasting (melted / long format)
    # columns: date, item_id, store_id, sales
    "m5": {},  # already matches DataConfig defaults
    # Kaggle – DataCo Global Supply Chain
    # columns: Order Date, Product Name, Order Item Quantity, …
    "dataco": {
        "Order Date": "date",
        "Product Name": "item_id",
        "Order Item Quantity": "sales",
        "Department Name": "store_id",
    },
}


def adapt(
    df: pd.DataFrame,
    preset: str,
    extra_renames: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Apply a column-rename preset and optional extra renames to *df*.

    Parameters
    ----------
    df:
        Raw DataFrame straight from ``pd.read_csv``.
    preset:
        Key into ``COLUMN_MAPS`` (e.g. ``"store_item"``).
    extra_renames:
        Any additional column renames to apply on top of the preset.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with renamed columns and ``date`` parsed to datetime.
    """
    if preset not in COLUMN_MAPS:
        raise ValueError(
            f"Unknown preset '{preset}'. Available: {list(COLUMN_MAPS.keys())}"
        )

    rename_map = dict(COLUMN_MAPS[preset])
    if extra_renames:
        rename_map.update(extra_renames)

    df = df.copy()
    df = df.rename(columns=rename_map)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    return df


def load_and_adapt(
    path: str | Path,
    preset: str,
    extra_renames: Optional[Dict[str, str]] = None,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """Convenience wrapper: read CSV then adapt columns.

    Parameters
    ----------
    path:
        Path to the raw CSV file.
    preset:
        Dataset preset name (see ``COLUMN_MAPS``).
    extra_renames:
        Extra column renames beyond the preset.
    **read_csv_kwargs:
        Forwarded to ``pd.read_csv``.
    """
    df = pd.read_csv(path, **read_csv_kwargs)
    return adapt(df, preset=preset, extra_renames=extra_renames)