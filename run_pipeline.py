"""run_pipeline.py
End-to-end pipeline entry point.

Usage
-----
    python3 run_pipeline.py                        # full run, store 1 / item 1
    python3 run_pipeline.py --store 2 --item 5     # specific SKU
    python3 run_pipeline.py --skip-data            # skip feature generation
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Must be set before any matplotlib/visualization import
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Make sure src/ is on the path so modules resolve without a package install
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str = "config/parameters.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Stage 1 – Data ingestion & feature engineering
# ---------------------------------------------------------------------------

def stage_data(cfg: dict) -> pd.DataFrame:
    from dataset_adapter import load_and_adapt
    from data_loader import DataConfig, build_demand_dataset

    dcfg = cfg["data"]
    raw_dir = Path(dcfg["raw_dir"])
    processed_dir = Path(dcfg["processed_dir"])
    processed_path = processed_dir / dcfg["processed_file"]

    if processed_path.exists():
        log.info("Processed dataset already exists at %s — skipping ingestion.", processed_path)
        return pd.read_parquet(processed_path)

    log.info("Loading raw data from %s …", raw_dir)
    raw_csv = raw_dir / "train.csv"
    if not raw_csv.exists():
        raise FileNotFoundError(
            f"Raw CSV not found at {raw_csv}.\n"
            "Download the Store Item Demand Forecasting dataset from Kaggle and place "
            "train.csv in data/raw/."
        )

    raw_df = load_and_adapt(raw_csv, preset=dcfg.get("dataset_preset", "store_item"))

    config = DataConfig(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        processed_file=dcfg["processed_file"],
        date_column=dcfg["date_column"],
        demand_column=dcfg["demand_column"],
        sku_column=dcfg["sku_column"],
        store_column=dcfg.get("store_column"),
        promo_column=dcfg.get("promo_column"),
        frequency=dcfg.get("frequency", "D"),
    )

    ensure_dirs(processed_dir)
    tmp_path = processed_dir / "_adapted_train.csv"
    raw_df.to_csv(tmp_path, index=False)

    log.info("Building feature dataset …")
    enriched = build_demand_dataset(csv_paths=[tmp_path], config=config, save=True)
    tmp_path.unlink(missing_ok=True)
    log.info("Saved processed dataset → %s", processed_path)
    return enriched


# ---------------------------------------------------------------------------
# Stage 2 – Forecasting
# ---------------------------------------------------------------------------

def stage_forecasting(
    df: pd.DataFrame,
    cfg: dict,
    store_id: int,
    item_id: int,
) -> tuple[pd.DataFrame, dict]:
    from forecasting import run_univariate_forecasting

    fcfg = cfg["forecasting"]
    dcfg = cfg["data"]

    mask = (df[dcfg["store_column"]] == store_id) & (df[dcfg["sku_column"]] == item_id)
    subset = df[mask].set_index(dcfg["date_column"]).sort_index()
    series = subset[dcfg["demand_column"]].astype(float)

    if len(series) < fcfg["test_horizon_days"] + fcfg["forecast_horizon_days"]:
        raise ValueError(f"Series for store={store_id}, item={item_id} is too short.")

    log.info("Running forecasting for store=%s, item=%s …", store_id, item_id)
    future_fc, metrics = run_univariate_forecasting(
        series,
        horizon=fcfg["forecast_horizon_days"],
        test_horizon=fcfg["test_horizon_days"],
        freq=fcfg.get("frequency", "D"),
        use_gradient_boosting=fcfg.get("use_gradient_boosting"),
        lag_features=fcfg.get("gradient_boosting", {}).get("lag_features"),
    )

    results_dir = Path(cfg["results"]["forecasts_dir"])
    ensure_dirs(results_dir)
    out_path = results_dir / f"store{store_id}_item{item_id}_forecast.csv"
    future_fc.to_csv(out_path)
    log.info("Saved forecasts → %s", out_path)

    log.info("Forecast metrics:")
    for model, res in metrics.items():
        log.info("  %-20s  MAE=%.2f  RMSE=%.2f  MAPE=%.1f%%", model, res.mae, res.rmse, res.mape)

    return future_fc, metrics


# ---------------------------------------------------------------------------
# Stage 3 – Inventory optimisation
# ---------------------------------------------------------------------------

def stage_inventory(future_fc: pd.DataFrame, cfg: dict) -> object:
    from inventory_models import InventoryParameters

    icfg = cfg["inventory"]
    valid_cols = future_fc.dropna(axis=1, how="all").columns.tolist()
    demand_signal = future_fc[valid_cols].mean(axis=1)

    params = InventoryParameters(
        demand_mean=float(demand_signal.mean()),
        demand_std=float(demand_signal.std()) or 1.0,
        lead_time=float(icfg["default_lead_time_days"]),
        holding_cost=float(icfg["annual_holding_cost_per_unit"]),
        ordering_cost=float(icfg["ordering_cost_per_order"]),
        stockout_cost=float(icfg["stockout_cost_per_unit"]),
        service_level=0.95,
    )
    log.info(
        "Inventory params  demand_mean=%.2f  demand_std=%.2f  lead_time=%s",
        params.demand_mean,
        params.demand_std,
        params.lead_time,
    )
    return params


# ---------------------------------------------------------------------------
# Stage 4 – Scenario analysis (Monte Carlo)
# ---------------------------------------------------------------------------

def stage_simulation(params, cfg: dict) -> pd.DataFrame:
    from simulation import SimulationConfig, run_scenario_analysis

    scfg = cfg["simulation"]
    sim_config = SimulationConfig(
        horizon_days=scfg["horizon_days"],
        n_scenarios=scfg["n_scenarios"],
        seed=scfg["seed"],
        demand_distribution=scfg["demand_distribution"],
        lead_time_days=tuple(scfg["lead_time_days"]),
        lead_time_probs=tuple(scfg["lead_time_probs"]),
    )

    service_levels = cfg.get("service_levels") or cfg["inventory"]["service_levels"]
    log.info("Running scenario analysis for service levels: %s …", service_levels)
    scenarios = run_scenario_analysis(params, sim_config, service_levels=service_levels)

    out_path = Path(cfg["results"]["scenario_file"])
    ensure_dirs(out_path.parent)
    scenarios.to_csv(out_path, index=False)
    log.info("Saved scenario results → %s", out_path)
    log.info("\n%s", scenarios.to_string(index=False))
    return scenarios


# ---------------------------------------------------------------------------
# Stage 5 – Plots
# ---------------------------------------------------------------------------

def stage_plots(df: pd.DataFrame, future_fc: pd.DataFrame, scenarios: pd.DataFrame, cfg: dict) -> None:
    from visualization import (
        plot_demand_trend,
        plot_forecast_vs_actual,
        plot_cost_distribution,
        plot_policy_tradeoff,
        set_default_style,
    )
    from forecasting import rolling_origin_split

    set_default_style()
    plots_dir = Path(cfg["results"]["plots_dir"])
    ensure_dirs(plots_dir)
    dcfg = cfg["data"]
    fcfg = cfg["forecasting"]

    # Demand trend
    fig = plot_demand_trend(df, date_col=dcfg["date_column"], demand_col=dcfg["demand_column"])
    fig.savefig(plots_dir / "demand_trend.png", dpi=150)
    log.info("Saved demand_trend.png")

    # Forecast vs actual
    mask = (df[dcfg["store_column"]] == 1) & (df[dcfg["sku_column"]] == 1)
    series = df[mask].set_index(dcfg["date_column"]).sort_index()[dcfg["demand_column"]].astype(float)
    _, test = rolling_origin_split(series, forecast_horizon=fcfg["forecast_horizon_days"], test_horizon=fcfg["test_horizon_days"])
    fig = plot_forecast_vs_actual(test, future_fc.iloc[: fcfg["test_horizon_days"]])
    fig.savefig(plots_dir / "forecast_vs_actual.png", dpi=150)
    log.info("Saved forecast_vs_actual.png")

    # Cost & tradeoff
    fig = plot_cost_distribution(scenarios)
    fig.savefig(plots_dir / "cost_by_service_level.png", dpi=150)

    fig = plot_policy_tradeoff(scenarios)
    fig.savefig(plots_dir / "service_tradeoff.png", dpi=150)
    log.info("Saved service_tradeoff.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inventory analytics pipeline.")
    parser.add_argument("--config", default="config/parameters.yaml")  # fixed
    parser.add_argument("--store", type=int, default=1)
    parser.add_argument("--item", type=int, default=1)
    parser.add_argument("--skip-data", action="store_true", help="Skip data ingestion stage.")
    parser.add_argument("--skip-plots", action="store_true", help="Skip plot generation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.skip_data:
        processed_path = Path(cfg["data"]["processed_dir"]) / cfg["data"]["processed_file"]
        log.info("Loading existing processed dataset from %s …", processed_path)
        df = pd.read_parquet(processed_path)
    else:
        df = stage_data(cfg)

    future_fc, metrics = stage_forecasting(df, cfg, store_id=args.store, item_id=args.item)
    params = stage_inventory(future_fc, cfg)
    scenarios = stage_simulation(params, cfg)

    if not args.skip_plots:
        stage_plots(df, future_fc, scenarios, cfg)

    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()