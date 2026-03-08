from __future__ import annotations

from typing import Iterable, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def set_default_style(style: str = "seaborn-v0_8") -> None:
    """Apply a consistent plotting style across the project."""
    try:
        plt.style.use(style)
    except OSError:
        # Fall back silently if style is not available.
        pass
    sns.set_palette("deep")


def plot_demand_trend(
    df: pd.DataFrame,
    date_col: str,
    demand_col: str,
    title: str = "Demand Over Time",
) -> plt.Figure:
    """Line plot of demand over time."""
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=df, x=date_col, y=demand_col, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Demand (units)")
    fig.tight_layout()
    return fig


def plot_forecast_vs_actual(
    actual: pd.Series,
    forecasts: pd.DataFrame,
    title: str = "Forecast vs Actual",
) -> plt.Figure:
    """Plot actuals and one or more forecast series on the same axes."""
    fig, ax = plt.subplots(figsize=(10, 4))
    actual.plot(ax=ax, label="Actual", color="black", linewidth=2)
    for col in forecasts.columns:
        forecasts[col].plot(ax=ax, label=col)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Demand (units)")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_inventory_trajectory(
    inventory_history: pd.Series,
    title: str = "Inventory Trajectory",
) -> plt.Figure:
    """Plot inventory on hand over time for a single simulation path."""
    fig, ax = plt.subplots(figsize=(10, 4))
    inventory_history.plot(ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Day")
    ax.set_ylabel("On-hand Inventory")
    fig.tight_layout()
    return fig


def plot_cost_distribution(
    scenario_df: pd.DataFrame,
    cost_col: str = "avg_total_cost",
    title: str = "Cost Distribution by Scenario",
) -> plt.Figure:
    """Boxplot or violin plot of cost metrics across scenarios."""
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=scenario_df, x="service_level_target", y=cost_col, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Target Service Level")
    ax.set_ylabel("Average Daily Total Cost")
    fig.tight_layout()
    return fig


def plot_policy_tradeoff(
    scenario_df: pd.DataFrame,
    x_col: str = "average_inventory",
    y_col: str = "avg_total_cost",
    hue_col: str = "service_level_target",
    title: str = "Inventory vs Cost Trade-off",
) -> plt.Figure:
    """Scatter plot showing how policies trade off inventory and cost."""
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(data=scenario_df, x=x_col, y=y_col, hue=hue_col, s=80, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Average Inventory")
    ax.set_ylabel("Average Daily Total Cost")
    fig.tight_layout()
    return fig

