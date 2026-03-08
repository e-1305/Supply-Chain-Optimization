from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from inventory_models import InventoryParameters, reorder_point, safety_stock

@dataclass
class SimulationConfig:
    """Configuration for Monte Carlo inventory simulation."""

    horizon_days: int = 365
    n_scenarios: int = 1000
    seed: int = 42
    demand_distribution: str = "poisson"  # or "normal"
    lead_time_days: Tuple[int, ...] = (5, 7, 10)
    lead_time_probs: Tuple[float, ...] = (0.3, 0.5, 0.2)


@dataclass
class Policy:
    """Inventory control policy parameters."""

    order_quantity: float
    reorder_point: float
    safety_stock: float
    service_level_target: float


@dataclass
class SimulationResults:
    """Aggregated metrics from Monte Carlo simulation."""

    policy: Policy
    average_inventory: float
    stockout_rate: float
    fill_rate: float
    average_holding_cost: float
    average_stockout_cost: float
    average_ordering_cost: float
    average_total_cost: float


def _sample_demand(
    rng: np.random.Generator,
    params: InventoryParameters,
    config: SimulationConfig,
    size: int,
) -> np.ndarray:
    if config.demand_distribution == "poisson":
        return rng.poisson(lam=max(params.demand_mean, 1e-6), size=size)
    if config.demand_distribution == "normal":
        samples = rng.normal(loc=params.demand_mean, scale=params.demand_std, size=size)
        return np.clip(samples, 0.0, None)
    raise ValueError(f"Unsupported demand distribution: {config.demand_distribution}")


def _sample_lead_time(rng: np.random.Generator, config: SimulationConfig) -> int:
    return int(rng.choice(config.lead_time_days, p=config.lead_time_probs))


def simulate_policy(
    params: InventoryParameters,
    policy: Policy,
    sim_config: SimulationConfig,
) -> SimulationResults:
    """Simulate daily inventory dynamics for a single (Q, R) policy.

    The simulation follows a simple continuous-review policy:
    - Each day, observe demand and satisfy it from on-hand inventory.
    - If inventory position (on-hand + on-order) <= R, place an order of size Q.
    - After a random lead time, the order arrives and increases on-hand inventory.
    """
    rng = np.random.default_rng(sim_config.seed)

    on_hand = policy.order_quantity + policy.safety_stock
    inventory_position = on_hand
    pipeline: List[Tuple[int, float]] = []  # (arrival_day, quantity)

    total_demand = 0.0
    total_fulfilled = 0.0
    holding_costs = 0.0
    stockout_costs = 0.0
    ordering_costs = 0.0

    on_hand_history: List[float] = []
    stockout_days = 0

    daily_demand = _sample_demand(
        rng,
        params=params,
        config=sim_config,
        size=sim_config.horizon_days,
    )

    for day in range(sim_config.horizon_days):
        # Receive any orders that arrive today.
        arrivals_today = [q for (arrival_day, q) in pipeline if arrival_day == day]
        if arrivals_today:
            arrival_qty = float(sum(arrivals_today))
            on_hand += arrival_qty
            inventory_position += arrival_qty
        pipeline = [(d, q) for (d, q) in pipeline if d > day]

        demand = float(daily_demand[day])
        total_demand += demand

        shipped = min(demand, on_hand)
        total_fulfilled += shipped
        lost_sales = demand - shipped

        on_hand -= shipped
        if lost_sales > 0:
            stockout_days += 1
            stockout_costs += lost_sales * params.stockout_cost

        # Continuous-review reorder check.
        inventory_position = on_hand + sum(q for (_, q) in pipeline)
        if inventory_position <= policy.reorder_point:
            lt = _sample_lead_time(rng, sim_config)
            arrival_day = day + lt
            pipeline.append((arrival_day, policy.order_quantity))
            inventory_position += policy.order_quantity
            ordering_costs += params.ordering_cost

        holding_costs += on_hand * params.holding_cost
        on_hand_history.append(on_hand)

    avg_inventory = float(np.mean(on_hand_history))
    stockout_rate = stockout_days / sim_config.horizon_days
    fill_rate = total_fulfilled / total_demand if total_demand > 0 else 1.0
    total_cost = holding_costs + stockout_costs + ordering_costs

    return SimulationResults(
        policy=policy,
        average_inventory=avg_inventory,
        stockout_rate=stockout_rate,
        fill_rate=fill_rate,
        average_holding_cost=holding_costs / sim_config.horizon_days,
        average_stockout_cost=stockout_costs / sim_config.horizon_days,
        average_ordering_cost=ordering_costs / sim_config.horizon_days,
        average_total_cost=total_cost / sim_config.horizon_days,
    )


def build_policy_for_service_level(
    params: InventoryParameters,
    service_level: float,
    annual_demand: Optional[float] = None,
) -> Policy:
    """Construct a (Q, R) policy for a given target service level."""
    from inventory_models import eoq

    local_params = InventoryParameters(
        demand_mean=params.demand_mean,
        demand_std=params.demand_std,
        lead_time=params.lead_time,
        holding_cost=params.holding_cost,
        ordering_cost=params.ordering_cost,
        stockout_cost=params.stockout_cost,
        service_level=service_level,
    )

    if annual_demand is None:
        annual_demand = params.demand_mean * 365.0
    q_star = eoq(annual_demand=annual_demand, ordering_cost=params.ordering_cost, holding_cost=params.holding_cost)
    ss = safety_stock(local_params)
    r_point = reorder_point(local_params)
    return Policy(
        order_quantity=q_star,
        reorder_point=r_point,
        safety_stock=ss,
        service_level_target=service_level,
    )


def run_scenario_analysis(
    params: InventoryParameters,
    sim_config: SimulationConfig,
    service_levels: List[float],
) -> pd.DataFrame:
    """Run Monte Carlo simulation for multiple service level scenarios.

    This is the core of the Consulting-style scenario analysis:
    e.g. 90% vs 95% vs 99% service level, and how that trades off cost vs
    inventory and stockouts.
    """
    records: List[Dict[str, float]] = []
    for sl in service_levels:
        policy = build_policy_for_service_level(params, service_level=sl)
        results = simulate_policy(params=params, policy=policy, sim_config=sim_config)
        records.append(
            {
                "service_level_target": sl,
                "order_quantity": policy.order_quantity,
                "reorder_point": policy.reorder_point,
                "safety_stock": policy.safety_stock,
                "average_inventory": results.average_inventory,
                "stockout_rate": results.stockout_rate,
                "fill_rate": results.fill_rate,
                "avg_holding_cost": results.average_holding_cost,
                "avg_stockout_cost": results.average_stockout_cost,
                "avg_ordering_cost": results.average_ordering_cost,
                "avg_total_cost": results.average_total_cost,
            }
        )
    return pd.DataFrame.from_records(records)

