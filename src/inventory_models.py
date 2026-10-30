from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Optional


@dataclass
class InventoryParameters:
    """Core parameters for classical inventory models.

    This structure is aligned with standard operations research texts such as
    Silver et al. (Inventory Management and Production Planning and Scheduling).
    """

    demand_mean: float  # average demand per period (e.g. per day)
    demand_std: float  # std of demand per period (use forecast error, e.g. RMSE)
    lead_time: float  # lead time in periods (e.g. days)
    holding_cost: float  # ANNUAL per-unit holding cost (as used by EOQ)
    ordering_cost: float  # fixed cost per replenishment order
    stockout_cost: float  # per-unit stockout or penalty cost
    service_level: float  # target cycle service level, between 0 and 1


def eoq(annual_demand: float, ordering_cost: float, holding_cost: float) -> float:
    """Economic Order Quantity (EOQ) model.

    EOQ = sqrt(2 * D * K / h)
    where:
    - D: annual demand
    - K: fixed ordering cost per order
    - h: annual holding cost per unit
    """
    if annual_demand <= 0 or ordering_cost <= 0 or holding_cost <= 0:
        raise ValueError("EOQ parameters must be positive.")
    from math import sqrt

    return sqrt(2.0 * annual_demand * ordering_cost / holding_cost)


def _z_from_service_level(service_level: float) -> float:
    """Return z-score corresponding to a cycle service level under Normal demand."""
    if not (0.0 < service_level < 1.0):
        raise ValueError("Service level must be between 0 and 1.")
    return NormalDist().inv_cdf(service_level)


def safety_stock(params: InventoryParameters, demand_during_lead_time_std: Optional[float] = None) -> float:
    """Safety stock for a continuous-review (Q, R) model.

    SS = z * sigma_L
    sigma_L = std deviation of demand during lead time.
    For demand ~ Normal with mean μ and std σ per period, and constant lead time L:
    sigma_L = σ * sqrt(L)
    """
    z = _z_from_service_level(params.service_level)
    if demand_during_lead_time_std is None:
        from math import sqrt

        demand_during_lead_time_std = params.demand_std * (params.lead_time**0.5)
    return z * demand_during_lead_time_std


def reorder_point(params: InventoryParameters) -> float:
    """Reorder point R for a continuous-review (Q, R) model.

    R = μ_L + SS
    where μ_L = μ * L is mean demand during lead time.
    """
    mean_demand_during_lead_time = params.demand_mean * params.lead_time
    ss = safety_stock(params)
    return mean_demand_during_lead_time + ss


def newsvendor_critical_fractile(params: InventoryParameters) -> float:
    """Critical fractile for the Newsvendor model.

    F(Q*) = Cu / (Cu + Co)
    where:
    - Cu: underage cost (stockout cost)
    - Co: overage cost (holding or markdown cost)
    """
    cu = params.stockout_cost
    co = params.holding_cost
    if cu <= 0 or co <= 0:
        raise ValueError("Underage and overage costs must be positive.")
    return cu / (cu + co)


def newsvendor_optimal_quantity(params: InventoryParameters) -> float:
    """Optimal order-up-to quantity Q* for the Newsvendor model with Normal demand.

    Q* = F^{-1}(Cu / (Cu + Co))
    where F^{-1} is the inverse CDF of demand (assumed Normal here).
    """
    crit = newsvendor_critical_fractile(params)
    z_star = NormalDist().inv_cdf(crit)
    return params.demand_mean + z_star * params.demand_std


def expected_cycle_service_level(params: InventoryParameters, reorder_point_value: Optional[float] = None) -> float:
    """Approximate achieved cycle service level under a (Q, R) policy.

    Uses the Normal approximation:
    CSL ≈ P(D_L <= R) where D_L ~ Normal(μ_L, σ_L^2)
    """
    from math import sqrt

    mean_demand_during_lead_time = params.demand_mean * params.lead_time
    std_demand_during_lead_time = params.demand_std * sqrt(params.lead_time)
    if reorder_point_value is None:
        reorder_point_value = reorder_point(params)
    z = (reorder_point_value - mean_demand_during_lead_time) / std_demand_during_lead_time
    return NormalDist().cdf(z)

