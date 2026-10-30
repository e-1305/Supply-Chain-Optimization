"""Fast checks on the inventory maths and the simulation (no data needed)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inventory_models import InventoryParameters, eoq, safety_stock  # noqa: E402
from simulation import Policy, SimulationConfig, run_scenario_analysis, simulate_policy  # noqa: E402


def params(**kw):
    base = dict(demand_mean=19.4, demand_std=4.7, lead_time=7, holding_cost=1.5,
                ordering_cost=50.0, stockout_cost=10.0, service_level=0.95)
    base.update(kw)
    return InventoryParameters(**base)


def test_eoq_matches_textbook_value():
    # sqrt(2 * 19.4 * 365 * 50 / 1.5) ≈ 687
    assert eoq(19.4 * 365, 50.0, 1.5) == pytest.approx(687, abs=1)


def test_safety_stock_grows_with_service_level():
    assert safety_stock(params(service_level=0.90)) < safety_stock(params(service_level=0.99))


def test_holding_cost_is_annual_in_simulation():
    # No demand: 365 units on hand all year at $1.50/unit/year -> $1.50/day.
    cfg = SimulationConfig(horizon_days=365, n_scenarios=1, demand_distribution="normal")
    p = params(demand_mean=0.0, demand_std=0.0)
    policy = Policy(order_quantity=365.0, reorder_point=-1.0, safety_stock=0.0, service_level_target=0.95)
    res = simulate_policy(p, policy, cfg)
    assert res.average_holding_cost == pytest.approx(1.5, rel=1e-6)


def test_scenario_analysis_runs_all_scenarios():
    cfg = SimulationConfig(horizon_days=90, n_scenarios=20)
    table = run_scenario_analysis(params(), cfg, service_levels=[0.90, 0.99])
    assert list(table["n_scenarios"]) == [20, 20]
    assert (table["p95_total_cost"] >= table["avg_total_cost"]).all()
