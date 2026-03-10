# Supply Chain Optimization
### Demand Forecasting · Inventory Optimization · Monte Carlo Simulation

> An end-to-end, data-driven supply chain analytics pipeline built in the style of an operations consulting engagement — connecting statistical forecasting, classical inventory theory, and stochastic simulation into a single reproducible system.

---

## Business Problem

Retailers and manufacturers face a fundamental tension: **hold too little inventory and you lose sales; hold too much and you destroy working capital.** Traditional planning processes rely on spreadsheets and intuition-based forecasts that:

- Miss seasonality, trend shifts, and promotion effects
- Apply fixed safety stock rules regardless of demand volatility or lead time
- Offer no way to quantify the cost of choosing 90% vs 99% service level

This project replaces that with a **data-driven pipeline** that connects demand forecasting directly to inventory policy optimization and validates those policies through Monte Carlo simulation — exactly the workflow an operations or supply chain consulting team would build for a retail client.

---

## Results (Store 1 · Item 1 — Kaggle Store Item Demand Dataset)

### Forecasting Performance

All models were evaluated on a 30-day hold-out period against 5 years of daily sales history.

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| Naive (baseline) | 4.40 | 5.80 | 36.7% |
| Moving Average (7-day) | 4.80 | 6.13 | 40.4% |
| ARIMA (1,1,1) | 6.13 | 7.28 | 51.3% |
| **Prophet** | **3.58** | **4.66** | **27.9%** |

**Prophet outperforms every other model**, reducing MAPE by 24% vs the naive baseline. On a SKU averaging ~19 units/day, Prophet's predictions are off by 3.6 units per day on average — tight enough to drive real inventory decisions.

ARIMA underperforms here because daily retail demand has strong weekly seasonality that ARIMA's linear structure doesn't capture as effectively as Prophet's Fourier decomposition.

### Inventory Scenario Analysis (Monte Carlo · 365-day horizon · 1,000 scenarios)

Demand parameters derived directly from the Prophet 90-day forecast (mean = 19.4 units/day, std = 0.81).

| Service Level | Safety Stock | Reorder Point | Avg Inventory | Stockout Rate | Fill Rate | Daily Total Cost |
|---|---|---|---|---|---|---|
| 90% | 2.8 units | 138.6 | 366.0 | 1.92% | 98.7% | $552.89 |
| 95% | 3.5 units | 139.4 | 366.1 | 1.92% | 98.7% | $552.98 |
| **99%** | **5.0 units** | **140.8** | **367.4** | **1.64%** | **98.9%** | **$554.40** |

### Key Business Insight

**Moving from 90% to 99% service level costs only $1.51/day (+0.3%) while cutting stockout frequency by 14%.** The decision is straightforward: the cost of a single lost sale or customer churn event almost certainly exceeds $1.51. For most retail contexts, the 99% policy dominates.

The EOQ of ~687 units reflects the high fixed ordering cost ($50/order) relative to the low daily holding cost ($1.50/unit), producing an optimal policy of large, infrequent orders. If the ordering cost were negotiated down to $20, EOQ drops to ~435 units and order frequency increases — a lever worth modeling when negotiating supplier contracts.

## Key Insight

Inventory decisions should not be driven purely by service level targets but by the marginal cost of preventing stockouts.

---

## Architecture

```
data/raw/train.csv
        │
        ▼
[ dataset_adapter.py ]   ← column name normalization
        │
        ▼
[ data_loader.py ]       ← cleaning, aggregation, lag/calendar features
        │
        ▼
data/processed/demand_features.parquet
        │
        ▼
[ forecasting.py ]       ← Naive · MA · ARIMA · Prophet · XGBoost
        │
        ├── results/forecasts/store{n}_item{n}_forecast.csv
        │
        ▼
[ inventory_models.py ]  ← EOQ · Safety Stock · Reorder Point · Newsvendor
        │
        ▼
[ simulation.py ]        ← Monte Carlo (Q,R) policy simulation
        │
        ├── results/scenario_analysis.csv
        │
        ▼
[ visualization.py ]     ← demand trend · forecast vs actual · cost tradeoff plots
        │
        ▼
results/plots/
```

---

## Repository Structure

```
Supply-Chain-Optimization/
├── config/
│   └── parameters.yaml          # single source of truth for all settings
├── data/
│   ├── raw/                     # place Kaggle CSVs here
│   └── processed/               # auto-generated parquet features
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_demand_forecasting.ipynb
│   ├── 03_inventory_optimization.ipynb
│   └── 04_simulation_analysis.ipynb
├── results/
│   ├── forecasts/               # per-SKU forecast CSVs
│   ├── plots/                   # PNG charts
│   └── scenario_analysis.csv    # scenario comparison table
├── src/
│   ├── data_loader.py
│   ├── dataset_adapter.py
│   ├── forecasting.py
│   ├── inventory_models.py
│   ├── simulation.py
│   └── visualization.py
├── run_pipeline.py              # end-to-end entry point
├── requirements.txt
└── .gitignore
```

---

## Methodology

### Phase 1 — Data Pipeline
Raw CSVs are ingested, column-mapped via `dataset_adapter.py`, cleaned (negative/missing demand set to zero), and aggregated to daily SKU-level series. Calendar features (day of week, month, weekend flag), lag features (1, 7, 14, 28 days), and rolling means are added to produce `demand_features.parquet`.

### Phase 2 — Demand Forecasting
Five models are trained on a rolling-origin train/test split and evaluated on MAE, RMSE, and MAPE. Prophet handles weekly and yearly seasonality natively and produces the best results on this dataset. The winning model's 90-day forward forecast is saved and passed downstream as the demand signal for inventory optimization.

### Phase 3 — Inventory Optimization
Classical operations research models from Silver et al.:

- **EOQ**: $Q^* = \sqrt{2DK/h}$ — optimal order quantity balancing ordering and holding costs
- **Safety Stock**: $SS = z \cdot \sigma_L$ where $\sigma_L = \sigma\sqrt{L}$ — buffer against demand variance during lead time
- **Reorder Point**: $R = \mu_L + SS$ — trigger level for replenishment orders
- **Newsvendor**: $Q^* = F^{-1}(C_u / (C_u + C_o))$ — optimal single-period quantity under uncertainty

Demand parameters ($\mu$, $\sigma$) are drawn directly from the forecast, not historical averages — this is the critical link between forecasting and inventory.

### Phase 4 — Monte Carlo Simulation
A continuous-review $(Q, R)$ policy is simulated for 365 days across 1,000 scenarios with:
- Stochastic daily demand (Poisson or Normal)
- Discrete lead time distribution ({5, 7, 10} days with configurable probabilities)
- Full order pipeline tracking (orders in transit)
- Per-day cost accounting (holding, stockout, ordering)

The simulation reveals the distribution of outcomes, not just point estimates — critical for risk-aware decision-making.

---

## How to Run

```bash
# 1. Clone and set up environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Place Kaggle dataset in data/raw/
#    Download: Store Item Demand Forecasting Challenge (train.csv)

# 3. Run the full pipeline (store 1, item 1)
python3 run_pipeline.py

# 4. Run for a different SKU
python3 run_pipeline.py --store 2 --item 15

# 5. Skip data ingestion if parquet already exists
python3 run_pipeline.py --skip-data
```

Outputs are written to `results/forecasts/`, `results/plots/`, and `results/scenario_analysis.csv`.

---

## Tech Stack

| Layer | Libraries |
|---|---|
| Data | pandas, numpy, pyarrow |
| Forecasting | statsmodels (ARIMA), prophet, xgboost, scikit-learn |
| Inventory math | Standard library (statistics.NormalDist) |
| Simulation | numpy (vectorized Monte Carlo) |
| Visualization | matplotlib, seaborn |
| Config | PyYAML |
| Notebooks | JupyterLab, ipykernel |

---

## Future Work

### Immediate Extensions
- **Multi-SKU pipeline**: Scale to all 500 series (10 stores × 50 items) with parallel execution via `joblib`. Aggregate results into a SKU-level leaderboard showing worst forecast accuracy and highest stockout risk by segment.
- **XGBoost forecasting**: Enable gradient boosting with lag features for SKUs where Prophet underperforms — particularly intermittent or highly promotional demand.
- **Forecast-to-inventory feedback loop**: Automatically select the best-performing model per SKU and use its forecast as the inventory input, rather than a fixed model choice.

### Analytical Depth
- **Multi-echelon inventory**: Extend the (Q,R) model to a two-echelon system (DC → store), capturing the bullwhip effect and inter-location stock transfers.
- **Hierarchical forecasting**: Reconcile forecasts top-down (national → region → store → SKU) using the MinT or BU reconciliation methods to ensure coherent aggregate plans.
- **Promotion modeling**: Incorporate the `onpromotion` flag as a regressor in Prophet and XGBoost to quantify lift and pre-position inventory before promotional events.
- **Newsvendor at scale**: Apply the Newsvendor model to perishable or fashion SKUs where a single-period assumption is more appropriate than continuous review.

### Production Path
- **Streamlit dashboard**: Interactive UI where a planner selects a SKU, adjusts service level, and sees real-time reorder point, safety stock, and simulation output — the consulting deliverable made self-service.
- **Supply chain network optimization**: Layer in a linear program to allocate inventory across a network of warehouses, minimizing total landed cost subject to service level constraints.
- **ERP integration**: Replace the CSV ingestion layer with API connectors to SAP, NetSuite, or Snowflake for live demand signals and automatic policy updates.

---

## Dataset

[Store Item Demand Forecasting Challenge](https://www.kaggle.com/competitions/demand-forecasting-kernels-only) — Kaggle  
50 items × 10 stores × 5 years of daily sales. Clean panel time series with strong weekly and yearly seasonality.

---

## Example Workflow

The pipeline follows a typical analytics workflow used in operations consulting:

1. **Ingest demand data**
   - Load transactional sales data
   - Normalize schema across datasets
   - Generate time-series features

2. **Forecast demand**
   - Train multiple forecasting models
   - Evaluate with MAE, RMSE, and MAPE
   - Select the best-performing model per SKU

3. **Optimize inventory policy**
   - Compute EOQ based on demand and cost structure
   - Estimate safety stock under service-level constraints
   - Derive reorder points accounting for lead-time uncertainty

4. **Validate via simulation**
   - Run Monte Carlo simulations of the $(Q,R)$ policy
   - Measure service levels, stockouts, and total costs
   - Compare policies across multiple service-level scenarios

This workflow mirrors the analytical approach used in supply chain consulting projects when transitioning organizations from spreadsheet-based planning to **data-driven inventory optimization systems**.

---

## Reproducibility

All experiments are fully reproducible. The repository is structured so that:

- Configuration parameters are centralized in `config/parameters.yaml`
- Raw datasets are stored in `data/raw/`
- Intermediate datasets are stored in `data/processed/`
- Results and artifacts are written to `results/`

Running `run_pipeline.py` executes the full forecasting → optimization → simulation workflow.