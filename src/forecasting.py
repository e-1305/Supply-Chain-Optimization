from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error with protection against division by zero."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not np.any(mask):
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0


def naive_forecast(series: pd.Series, horizon: int) -> np.ndarray:
    """Naive forecast: last observed value repeated."""
    if series.empty:
        raise ValueError("Time series is empty.")
    last_value = series.iloc[-1]
    return np.full(shape=horizon, fill_value=last_value, dtype=float)


def moving_average_forecast(series: pd.Series, horizon: int, window: int = 7) -> np.ndarray:
    """Moving average forecast: mean of last `window` observations."""
    if len(series) < window:
        window = len(series)
    avg = series.iloc[-window:].mean()
    return np.full(shape=horizon, fill_value=avg, dtype=float)


def arima_forecast(series: pd.Series, horizon: int, order=(1, 1, 1)) -> np.ndarray:
    """ARIMA forecast using statsmodels."""
    from statsmodels.tsa.arima.model import ARIMA  # lazy import

    model = ARIMA(series.astype(float), order=order)
    fitted = model.fit()
    forecast = fitted.forecast(steps=horizon)
    return forecast.to_numpy()


def prophet_forecast(series: pd.Series, horizon: int, freq: str = "D") -> np.ndarray:
    """Forecast with Prophet. Expects a univariate time series with a DatetimeIndex."""
    from prophet import Prophet  # lazy import

    df = pd.DataFrame({"ds": series.index, "y": series.values})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=horizon, freq=freq, include_history=False)
    forecast = model.predict(future)
    return forecast["yhat"].to_numpy()


def gradient_boosting_forecast(
    df: pd.DataFrame,
    target_col: str,
    horizon: int,
    lag_features: List[int],
) -> np.ndarray:
    """Gradient boosting (XGBoost) forecast using lag features.

    The function assumes `df` already contains lag features (e.g. from the data
    pipeline) and that the last row corresponds to the most recent date.
    """
    from xgboost import XGBRegressor  # lazy import

    feature_cols = [f"lag_{lag}" for lag in lag_features if f"lag_{lag}" in df.columns]
    if not feature_cols:
        raise ValueError("No lag features available for gradient boosting model.")

    train_df = df.dropna(subset=feature_cols + [target_col])
    X = train_df[feature_cols].to_numpy()
    y = train_df[target_col].to_numpy()

    model = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
    )
    model.fit(X, y)

    # Recursive multi-step forecast using last observed lags.
    history = df.copy()
    preds: List[float] = []
    for _ in range(horizon):
        row = history.iloc[-1]
        x_next = row[feature_cols].to_numpy().reshape(1, -1)
        y_next = float(model.predict(x_next)[0])
        preds.append(y_next)

        new_row = row.copy()
        new_row[target_col] = y_next
        for lag in sorted(lag_features):
            lag_name = f"lag_{lag}"
            if lag_name not in history.columns:
                continue
            if lag == 1:
                new_row[lag_name] = y_next
            else:
                prev_lag_name = f"lag_{lag-1}"
                if prev_lag_name in history.columns:
                    new_row[lag_name] = new_row[prev_lag_name]
        history = pd.concat([history, new_row.to_frame().T], ignore_index=True)

    return np.asarray(preds)


@dataclass
class ForecastResult:
    """Container for model forecasts and evaluation metrics."""

    model_name: str
    y_true: np.ndarray
    y_pred: np.ndarray
    mae: float
    rmse: float
    mape: float


def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> ForecastResult:
    """Compute MAE, RMSE, and MAPE for a forecast."""
    mae_val = float(mean_absolute_error(y_true, y_pred))
    rmse_val = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape_val = float(mape(y_true, y_pred))
    return ForecastResult(
        model_name=model_name,
        y_true=y_true,
        y_pred=y_pred,
        mae=mae_val,
        rmse=rmse_val,
        mape=mape_val,
    )


def rolling_origin_split(
    series: pd.Series,
    forecast_horizon: int,
    test_horizon: int,
) -> Tuple[pd.Series, pd.Series]:
    """Simple train/test split for time series."""
    if len(series) <= test_horizon + forecast_horizon:
        raise ValueError("Time series too short for specified horizons.")
    train = series.iloc[: -test_horizon]
    test = series.iloc[-test_horizon:]
    return train, test


def run_univariate_forecasting(
    series: pd.Series,
    horizon: int = 30,
    test_horizon: int = 30,
    freq: str = "D",
    use_gradient_boosting: bool = False,
    lag_features: Optional[List[int]] = None,
) -> Tuple[pd.DataFrame, Dict[str, ForecastResult]]:
    """Run multiple models on a univariate series and return forecasts and metrics.

    Returns
    -------
    forecasts_df:
        DataFrame with one column per model for the forecast horizon.
    metrics:
        Mapping of model name to `ForecastResult`.
    """
    train, test = rolling_origin_split(series, forecast_horizon=horizon, test_horizon=test_horizon)
    metrics: Dict[str, ForecastResult] = {}

    # Naive
    naive_pred = naive_forecast(train, horizon=test_horizon)
    metrics["naive"] = evaluate_forecast(test.to_numpy(), naive_pred, "naive")

    # Moving average
    ma_pred = moving_average_forecast(train, horizon=test_horizon, window=7)
    metrics["moving_average"] = evaluate_forecast(test.to_numpy(), ma_pred, "moving_average")

    # ARIMA
    try:
        arima_pred = arima_forecast(train, horizon=test_horizon)
        metrics["arima"] = evaluate_forecast(test.to_numpy(), arima_pred, "arima")
    except Exception:
        arima_pred = np.full(test_horizon, np.nan)

    # Prophet
    try:
        prophet_pred = prophet_forecast(train, horizon=test_horizon, freq=freq)
        metrics["prophet"] = evaluate_forecast(test.to_numpy(), prophet_pred, "prophet")
    except Exception:
        prophet_pred = np.full(test_horizon, np.nan)

    # Gradient boosting (optional, requires lag features prepared externally)
    if use_gradient_boosting:
        if lag_features is None:
            lag_features = [1, 2, 3, 7, 14, 28]
        df = series.to_frame(name="y")
        for lag in lag_features:
            df[f"lag_{lag}"] = df["y"].shift(lag)
        df = df.dropna()
        train_df = df.iloc[: -test_horizon]
        gb_pred = gradient_boosting_forecast(
            train_df,
            target_col="y",
            horizon=test_horizon,
            lag_features=lag_features,
        )
        metrics["gradient_boosting"] = evaluate_forecast(test.to_numpy(), gb_pred, "gradient_boosting")
    else:
        gb_pred = np.full(test_horizon, np.nan)

    # Collect forecasts into a single DataFrame for the next horizon (30–90 days)
    future_index = pd.date_range(start=series.index.max() + pd.Timedelta(1, unit=freq), periods=horizon, freq=freq)
    future_forecasts = pd.DataFrame(index=future_index)
    future_forecasts["naive"] = naive_forecast(series, horizon=horizon)
    future_forecasts["moving_average"] = moving_average_forecast(series, horizon=horizon, window=7)
    try:
        future_forecasts["arima"] = arima_forecast(series, horizon=horizon)
    except Exception:
        future_forecasts["arima"] = np.full(horizon, np.nan)
    try:
        future_forecasts["prophet"] = prophet_forecast(series, horizon=horizon, freq=freq)
    except Exception:
        future_forecasts["prophet"] = np.full(horizon, np.nan)
    if use_gradient_boosting and "gradient_boosting" in metrics:
        # Refit gradient boosting on full series for future horizon
        df_full = series.to_frame(name="y")
        for lag in lag_features or []:
            df_full[f"lag_{lag}"] = df_full["y"].shift(lag)
        df_full = df_full.dropna()
        future_forecasts["gradient_boosting"] = gradient_boosting_forecast(
            df_full, target_col="y", horizon=horizon, lag_features=lag_features or []
        )

    return future_forecasts, metrics

