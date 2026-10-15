"""A small, transparent ML demonstration for forecasting inflation.

IMPORTANT: This module generates synthetic data for a classroom demonstration.
It does not download or use official historical inflation data.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def generate_synthetic_inflation(n_months: int = 240, seed: int = 42) -> pd.DataFrame:
    """Generate monthly synthetic inflation with a cycle and mean-reverting noise."""
    rng = np.random.default_rng(seed)
    months = np.arange(n_months)
    # Smoothly cyclical series with realistic-looking variation, in percent.
    values = 4.0 + 0.9 * np.sin(2 * np.pi * months / 60) + rng.normal(0, 0.35, n_months)
    values = np.clip(values, 0.2, 10.0)
    return pd.DataFrame({"month": months, "inflation_pct": values})


def make_lagged_features(series: pd.Series, lags: int = 12):
    """Each row predicts the next month using the previous `lags` observations."""
    values = np.asarray(series, dtype=float)
    X, y = [], []
    for idx in range(lags, len(values)):
        X.append(values[idx-lags:idx][::-1])  # lag_1 is most recent
        y.append(values[idx])
    columns = [f"lag_{i}" for i in range(1, lags + 1)]
    return pd.DataFrame(X, columns=columns), np.asarray(y)


def train_and_forecast(n_months: int = 240, forecast_months: int = 12, seed: int = 42):
    """Time-ordered train/test evaluation, then recursive future prediction."""
    df = generate_synthetic_inflation(n_months=n_months, seed=seed)
    X, y = make_lagged_features(df["inflation_pct"], lags=12)
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y[:split], y[split:]

    evaluation_model = RandomForestRegressor(
        n_estimators=250, max_depth=5, min_samples_leaf=3, random_state=seed
    )
    evaluation_model.fit(X_train, y_train)
    predictions = evaluation_model.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "test_actual": y_test,
        "test_predicted": predictions,
        "test_months": df["month"].to_numpy()[12 + split:],
    }

    # Refit on all observed synthetic data before forecasting forward.
    final_model = RandomForestRegressor(
        n_estimators=250, max_depth=5, min_samples_leaf=3, random_state=seed
    )
    final_model.fit(X, y)
    history = list(df["inflation_pct"].to_numpy())
    future = []
    for _ in range(forecast_months):
        latest_lags = pd.DataFrame([history[-12:][::-1]], columns=X.columns)
        next_value = float(final_model.predict(latest_lags)[0])
        future.append(next_value)
        history.append(next_value)

    future_months = np.arange(n_months, n_months + forecast_months)
    future_df = pd.DataFrame({
        "month": future_months,
        "predicted_inflation_pct": future
    })
    return df, future_df, metrics
