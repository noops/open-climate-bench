"""Train/test splitting, metrics, and benchmark orchestration."""

import numpy as np
import pandas as pd

from climate_ledger.baselines import BASELINES

TEST_WINDOW_MONTHS = 60  # 5-year holdout


def time_based_split(df: pd.DataFrame, test_window_months: int = TEST_WINDOW_MONTHS):
    cutoff_index = len(df) - test_window_months
    return df.iloc[:cutoff_index].reset_index(drop=True), df.iloc[cutoff_index:].reset_index(
        drop=True
    )


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs((actual - predicted) / actual)) * 100)


def run_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    train, test = time_based_split(df)
    actual = test["co2_ppm"].to_numpy()

    rows = []
    for name, forecast_fn in BASELINES.items():
        predicted = forecast_fn(train, test["date"])
        rows.append(
            {
                "model": name,
                "rmse_ppm": round(rmse(actual, predicted), 3),
                "mape_pct": round(mape(actual, predicted), 3),
            }
        )

    return pd.DataFrame(rows).sort_values("mape_pct").reset_index(drop=True)
