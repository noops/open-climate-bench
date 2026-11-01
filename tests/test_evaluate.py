import numpy as np
import pandas as pd

from climate_ledger.evaluate import mape, rmse, run_benchmark, time_based_split


def test_time_based_split_keeps_test_window_size():
    df = pd.DataFrame({"date": pd.date_range("2000-01-01", periods=100, freq="MS")})

    _, test = time_based_split(df, test_window_months=12)

    assert len(test) == 12


def test_rmse_is_zero_when_predictions_match_actuals():
    actual = np.array([1.0, 2.0, 3.0])

    result = rmse(actual, actual.copy())

    assert result == 0.0


def test_mape_is_zero_when_predictions_match_actuals():
    actual = np.array([1.0, 2.0, 3.0])

    result = mape(actual, actual.copy())

    assert result == 0.0


def test_run_benchmark_ranks_linear_trend_model_above_persistence_on_trending_data():
    dates = pd.date_range("1990-01-01", periods=120, freq="MS")
    co2_ppm = 300 + np.arange(120) * 0.1  # steady upward trend, no seasonality
    df = pd.DataFrame({"date": dates, "co2_ppm": co2_ppm})

    results = run_benchmark(df)

    ranked_models = list(results["model"])
    assert ranked_models.index("linear") < ranked_models.index("persistence")
