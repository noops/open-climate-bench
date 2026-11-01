import numpy as np
import pandas as pd
import pytest

from climate_ledger.baselines import linear_forecast, persistence_forecast

TRAIN_DATES = pd.date_range("2000-01-01", periods=24, freq="MS")
LINEAR_TREND_VALUES = 300 + np.arange(24) * 0.5  # exactly linear, no seasonality
TEST_DATES = pd.Series(pd.date_range("2002-01-01", periods=3, freq="MS"))


def test_persistence_forecast_repeats_last_training_value():
    train = pd.DataFrame({"date": TRAIN_DATES, "co2_ppm": LINEAR_TREND_VALUES})

    forecast = persistence_forecast(train, TEST_DATES)

    assert np.all(forecast == LINEAR_TREND_VALUES[-1])


def test_linear_forecast_extrapolates_a_perfect_linear_trend():
    train = pd.DataFrame({"date": TRAIN_DATES, "co2_ppm": LINEAR_TREND_VALUES})
    expected_next_value = LINEAR_TREND_VALUES[-1] + 0.5  # one month past training data

    forecast = linear_forecast(train, TEST_DATES)

    assert forecast[0] == pytest.approx(expected_next_value, abs=0.01)
