"""Baseline forecasters for the CO2 benchmark.

Each baseline fits on a training frame (columns: date, co2_ppm) and predicts
co2_ppm for a held-out set of future dates, using only information available
at the end of the training period.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

MONTHS_PER_YEAR = 12


def _time_features(dates: pd.Series, origin: pd.Timestamp) -> pd.DataFrame:
    months_since_origin = (dates.dt.year - origin.year) * MONTHS_PER_YEAR + (
        dates.dt.month - origin.month
    )
    angle = 2 * np.pi * dates.dt.month / MONTHS_PER_YEAR
    return pd.DataFrame(
        {
            "months_since_origin": months_since_origin,
            "month_sin": np.sin(angle),
            "month_cos": np.cos(angle),
        }
    )


def persistence_forecast(train: pd.DataFrame, test_dates: pd.Series) -> np.ndarray:
    """Carry the last observed training value forward for every future date."""
    last_value = train["co2_ppm"].iloc[-1]
    return np.full(len(test_dates), last_value)


def linear_forecast(train: pd.DataFrame, test_dates: pd.Series) -> np.ndarray:
    """Fit a linear trend + seasonal model and extrapolate it forward."""
    origin = train["date"].iloc[0]
    model = LinearRegression()
    model.fit(_time_features(train["date"], origin), train["co2_ppm"])
    return model.predict(_time_features(test_dates, origin))


def xgboost_forecast(train: pd.DataFrame, test_dates: pd.Series) -> np.ndarray:
    """Fit a gradient-boosted tree model on trend + seasonal features."""
    origin = train["date"].iloc[0]
    model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05)
    model.fit(_time_features(train["date"], origin), train["co2_ppm"])
    return model.predict(_time_features(test_dates, origin))


BASELINES = {
    "persistence": persistence_forecast,
    "linear": linear_forecast,
    "xgboost": xgboost_forecast,
}
