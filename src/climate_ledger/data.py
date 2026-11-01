"""Load the NOAA Mauna Loa monthly mean CO2 record (the Keeling Curve)."""

from pathlib import Path

import pandas as pd
import requests

CO2_SOURCE_URL = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.txt"
RAW_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "co2_mm_mlo.txt"

# NOAA uses -99.99 as a missing-value sentinel for the monthly average column.
MISSING_VALUE_SENTINEL = -99.99

COLUMN_NAMES = [
    "year",
    "month",
    "decimal_date",
    "co2_ppm",
    "co2_ppm_deseasonalized",
    "n_days",
    "stdev_days",
    "unc_monthly_mean",
]


def download_co2(destination: Path = RAW_DATA_PATH) -> Path:
    """Download the raw NOAA CO2 file if it isn't already cached locally."""
    if destination.exists():
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(CO2_SOURCE_URL, timeout=30)
    response.raise_for_status()
    destination.write_text(response.text)
    return destination


def load_co2(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Parse the cached NOAA CO2 file into a tidy, time-ordered DataFrame."""
    df = pd.read_csv(
        path,
        comment="#",
        sep=r"\s+",
        names=COLUMN_NAMES,
        header=None,
    )
    df = df[df["co2_ppm"] != MISSING_VALUE_SENTINEL]
    df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))
    df = df.sort_values("date").reset_index(drop=True)
    return df[["date", "year", "month", "co2_ppm"]]
