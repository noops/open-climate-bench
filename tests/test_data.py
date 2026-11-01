from pathlib import Path

from climate_ledger.data import load_co2

FIXTURE_TEXT = """\
# header comment line
# another comment
#            decimal       monthly    de-season  #days  st.dev  unc. of
#             date         average     alized          of days  mon mean
 1958    3   1958.2027      315.71      314.44     -1   -9.99   -0.99
 1958    4   1958.2877      -99.99      315.16     -1   -9.99   -0.99
 1958    5   1958.3699      317.51      314.69     -1   -9.99   -0.99
"""


def test_load_co2_drops_missing_value_sentinel_rows(tmp_path: Path):
    raw_file = tmp_path / "co2_mm_mlo.txt"
    raw_file.write_text(FIXTURE_TEXT)

    df = load_co2(raw_file)

    assert -99.99 not in df["co2_ppm"].values


def test_load_co2_returns_dates_in_ascending_order(tmp_path: Path):
    raw_file = tmp_path / "co2_mm_mlo.txt"
    raw_file.write_text(FIXTURE_TEXT)

    df = load_co2(raw_file)

    assert list(df["date"]) == sorted(df["date"])
