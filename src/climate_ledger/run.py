"""Entry point: download data, run the benchmark, write the leaderboard."""

from climate_ledger.data import download_co2, load_co2
from climate_ledger.evaluate import run_benchmark
from climate_ledger.leaderboard import write_leaderboard

DATASET_NAME = "Mauna Loa CO2 (NOAA/Scripps Keeling Curve)"


def main() -> None:
    download_co2()
    df = load_co2()
    results = run_benchmark(df)
    path = write_leaderboard(results, DATASET_NAME)
    print(results.to_string(index=False))
    print(f"\nLeaderboard written to {path}")


if __name__ == "__main__":
    main()
