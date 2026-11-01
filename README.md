# Climate Ledger

A documented, reproducible benchmark of public climate datasets and model evals.

Status: idea. Part of [Bat City AI](https://batcityai.com)'s Year One programs. (Formerly "Open Climate Bench" — renamed to avoid confusion with the existing [ClimateBench](https://github.com/duncanwp/ClimateBench) project.)

## v0

- **Data**: [NOAA/Scripps Mauna Loa monthly CO2 record](https://gml.noaa.gov/ccgg/trends/) (the Keeling Curve)
- **Stack**: Python, pandas, scikit-learn, XGBoost
- **Task**: forecast CO2 ppm 5 years ahead, trained on everything before the holdout window
- **Baselines**: persistence (flat), linear trend + seasonal, XGBoost
- **Output**: static leaderboard at [`results/leaderboard.md`](results/leaderboard.md)

## Running it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest -q          # tests
python -m climate_ledger.run # downloads data, runs the benchmark, writes the leaderboard
```

## Status

v0 pipeline works end to end. Current result: XGBoost beats both baselines; a pure
linear trend fit on the whole 1958–present record actually loses to naive persistence,
because CO2 growth has accelerated over the record and a single straight-line fit
undershoots the recent rate. See [Issues](../../issues) for what's next.

## License

MIT
