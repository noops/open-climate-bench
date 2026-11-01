"""Render benchmark results to a static markdown leaderboard."""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

LEADERBOARD_PATH = Path(__file__).resolve().parents[2] / "results" / "leaderboard.md"


def render_leaderboard(results: pd.DataFrame, dataset_name: str) -> str:
    generated_on = datetime.now(timezone.utc).date().isoformat()
    lines = [
        f"# {dataset_name} Leaderboard",
        "",
        f"Generated: {generated_on}",
        "",
        "| Rank | Model | RMSE (ppm) | MAPE (%) |",
        "|---|---|---|---|",
    ]
    for rank, row in enumerate(results.itertuples(index=False), start=1):
        lines.append(f"| {rank} | {row.model} | {row.rmse_ppm} | {row.mape_pct} |")
    return "\n".join(lines) + "\n"


def write_leaderboard(results: pd.DataFrame, dataset_name: str, path: Path = LEADERBOARD_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_leaderboard(results, dataset_name))
    return path
