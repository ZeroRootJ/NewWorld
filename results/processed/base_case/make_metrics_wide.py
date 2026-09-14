"""Pivot results/processed/base_case/metrics.csv (tidy long-format,
case/axis/axis_level/method/metric/value) into a wide table (rows=method,
columns=mse/umg) for easier reading in the report.

Run with: .venv/Scripts/python.exe results/processed/base_case/make_metrics_wide.py
"""

from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PROCESSED_DIR = _REPO_ROOT / "results" / "processed" / "base_case"


def main():
    df = pd.read_csv(PROCESSED_DIR / "metrics.csv")
    wide = df.pivot(index="method", columns="metric", values="value").reset_index()
    wide = wide[["method", "mse", "umg"]]
    out_path = PROCESSED_DIR / "metrics_wide.csv"
    wide.to_csv(out_path, index=False)
    print(wide.to_string(index=False))
    print(f"\nmetrics_wide.csv: {out_path}")
    return wide


if __name__ == "__main__":
    main()
