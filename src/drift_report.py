"""Generate a data drift report comparing train vs current data."""

import argparse
from pathlib import Path

from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from sklearn.model_selection import train_test_split

from src.config import ARTIFACTS_DIR, DATA_RAW, RANDOM_STATE
from src.features import load_dataset, split_xy
from src.utils import ensure_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=str(DATA_RAW))
    parser.add_argument("--report-dir", type=str, default=str(ARTIFACTS_DIR))
    args = parser.parse_args()

    df = load_dataset(args.data)
    X, _ = split_xy(df)

    # Split into reference (training) and current (validation) sets
    X_ref, X_curr = train_test_split(X, test_size=0.25, random_state=RANDOM_STATE)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=X_ref, current_data=X_curr)

    report_dir = Path(args.report_dir)
    ensure_dir(report_dir)

    html_path = report_dir / "drift_report.html"
    report.save_html(str(html_path))
    print(f"Drift report saved to: {html_path}")

    # Also save JSON summary
    json_path = report_dir / "drift_report.json"
    report.save_json(str(json_path))
    print(f"Drift JSON saved to: {json_path}")


if __name__ == "__main__":
    main()
