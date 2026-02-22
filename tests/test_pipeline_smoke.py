from pathlib import Path
import json
import os
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parent.parent


def run(cmd):
    env = {**os.environ, "MLFLOW_TRACKING_URI": "./mlruns"}
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env)
    if p.returncode != 0:
        print(p.stdout)
        print(p.stderr, file=sys.stderr)
    assert p.returncode == 0
    return p.stdout


@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Skipped in CI — train/eval/promote steps already run explicitly",
)
def test_train_and_eval_smoke(tmp_path):
    out_dir = tmp_path / "models"
    report_dir = tmp_path / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    run(
        [
            sys.executable,
            "-m",
            "src.train",
            "--out-dir",
            str(out_dir),
            "--report-dir",
            str(report_dir),
        ]
    )
    train_report = json.loads((report_dir / "train_report.json").read_text())
    model_version = train_report["model_version"]

    model_dir = out_dir / model_version
    assert (model_dir / "model.joblib").exists()
    assert (model_dir / "metadata.json").exists()

    run(
        [
            sys.executable,
            "-m",
            "src.evaluate",
            "--model-dir",
            str(model_dir),
            "--report-dir",
            str(report_dir),
        ]
    )
    eval_report = json.loads((report_dir / "eval_report.json").read_text())
    assert "test_f1" in eval_report
