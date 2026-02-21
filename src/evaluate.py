import argparse
from pathlib import Path

import joblib
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

from src.config import ARTIFACTS_DIR, DATA_RAW, RANDOM_STATE
from src.features import load_dataset, split_xy
from src.utils import read_json, write_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--data", type=str, default=str(DATA_RAW))
    parser.add_argument("--report-dir", type=str, default=str(ARTIFACTS_DIR))
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    model_path = model_dir / "model.joblib"
    meta_path = model_dir / "metadata.json"

    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError("model.joblib or metadata.json not found in model-dir")

    model = joblib.load(model_path)
    meta = read_json(meta_path)

    df = load_dataset(args.data)
    X, y = split_xy(df)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    pred = model.predict(X_test)
    f1 = float(f1_score(y_test, pred))
    report = classification_report(y_test, pred, output_dict=True)

    payload = {
        "model_version": meta["model_version"],
        "test_f1": f1,
        "classification_report": report,
    }

    out_path = Path(args.report_dir) / "eval_report.json"
    write_json(out_path, payload)

    print(f"Eval F1: {f1:.4f}")
    print(f"Wrote report: {out_path}")

if __name__ == "__main__":
    main()
