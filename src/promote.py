import argparse
import shutil
from pathlib import Path

from src.config import MIN_F1, MODELS_LATEST_DIR
from src.utils import ensure_dir, read_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--eval-report", type=str, required=True)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    eval_report = Path(args.eval_report)

    metrics = read_json(eval_report)
    f1 = float(metrics["test_f1"])

    if f1 < MIN_F1:
        raise RuntimeError(f"Refusing promotion, F1 {f1:.4f} < MIN_F1 {MIN_F1:.2f}")

    ensure_dir(MODELS_LATEST_DIR)

    for name in ["model.joblib", "metadata.json"]:
        src = model_dir / name
        dst = MODELS_LATEST_DIR / name
        if not src.exists():
            raise FileNotFoundError(f"Missing {name} in {model_dir}")
        shutil.copy2(src, dst)

    print(f"Promoted model to {MODELS_LATEST_DIR}")
    print(f"Promoted version: {metrics['model_version']} with test_f1={f1:.4f}")

if __name__ == "__main__":
    main()
