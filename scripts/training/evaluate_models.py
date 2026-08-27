"""Evaluate trained image CNN on the held-out test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _training_utils import build_arrays, ensure_backend_on_path, load_split_files, save_metrics

ensure_backend_on_path()

from ml.features.image_features import DEFAULT_INPUT_SIZE, load_preprocessing_config  # noqa: E402
from ml.inference.image_predictor import ImagePredictor  # noqa: E402


def evaluate(data_dir: Path, model_dir: Path, reports_dir: Path) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    model_path = model_dir / "model.keras"
    if not model_path.is_file():
        raise FileNotFoundError(
            f"No trained model at {model_path}. Train first with scripts/training/train_image_model.py"
        )

    prep_path = model_dir / "preprocessing_config.json"
    if prep_path.is_file():
        prep = load_preprocessing_config(prep_path)
        input_size = tuple(prep["input_size"])
    else:
        input_size = DEFAULT_INPUT_SIZE

    test_pairs = load_split_files(data_dir, "test")
    if not test_pairs:
        raise FileNotFoundError(f"No test files found under {data_dir}/cover/test and stego/test")

    x_test, y_test = build_arrays(test_pairs, input_size)
    predictor = ImagePredictor(model_path, model_dir / "metadata.json", prep_path)
    predictor.load()

    y_prob = np.array([predictor.predict_file(path) for path, _ in test_pairs])
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "test_samples": int(len(test_pairs)),
        "model_dir": str(model_dir),
        "data_dir": str(data_dir),
    }

    reports_dir.mkdir(parents=True, exist_ok=True)
    save_metrics(reports_dir / "image_eval_metrics.json", metrics)

    # Save confusion matrix as readable text
    cm = np.array(metrics["confusion_matrix"])
    cm_text = (
        "Confusion matrix (rows=true, cols=pred) [cover=0, stego=1]:\n"
        f"  [[TN={cm[0,0]}, FP={cm[0,1]}],\n"
        f"   [FN={cm[1,0]}, TP={cm[1,1]}]]\n"
    )
    (reports_dir / "image_confusion_matrix.txt").write_text(cm_text, encoding="utf-8")

    print(json.dumps({k: v for k, v in metrics.items() if k != "confusion_matrix"}, indent=2))
    print(cm_text)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate image CNN on test split")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/images"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/image_cnn"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports/eval"))
    args = parser.parse_args()
    evaluate(args.data_dir, args.model_dir, args.reports_dir)


if __name__ == "__main__":
    main()
