"""
FINAL INDEPENDENT EVALUATION
AI Multimedia Steganography Detection

Evaluates the already-trained classical Random Forest model
on the untouched test set.

IMPORTANT:
- Does NOT retrain the model.
- Does NOT modify the model.
- Does NOT use validation data.
- Uses the saved threshold from metadata.
- Reports overall and payload-level performance.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DATASET = ROOT / "data" / "processed" / "images"
MODEL_DIR = ROOT / "models" / "classical_stego"

MODEL_PATH = MODEL_DIR / "best_model.pkl"
METADATA_PATH = MODEL_DIR / "metadata.json"

# Import the EXACT feature extractor used during training.
sys.path.insert(0, str(ROOT / "backend"))

from ml.features.stego_features import extract_features_from_file


# ---------------------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------------------

print("=" * 75)
print("FINAL INDEPENDENT CLASSICAL MODEL EVALUATION")
print("=" * 75)

if not MODEL_PATH.exists():
    print()
    print("ERROR: Trained model not found:")
    print(MODEL_PATH)
    raise SystemExit(1)

if not METADATA_PATH.exists():
    print()
    print("ERROR: Model metadata not found:")
    print(METADATA_PATH)
    raise SystemExit(1)

print()
print("Loading trained model...")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

threshold = float(metadata["selected_threshold"])

print("Model              :", metadata.get("model_type", "Unknown"))
print("Feature count      :", metadata.get("feature_count", "Unknown"))
print("Decision threshold :", f"{threshold:.6f}")


# ---------------------------------------------------------------------
# COLLECT TEST DATA
# ---------------------------------------------------------------------

cover_dir = DATASET / "cover" / "test"
stego_dir = DATASET / "stego" / "test"

cover_files = sorted(cover_dir.glob("*.png"))
stego_files = sorted(stego_dir.glob("*.png"))

print()
print("=" * 75)
print("TEST DATASET")
print("=" * 75)

print("Cover test images :", len(cover_files))
print("Stego test images :", len(stego_files))
print("Total             :", len(cover_files) + len(stego_files))


# ---------------------------------------------------------------------
# FEATURE EXTRACTION
# ---------------------------------------------------------------------

def extract_dataset(files, label, name):
    features = []
    labels = []
    paths = []

    total = len(files)

    print()
    print("-" * 75)
    print(f"EXTRACTING FEATURES: {name}")
    print("-" * 75)

    start = time.time()

    for i, path in enumerate(files, 1):

        try:
            vector = extract_features_from_file(path)

            if vector.shape != (99,):
                raise ValueError(
                    f"Expected 99 features, got {vector.shape}"
                )

            if not np.isfinite(vector).all():
                raise ValueError("Feature vector contains NaN/Inf")

            features.append(vector)
            labels.append(label)
            paths.append(path)

        except Exception as exc:
            print()
            print("ERROR processing:", path)
            print("Reason:", exc)
            raise

        if i % 500 == 0 or i == total:
            elapsed = time.time() - start
            print(
                f"Processed {i}/{total} "
                f"({elapsed:.1f}s)"
            )

    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.int32),
        paths,
    )


X_cover, y_cover, paths_cover = extract_dataset(
    cover_files,
    0,
    "COVER TEST SET",
)

X_stego, y_stego, paths_stego = extract_dataset(
    stego_files,
    1,
    "STEGO TEST SET",
)


# ---------------------------------------------------------------------
# COMBINE
# ---------------------------------------------------------------------

X_test = np.vstack([X_cover, X_stego])
y_test = np.concatenate([y_cover, y_stego])

paths = paths_cover + paths_stego

print()
print("=" * 75)
print("FEATURE MATRIX")
print("=" * 75)

print("X_test shape :", X_test.shape)
print("y_test shape :", y_test.shape)

if X_test.shape[1] != 99:
    print()
    print("ERROR: Feature count mismatch.")
    print("Expected: 99")
    print("Actual  :", X_test.shape[1])
    raise SystemExit(1)


# ---------------------------------------------------------------------
# PREDICTIONS
# ---------------------------------------------------------------------

print()
print("=" * 75)
print("RUNNING FINAL MODEL")
print("=" * 75)

probabilities = model.predict_proba(X_test)[:, 1]

predictions = (
    probabilities >= threshold
).astype(np.int32)


# ---------------------------------------------------------------------
# OVERALL METRICS
# ---------------------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    predictions,
)

balanced_accuracy = balanced_accuracy_score(
    y_test,
    predictions,
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0,
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0,
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0,
)

auc = roc_auc_score(
    y_test,
    probabilities,
)

cm = confusion_matrix(
    y_test,
    predictions,
)


# ---------------------------------------------------------------------
# PRINT RESULTS
# ---------------------------------------------------------------------

print()
print("=" * 75)
print("FINAL INDEPENDENT TEST RESULTS")
print("=" * 75)

print()
print(f"Accuracy              : {accuracy:.6f}")
print(f"Balanced Accuracy     : {balanced_accuracy:.6f}")
print(f"Precision             : {precision:.6f}")
print(f"Recall                : {recall:.6f}")
print(f"F1 Score              : {f1:.6f}")
print(f"ROC-AUC               : {auc:.6f}")
print(f"Decision Threshold     : {threshold:.6f}")


# ---------------------------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------------------------

print()
print("=" * 75)
print("CONFUSION MATRIX")
print("=" * 75)

print()
print("                    Predicted")
print("                  Cover     Stego")
print()
print(
    f"Actual Cover      {cm[0,0]:6d}   {cm[0,1]:6d}"
)
print(
    f"Actual Stego      {cm[1,0]:6d}   {cm[1,1]:6d}"
)


# ---------------------------------------------------------------------
# ERROR COUNTS
# ---------------------------------------------------------------------

false_positives = int(cm[0, 1])
false_negatives = int(cm[1, 0])
true_negatives = int(cm[0, 0])
true_positives = int(cm[1, 1])

cover_total = true_negatives + false_positives
stego_total = true_positives + false_negatives

cover_fpr = (
    false_positives / cover_total
    if cover_total
    else 0
)

stego_fnr = (
    false_negatives / stego_total
    if stego_total
    else 0
)

print()
print("=" * 75)
print("ERROR ANALYSIS")
print("=" * 75)

print()
print("True negatives       :", true_negatives)
print("False positives      :", false_positives)
print("True positives       :", true_positives)
print("False negatives      :", false_negatives)

print()
print(
    f"Cover false-positive rate : {cover_fpr:.6f}"
)

print(
    f"Stego false-negative rate : {stego_fnr:.6f}"
)


# ---------------------------------------------------------------------
# CLASSIFICATION REPORT
# ---------------------------------------------------------------------

print()
print("=" * 75)
print("CLASSIFICATION REPORT")
print("=" * 75)

print()

print(
    classification_report(
        y_test,
        predictions,
        target_names=["Cover", "Stego"],
        digits=4,
        zero_division=0,
    )
)


# ---------------------------------------------------------------------
# PAYLOAD-LEVEL ANALYSIS
# ---------------------------------------------------------------------

print()
print("=" * 75)
print("PAYLOAD-LEVEL STEGO ANALYSIS")
print("=" * 75)

payload_groups = {
    "0.10": [],
    "0.30": [],
    "0.50": [],
}

for path, probability, prediction in zip(
    paths_stego,
    probabilities[len(paths_cover):],
    predictions[len(paths_cover):],
):

    filename = path.name.lower()

    if "rate0.10" in filename:
        payload_groups["0.10"].append(
            (probability, prediction)
        )

    elif "rate0.30" in filename:
        payload_groups["0.30"].append(
            (probability, prediction)
        )

    elif "rate0.50" in filename:
        payload_groups["0.50"].append(
            (probability, prediction)
        )


for payload, results in payload_groups.items():

    if not results:
        print()
        print(f"Payload {payload}: no matching files")
        continue

    probs = np.asarray(
        [r[0] for r in results],
        dtype=np.float32,
    )

    preds = np.asarray(
        [r[1] for r in results],
        dtype=np.int32,
    )

    detected = int(preds.sum())
    total = len(preds)

    detection_rate = detected / total

    print()
    print(f"Payload rate : {payload}")
    print(f"Samples      : {total}")
    print(f"Detected     : {detected}")
    print(f"Missed       : {total - detected}")
    print(
        f"Detection rate : {detection_rate:.6f}"
    )
    print(
        f"Mean probability : {probs.mean():.6f}"
    )


# ---------------------------------------------------------------------
# SAVE FINAL REPORT
# ---------------------------------------------------------------------

report = {
    "model_type": metadata.get("model_type"),
    "feature_count": metadata.get("feature_count"),
    "decision_threshold": threshold,
    "test_samples": int(len(y_test)),
    "cover_samples": int(len(y_cover)),
    "stego_samples": int(len(y_stego)),
    "accuracy": float(accuracy),
    "balanced_accuracy": float(balanced_accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "roc_auc": float(auc),
    "true_negative": true_negatives,
    "false_positive": false_positives,
    "true_positive": true_positives,
    "false_negative": false_negatives,
    "cover_false_positive_rate": float(cover_fpr),
    "stego_false_negative_rate": float(stego_fnr),
    "payload_analysis": {},
}

for payload, results in payload_groups.items():

    if results:
        preds = np.asarray(
            [r[1] for r in results],
            dtype=np.int32,
        )

        report["payload_analysis"][payload] = {
            "samples": int(len(preds)),
            "detected": int(preds.sum()),
            "missed": int(len(preds) - preds.sum()),
            "detection_rate": float(preds.mean()),
        }

REPORT_PATH = MODEL_DIR / "final_test_report.json"

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        report,
        f,
        indent=2,
    )


# ---------------------------------------------------------------------
# FINAL STATUS
# ---------------------------------------------------------------------

print()
print("=" * 75)
print("FINAL EVALUATION COMPLETE")
print("=" * 75)

print()
print("Report saved:")
print(REPORT_PATH)

print()
print("IMPORTANT:")
print("This evaluation uses the independent test set.")
print("The model was NOT retrained during this evaluation.")
print("The test set was NOT used to select the model.")

print()
print("=" * 75)