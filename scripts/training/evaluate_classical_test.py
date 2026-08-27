"""
Final evaluation of the saved classical steganography detector.

IMPORTANT:
- No model training occurs here.
- The test set is used only for final evaluation.
- The threshold was selected using validation data.
- The test set is NOT used to tune the model.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

BACKEND = ROOT / "backend"

# Both are required because the project's backend packages use imports
# such as "ml.features..."
for path in (ROOT, BACKEND):
    path_string = str(path)
    if path_string not in sys.path:
        sys.path.insert(0, path_string)


# ---------------------------------------------------------------------
# THIRD-PARTY IMPORTS
# ---------------------------------------------------------------------

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
# PROJECT IMPORT
# ---------------------------------------------------------------------

from ml.features.stego_features import (
    extract_features_from_file,
)


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

TEST_ROOT = (
    ROOT
    / "data"
    / "processed"
    / "images"
)

MODEL_PATH = (
    ROOT
    / "models"
    / "classical_stego"
    / "best_model.pkl"
)

METADATA_PATH = (
    ROOT
    / "models"
    / "classical_stego"
    / "best_model_metadata.json"
)

OUTPUT_PATH = (
    ROOT
    / "models"
    / "classical_stego"
    / "test_results.json"
)


# ---------------------------------------------------------------------
# IMAGE EXTENSIONS
# ---------------------------------------------------------------------

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


# ---------------------------------------------------------------------
# COLLECT TEST IMAGES
# ---------------------------------------------------------------------

def collect_test_images():

    cover_dir = (
        TEST_ROOT
        / "cover"
        / "test"
    )

    stego_dir = (
        TEST_ROOT
        / "stego"
        / "test"
    )

    if not cover_dir.exists():
        raise FileNotFoundError(
            f"Cover test directory not found:\n{cover_dir}"
        )

    if not stego_dir.exists():
        raise FileNotFoundError(
            f"Stego test directory not found:\n{stego_dir}"
        )

    cover_files = sorted(
        [
            path
            for path in cover_dir.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        ]
    )

    stego_files = sorted(
        [
            path
            for path in stego_dir.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        ]
    )

    paths = cover_files + stego_files

    labels = (
        [0] * len(cover_files)
        + [1] * len(stego_files)
    )

    return (
        paths,
        np.asarray(
            labels,
            dtype=np.int32,
        ),
    )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    print("=" * 70)
    print("CLASSICAL STEGANOGRAPHY TEST EVALUATION")
    print("=" * 70)

    # ---------------------------------------------------------------
    # CHECK MODEL FILES
    # ---------------------------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata not found:\n{METADATA_PATH}"
        )

    # ---------------------------------------------------------------
    # LOAD MODEL
    # ---------------------------------------------------------------

    print()
    print("Loading saved model...")

    with open(
        MODEL_PATH,
        "rb",
    ) as file:
        model = pickle.load(file)

    # ---------------------------------------------------------------
    # LOAD METADATA
    # ---------------------------------------------------------------

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    threshold = float(
        metadata["threshold"]
    )

    print(
        f"Model     : {metadata['model_name']}"
    )

    print(
        f"Threshold : {threshold:.4f}"
    )

    # ---------------------------------------------------------------
    # COLLECT TEST DATA
    # ---------------------------------------------------------------

    paths, y_true = collect_test_images()

    print()
    print("=" * 70)
    print("TEST DATASET")
    print("=" * 70)

    print(
        f"Test images : {len(paths)}"
    )

    print(
        f"Cover       : {np.sum(y_true == 0)}"
    )

    print(
        f"Stego       : {np.sum(y_true == 1)}"
    )

    if len(paths) == 0:
        raise RuntimeError(
            "No test images found."
        )

    # ---------------------------------------------------------------
    # EXTRACT TEST FEATURES
    # ---------------------------------------------------------------

    print()
    print("=" * 70)
    print("EXTRACTING TEST FEATURES")
    print("=" * 70)

    start_time = time.time()

    features = []

    for index, path in enumerate(
        paths,
        start=1,
    ):

        feature_vector = (
            extract_features_from_file(path)
        )

        features.append(
            feature_vector
        )

        if (
            index % 250 == 0
            or index == len(paths)
        ):

            elapsed = (
                time.time()
                - start_time
            )

            print(
                f"Processed "
                f"{index}/{len(paths)} "
                f"({elapsed:.1f}s)"
            )

    X_test = np.asarray(
        features,
        dtype=np.float32,
    )

    print()
    print(
        f"Feature matrix: {X_test.shape}"
    )

    # ---------------------------------------------------------------
    # VALIDATE FEATURES
    # ---------------------------------------------------------------

    if X_test.ndim != 2:
        raise RuntimeError(
            "Test feature matrix must be 2-dimensional."
        )

    if X_test.shape[1] != 99:
        raise RuntimeError(
            f"Expected 99 features, "
            f"got {X_test.shape[1]}"
        )

    if not np.isfinite(X_test).all():
        raise RuntimeError(
            "Test feature matrix contains "
            "non-finite values."
        )

    # ---------------------------------------------------------------
    # PREDICTION
    # ---------------------------------------------------------------

    print()
    print("=" * 70)
    print("TEST PREDICTION")
    print("=" * 70)

    probabilities = (
        model.predict_proba(X_test)[:, 1]
    )

    predictions = (
        probabilities >= threshold
    ).astype(np.int32)

    # ---------------------------------------------------------------
    # METRICS
    # ---------------------------------------------------------------

    auc = roc_auc_score(
        y_true,
        probabilities,
    )

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_true,
            predictions,
        )
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    confusion = confusion_matrix(
        y_true,
        predictions,
    )

    # ---------------------------------------------------------------
    # FINAL RESULTS
    # ---------------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    print(
        f"AUC                : {auc:.6f}"
    )

    print(
        f"Accuracy           : {accuracy:.6f}"
    )

    print(
        f"Balanced Accuracy  : {balanced_accuracy:.6f}"
    )

    print(
        f"Precision          : {precision:.6f}"
    )

    print(
        f"Recall             : {recall:.6f}"
    )

    print(
        f"F1                 : {f1:.6f}"
    )

    # ---------------------------------------------------------------
    # CONFUSION MATRIX
    # ---------------------------------------------------------------

    print()
    print("=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    print(
        "                  Predicted"
    )

    print(
        "                  Cover    Stego"
    )

    print(
        f"Actual Cover      "
        f"{confusion[0, 0]:6d}   "
        f"{confusion[0, 1]:6d}"
    )

    print(
        f"Actual Stego      "
        f"{confusion[1, 0]:6d}   "
        f"{confusion[1, 1]:6d}"
    )

    # ---------------------------------------------------------------
    # CLASSIFICATION REPORT
    # ---------------------------------------------------------------

    print()
    print("=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    report = classification_report(
        y_true,
        predictions,
        target_names=[
            "Cover",
            "Stego",
        ],
        zero_division=0,
    )

    print(report)

    # ---------------------------------------------------------------
    # SAVE RESULTS
    # ---------------------------------------------------------------

    results = {
        "model": metadata["model_name"],
        "feature_count": 99,
        "threshold": threshold,
        "test_samples": int(len(paths)),
        "test_cover": int(np.sum(y_true == 0)),
        "test_stego": int(np.sum(y_true == 1)),
        "auc": float(auc),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(
            balanced_accuracy
        ),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": confusion.tolist(),
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
        )

    print()
    print("=" * 70)
    print("TEST RESULTS SAVED")
    print("=" * 70)

    print(
        f"File: {OUTPUT_PATH}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()