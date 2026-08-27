"""
FINAL TEST EVALUATION - 294 FEATURE MODEL

Uses the untouched 6,000-image test set.

Model:
    models/classical_stego/best_model_294.pkl

Metadata:
    models/classical_stego/best_model_294_metadata.json

Features:
    Improved 294-feature extractor

IMPORTANT:
    This test set is NOT used for training or model selection.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# =====================================================================
# PATH
# =====================================================================

ROOT = Path(__file__).resolve().parents[2]

BACKEND = ROOT / "backend"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


from ml.features.stego_features import (
    extract_features_from_file,
)


# =====================================================================
# MODEL
# =====================================================================

MODEL_PATH = (
    ROOT
    / "models"
    / "classical_stego"
    / "best_model_294.pkl"
)

METADATA_PATH = (
    ROOT
    / "models"
    / "classical_stego"
    / "best_model_294_metadata.json"
)


# =====================================================================
# TEST DIRECTORIES
# =====================================================================

IMAGE_ROOT = (
    ROOT
    / "data"
    / "processed"
    / "images"
)

COVER_TEST = (
    IMAGE_ROOT
    / "cover"
    / "test"
)

STEGO_TEST = (
    IMAGE_ROOT
    / "stego"
    / "test"
)


# =====================================================================
# OUTPUT
# =====================================================================

OUTPUT_DIR = (
    ROOT
    / "models"
    / "classical_stego"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULTS_PATH = (
    OUTPUT_DIR
    / "test_results_294.json"
)


# =====================================================================
# IMAGE EXTENSIONS
# =====================================================================

EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


# =====================================================================
# GET IMAGES
# =====================================================================

def get_images(directory):

    files = [
        p
        for p in directory.rglob("*")
        if (
            p.is_file()
            and p.suffix.lower()
            in EXTENSIONS
        )
    ]

    return sorted(files)


# =====================================================================
# EXTRACT DATASET
# =====================================================================

def extract_dataset(
    directory,
    label,
):

    images = get_images(
        directory
    )

    print()
    print(
        f"Directory : {directory}"
    )

    print(
        f"Images    : {len(images)}"
    )

    if not images:

        raise RuntimeError(
            f"No images found:\n{directory}"
        )

    X = []

    y = []

    start = time.time()

    for index, path in enumerate(
        images,
        start=1,
    ):

        features = (
            extract_features_from_file(
                path
            )
        )

        if len(features) != 294:

            raise RuntimeError(
                f"Expected 294 features, "
                f"got {len(features)}\n"
                f"Image: {path}"
            )

        X.append(
            features
        )

        y.append(
            label
        )

        if (
            index % 500 == 0
            or index == len(images)
        ):

            elapsed = (
                time.time()
                - start
            )

            rate = (
                index / elapsed
                if elapsed > 0
                else 0
            )

            remaining = (
                len(images)
                - index
            )

            eta = (
                remaining / rate
                if rate > 0
                else 0
            )

            print(
                f"Processed "
                f"{index}/{len(images)} "
                f"| {rate:.2f} img/s "
                f"| ETA {eta/60:.1f} min"
            )

    return (
        np.asarray(
            X,
            dtype=np.float32,
        ),
        np.asarray(
            y,
            dtype=np.int32,
        ),
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    print("=" * 70)
    print(
        "FINAL 294-FEATURE TEST EVALUATION"
    )
    print("=" * 70)

    # ---------------------------------------------------------------
    # CHECK MODEL
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
    print(
        "Loading model..."
    )

    with open(
        MODEL_PATH,
        "rb",
    ) as file:

        model = pickle.load(
            file
        )

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        metadata = json.load(
            file
        )

    threshold = float(
        metadata["threshold"]
    )

    print(
        f"Model     : "
        f"{metadata['model_name']}"
    )

    print(
        f"Features  : "
        f"{metadata['feature_count']}"
    )

    print(
        f"Threshold : "
        f"{threshold:.4f}"
    )

    # ---------------------------------------------------------------
    # CHECK TEST DATA
    # ---------------------------------------------------------------

    if not COVER_TEST.exists():

        raise FileNotFoundError(
            f"Cover test directory not found:\n"
            f"{COVER_TEST}"
        )

    if not STEGO_TEST.exists():

        raise FileNotFoundError(
            f"Stego test directory not found:\n"
            f"{STEGO_TEST}"
        )

    # ---------------------------------------------------------------
    # COVER
    # ---------------------------------------------------------------

    X_cover, y_cover = (
        extract_dataset(
            COVER_TEST,
            0,
        )
    )

    # ---------------------------------------------------------------
    # STEGO
    # ---------------------------------------------------------------

    X_stego, y_stego = (
        extract_dataset(
            STEGO_TEST,
            1,
        )
    )

    # ---------------------------------------------------------------
    # COMBINE
    # ---------------------------------------------------------------

    X_test = np.concatenate(
        [
            X_cover,
            X_stego,
        ],
        axis=0,
    )

    y_test = np.concatenate(
        [
            y_cover,
            y_stego,
        ],
        axis=0,
    )

    print()
    print("=" * 70)
    print(
        "TEST DATASET"
    )
    print("=" * 70)

    print(
        f"Total samples : "
        f"{len(y_test)}"
    )

    print(
        f"Cover         : "
        f"{np.sum(y_test == 0)}"
    )

    print(
        f"Stego         : "
        f"{np.sum(y_test == 1)}"
    )

    print(
        f"Features      : "
        f"{X_test.shape[1]}"
    )

    # ---------------------------------------------------------------
    # PREDICTION
    # ---------------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "TEST PREDICTION"
    )
    print("=" * 70)

    probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    predictions = (
        probabilities >= threshold
    ).astype(np.int32)

    # ---------------------------------------------------------------
    # METRICS
    # ---------------------------------------------------------------

    auc = roc_auc_score(
        y_test,
        probabilities,
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    balanced = (
        balanced_accuracy_score(
            y_test,
            predictions,
        )
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

    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    report = classification_report(
        y_test,
        predictions,
        target_names=[
            "Cover",
            "Stego",
        ],
        zero_division=0,
    )

    # ---------------------------------------------------------------
    # RESULTS
    # ---------------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "FINAL TEST RESULTS"
    )
    print("=" * 70)

    print(
        f"AUC                : "
        f"{auc:.6f}"
    )

    print(
        f"Accuracy           : "
        f"{accuracy:.6f}"
    )

    print(
        f"Balanced Accuracy  : "
        f"{balanced:.6f}"
    )

    print(
        f"Precision          : "
        f"{precision:.6f}"
    )

    print(
        f"Recall             : "
        f"{recall:.6f}"
    )

    print(
        f"F1                 : "
        f"{f1:.6f}"
    )

    # ---------------------------------------------------------------
    # CONFUSION MATRIX
    # ---------------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "CONFUSION MATRIX"
    )
    print("=" * 70)

    print(
        "                  Predicted"
    )

    print(
        "                  Cover    Stego"
    )

    print(
        f"Actual Cover     "
        f"{matrix[0,0]:7d}"
        f"{matrix[0,1]:9d}"
    )

    print(
        f"Actual Stego     "
        f"{matrix[1,0]:7d}"
        f"{matrix[1,1]:9d}"
    )

    # ---------------------------------------------------------------
    # CLASSIFICATION REPORT
    # ---------------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "CLASSIFICATION REPORT"
    )
    print("=" * 70)

    print(
        report
    )

    # ---------------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------------

    results = {

        "model":
            metadata["model_name"],

        "feature_count":
            294,

        "threshold":
            threshold,

        "test_samples":
            int(len(y_test)),

        "test_cover":
            int(np.sum(y_test == 0)),

        "test_stego":
            int(np.sum(y_test == 1)),

        "auc":
            float(auc),

        "accuracy":
            float(accuracy),

        "balanced_accuracy":
            float(balanced),

        "precision":
            float(precision),

        "recall":
            float(recall),

        "f1":
            float(f1),

        "confusion_matrix":
            matrix.tolist(),

        "classification_report":
            report,
    }

    with open(
        RESULTS_PATH,
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
    print(
        "TEST RESULTS SAVED"
    )
    print("=" * 70)

    print(
        f"File: {RESULTS_PATH}"
    )


if __name__ == "__main__":
    main()