"""
Robust classical steganography detector.

Pipeline:
    1. Extract the same 99 features used by inference.
    2. Train only on the training split.
    3. Select the best classifier using validation AUC.
    4. Select the classification threshold using validation data.
    5. Evaluate ONCE on the untouched test split.
    6. Save the trained model and metadata.

Important:
    The test set is NEVER used for model selection.
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import sys

# ---------------------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"

sys.path.insert(0, str(BACKEND))

from ml.features.stego_features import extract_features_from_file  # noqa: E402


DATASET = ROOT / "data" / "processed" / "images"
OUTPUT = ROOT / "models" / "classical_stego"

TRAIN = DATASET
VAL = DATASET
TEST = DATASET


# ---------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------

def collect_split(split: str):
    """
    Return image paths and labels.

    Label:
        0 = cover
        1 = stego
    """

    cover_dir = DATASET / "cover" / split
    stego_dir = DATASET / "stego" / split

    cover = sorted(cover_dir.glob("*.png"))
    stego = sorted(stego_dir.glob("*.png"))

    files = [(p, 0) for p in cover]
    files.extend((p, 1) for p in stego)

    return files


def extract_dataset(files, name: str):
    """
    Extract the 99-dimensional feature vector from every image.
    """

    print()
    print("=" * 70)
    print(f"EXTRACTING FEATURES: {name}")
    print("=" * 70)

    print(f"Samples: {len(files)}")

    X = []
    y = []

    start = time.time()

    for index, (path, label) in enumerate(files, start=1):

        features = extract_features_from_file(path)

        if features.shape != (99,):
            raise RuntimeError(
                f"Unexpected feature shape for {path}: "
                f"{features.shape}. Expected (99,)."
            )

        if not np.isfinite(features).all():
            raise RuntimeError(
                f"Non-finite feature values found in: {path}"
            )

        X.append(features)
        y.append(label)

        if index % 500 == 0 or index == len(files):
            elapsed = time.time() - start
            print(
                f"\rProcessed {index}/{len(files)} "
                f"({elapsed:.1f}s)",
                end="",
                flush=True,
            )

    print()

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)

    print(f"Feature matrix: {X.shape}")
    print(f"Labels        : {y.shape}")

    print(
        f"Cover         : {np.sum(y == 0)}"
    )
    print(
        f"Stego         : {np.sum(y == 1)}"
    )

    return X, y


# ---------------------------------------------------------------------
# THRESHOLD SELECTION
# ---------------------------------------------------------------------

def find_best_threshold(y_true, probabilities):
    """
    Select threshold ONLY using validation data.

    Objective:
        maximize balanced accuracy.

    This avoids simply choosing 0.5 when the dataset is imbalanced.
    """

    thresholds = np.linspace(0.05, 0.95, 181)

    best_threshold = 0.5
    best_score = -1.0

    for threshold in thresholds:

        pred = (probabilities >= threshold).astype(np.int32)

        score = balanced_accuracy_score(
            y_true,
            pred,
        )

        if score > best_score:

            best_score = score
            best_threshold = float(threshold)

    return best_threshold, best_score


# ---------------------------------------------------------------------
# MODEL EVALUATION
# ---------------------------------------------------------------------

def evaluate_validation(name, model, X_val, y_val):

    probabilities = model.predict_proba(X_val)[:, 1]

    auc = roc_auc_score(
        y_val,
        probabilities,
    )

    threshold, balanced_acc = find_best_threshold(
        y_val,
        probabilities,
    )

    predictions = (
        probabilities >= threshold
    ).astype(np.int32)

    accuracy = accuracy_score(
        y_val,
        predictions,
    )

    f1 = f1_score(
        y_val,
        predictions,
        zero_division=0,
    )

    return {
        "name": name,
        "auc": float(auc),
        "threshold": float(threshold),
        "balanced_accuracy": float(balanced_acc),
        "accuracy": float(accuracy),
        "f1": float(f1),
    }


# ---------------------------------------------------------------------
# FINAL TEST
# ---------------------------------------------------------------------

def evaluate_test(
    model,
    X_test,
    y_test,
    threshold,
):
    probabilities = model.predict_proba(X_test)[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(np.int32)

    auc = roc_auc_score(
        y_test,
        probabilities,
    )

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

    cm = confusion_matrix(
        y_test,
        predictions,
    )

    report = classification_report(
        y_test,
        predictions,
        target_names=["Cover", "Stego"],
        zero_division=0,
    )

    return {
        "auc": float(auc),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    print("=" * 70)
    print("AI MULTIMEDIA STEGANOGRAPHY DETECTION")
    print("CLASSICAL 99-FEATURE MODEL")
    print("=" * 70)

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # LOAD SPLITS
    # -------------------------------------------------------------

    train_files = collect_split("train")
    val_files = collect_split("val")
    test_files = collect_split("test")

    print()
    print("DATASET")
    print("-" * 70)

    print(
        f"Train: {len(train_files)} "
        f"({sum(label == 0 for _, label in train_files)} cover / "
        f"{sum(label == 1 for _, label in train_files)} stego)"
    )

    print(
        f"Val  : {len(val_files)} "
        f"({sum(label == 0 for _, label in val_files)} cover / "
        f"{sum(label == 1 for _, label in val_files)} stego)"
    )

    print(
        f"Test : {len(test_files)} "
        f"({sum(label == 0 for _, label in test_files)} cover / "
        f"{sum(label == 1 for _, label in test_files)} stego)"
    )

    if not train_files:
        raise RuntimeError("Training dataset is empty.")

    if not val_files:
        raise RuntimeError("Validation dataset is empty.")

    if not test_files:
        raise RuntimeError("Test dataset is empty.")

    # -------------------------------------------------------------
    # FEATURE EXTRACTION
    # -------------------------------------------------------------

    X_train, y_train = extract_dataset(
        train_files,
        "TRAIN",
    )

    X_val, y_val = extract_dataset(
        val_files,
        "VALIDATION",
    )

    X_test, y_test = extract_dataset(
        test_files,
        "TEST",
    )

    # -------------------------------------------------------------
    # FINAL SANITY CHECK
    # -------------------------------------------------------------

    if X_train.shape[1] != 99:
        raise RuntimeError(
            f"Expected 99 features, got {X_train.shape[1]}"
        )

    if X_val.shape[1] != 99:
        raise RuntimeError(
            f"Validation has {X_val.shape[1]} features."
        )

    if X_test.shape[1] != 99:
        raise RuntimeError(
            f"Test has {X_test.shape[1]} features."
        )

    print()
    print("=" * 70)
    print("FEATURE EXTRACTION VALIDATED")
    print("=" * 70)

    print("Train shape:", X_train.shape)
    print("Val shape  :", X_val.shape)
    print("Test shape :", X_test.shape)

    # -------------------------------------------------------------
    # MODELS
    # -------------------------------------------------------------

    models = {

        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),

        "RandomForest": RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),

        "LogisticRegression": Pipeline(
            [
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        C=1.0,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }

    # -------------------------------------------------------------
    # VALIDATION MODEL SELECTION
    # -------------------------------------------------------------

    print()
    print("=" * 70)
    print("MODEL SELECTION")
    print("Validation set ONLY")
    print("=" * 70)

    validation_results = []

    for name, model in models.items():

        print()
        print(f"Training: {name}")

        start = time.time()

        model.fit(
            X_train,
            y_train,
        )

        elapsed = time.time() - start

        result = evaluate_validation(
            name,
            model,
            X_val,
            y_val,
        )

        result["training_seconds"] = float(elapsed)

        validation_results.append(
            result
        )

        print(
            f"Validation AUC           : "
            f"{result['auc']:.6f}"
        )

        print(
            f"Validation threshold     : "
            f"{result['threshold']:.4f}"
        )

        print(
            f"Validation balanced acc. : "
            f"{result['balanced_accuracy']:.6f}"
        )

        print(
            f"Validation accuracy      : "
            f"{result['accuracy']:.6f}"
        )

        print(
            f"Validation F1            : "
            f"{result['f1']:.6f}"
        )

    # -------------------------------------------------------------
    # SELECT BEST MODEL BY VALIDATION AUC
    # -------------------------------------------------------------

    validation_results.sort(
        key=lambda x: x["auc"],
        reverse=True,
    )

    best_name = validation_results[0]["name"]

    best_result = validation_results[0]

    best_model = models[best_name]

    threshold = best_result["threshold"]

    print()
    print("=" * 70)
    print("BEST MODEL")
    print("=" * 70)

    print(
        f"Model               : {best_name}"
    )

    print(
        f"Validation AUC      : "
        f"{best_result['auc']:.6f}"
    )

    print(
        f"Selected threshold  : "
        f"{threshold:.4f}"
    )

    # -------------------------------------------------------------
    # IMPORTANT:
    # DO NOT RETRAIN USING VALIDATION OR TEST DATA.
    #
    # The model has already been selected using train + validation.
    # Test remains completely untouched.
    # -------------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)

    print(
        "IMPORTANT: Test data was not used for model selection."
    )

    test_results = evaluate_test(
        best_model,
        X_test,
        y_test,
        threshold,
    )

    print()
    print(
        f"TEST AUC              : "
        f"{test_results['auc']:.6f}"
    )

    print(
        f"TEST ACCURACY         : "
        f"{test_results['accuracy']:.6f}"
    )

    print(
        f"TEST BALANCED ACCURACY: "
        f"{test_results['balanced_accuracy']:.6f}"
    )

    print(
        f"TEST PRECISION        : "
        f"{test_results['precision']:.6f}"
    )

    print(
        f"TEST RECALL           : "
        f"{test_results['recall']:.6f}"
    )

    print(
        f"TEST F1               : "
        f"{test_results['f1']:.6f}"
    )

    print()
    print("CONFUSION MATRIX")
    print("-" * 70)

    cm = np.asarray(
        test_results["confusion_matrix"]
    )

    print(
        "                 Predicted"
    )
    print(
        "                 Cover   Stego"
    )
    print(
        f"Actual Cover     {cm[0,0]:6d} {cm[0,1]:7d}"
    )
    print(
        f"Actual Stego     {cm[1,0]:6d} {cm[1,1]:7d}"
    )

    print()
    print("CLASSIFICATION REPORT")
    print("-" * 70)

    print(
        test_results["classification_report"]
    )

    # -------------------------------------------------------------
    # SAVE MODEL
    # -------------------------------------------------------------

    model_path = OUTPUT / "best_model.pkl"

    with open(
        model_path,
        "wb",
    ) as file:

        pickle.dump(
            best_model,
            file,
        )

    # -------------------------------------------------------------
    # SAVE METADATA
    # -------------------------------------------------------------

    metadata = {
        "model_type": best_name,
        "feature_count": 99,
        "dataset": str(DATASET),
        "random_seed": 42,

        "train_samples": int(len(y_train)),
        "validation_samples": int(len(y_val)),
        "test_samples": int(len(y_test)),

        "train_cover": int(np.sum(y_train == 0)),
        "train_stego": int(np.sum(y_train == 1)),

        "validation_cover": int(np.sum(y_val == 0)),
        "validation_stego": int(np.sum(y_val == 1)),

        "test_cover": int(np.sum(y_test == 0)),
        "test_stego": int(np.sum(y_test == 1)),

        "validation_results": validation_results,

        "selected_threshold": float(threshold),

        "test_results": {
            "auc": test_results["auc"],
            "accuracy": test_results["accuracy"],
            "balanced_accuracy": test_results["balanced_accuracy"],
            "precision": test_results["precision"],
            "recall": test_results["recall"],
            "f1": test_results["f1"],
            "confusion_matrix": test_results["confusion_matrix"],
        },
    }

    metadata_path = OUTPUT / "metadata.json"

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    # -------------------------------------------------------------
    # FINAL STATUS
    # -------------------------------------------------------------

    print()
    print("=" * 70)
    print("MODEL SAVED")
    print("=" * 70)

    print(
        f"Model    : {model_path}"
    )

    print(
        f"Metadata : {metadata_path}"
    )

    print()
    print("=" * 70)
    print("FINAL STATUS")
    print("=" * 70)

    auc = test_results["auc"]
    bal_acc = test_results["balanced_accuracy"]

    if auc >= 0.90 and bal_acc >= 0.85:

        print(
            "STRONG RESULT"
        )

    elif auc >= 0.80 and bal_acc >= 0.75:

        print(
            "GOOD RESULT"
        )

    elif auc >= 0.70:

        print(
            "MODERATE RESULT"
        )

    else:

        print(
            "INSUFFICIENT RESULT"
        )

    print(
        f"Independent test AUC: {auc:.6f}"
    )

    print(
        f"Independent balanced accuracy: "
        f"{bal_acc:.6f}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
