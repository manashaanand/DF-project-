"""
294-FEATURE CLASSICAL STEGANOGRAPHY BENCHMARK

Uses:
    models/pycaret_benchmark/pycaret_development_features_294.csv

Models:
    RandomForest
    ExtraTrees
    HistGradientBoosting
    LogisticRegression

Selection:
    AUC first
    Balanced Accuracy second
    F1 third

The untouched test set is NOT used.
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# =====================================================================
# PATHS
# =====================================================================

ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = (
    ROOT
    / "models"
    / "pycaret_benchmark"
    / "pycaret_development_features_294.csv"
)

OUTPUT = (
    ROOT
    / "models"
    / "classical_stego"
)

OUTPUT.mkdir(
    parents=True,
    exist_ok=True,
)


# =====================================================================
# SETTINGS
# =====================================================================

TRAIN_SIZE = 27988

RANDOM_STATE = 42


# =====================================================================
# THRESHOLD SEARCH
# =====================================================================

def find_best_threshold(
    y_true,
    probabilities,
):

    thresholds = np.linspace(
        0.05,
        0.95,
        181,
    )

    best_threshold = 0.5
    best_score = -1.0

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(np.int32)

        score = balanced_accuracy_score(
            y_true,
            predictions,
        )

        if score > best_score:

            best_score = score
            best_threshold = float(
                threshold
            )

    return (
        best_threshold,
        best_score,
    )


# =====================================================================
# EVALUATE MODEL
# =====================================================================

def evaluate_model(
    name,
    model,
    X_train,
    y_train,
    X_val,
    y_val,
):

    print()
    print("=" * 70)
    print(
        f"TRAINING: {name}"
    )
    print("=" * 70)

    start = time.time()

    model.fit(
        X_train,
        y_train,
    )

    elapsed = (
        time.time()
        - start
    )

    probabilities = (
        model.predict_proba(
            X_val
        )[:, 1]
    )

    auc = roc_auc_score(
        y_val,
        probabilities,
    )

    threshold, balanced_acc = (
        find_best_threshold(
            y_val,
            probabilities,
        )
    )

    predictions = (
        probabilities >= threshold
    ).astype(np.int32)

    accuracy = accuracy_score(
        y_val,
        predictions,
    )

    precision = precision_score(
        y_val,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_val,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_val,
        predictions,
        zero_division=0,
    )

    print(
        f"Training time       : "
        f"{elapsed:.2f} sec"
    )

    print(
        f"Validation AUC      : "
        f"{auc:.6f}"
    )

    print(
        f"Best threshold      : "
        f"{threshold:.4f}"
    )

    print(
        f"Balanced accuracy   : "
        f"{balanced_acc:.6f}"
    )

    print(
        f"Accuracy            : "
        f"{accuracy:.6f}"
    )

    print(
        f"Precision           : "
        f"{precision:.6f}"
    )

    print(
        f"Recall              : "
        f"{recall:.6f}"
    )

    print(
        f"F1                  : "
        f"{f1:.6f}"
    )

    return {
        "name": name,
        "model": model,
        "auc": float(auc),
        "threshold": float(threshold),
        "balanced_accuracy": float(
            balanced_acc
        ),
        "accuracy": float(
            accuracy
        ),
        "precision": float(
            precision
        ),
        "recall": float(
            recall
        ),
        "f1": float(f1),
        "training_seconds": float(
            elapsed
        ),
    }


# =====================================================================
# MAIN
# =====================================================================

def main():

    print("=" * 70)
    print(
        "294-FEATURE CLASSICAL "
        "STEGANOGRAPHY BENCHMARK"
    )
    print("=" * 70)

    # ---------------------------------------------------------------
    # LOAD
    # ---------------------------------------------------------------

    if not CSV_PATH.exists():

        raise FileNotFoundError(
            f"294-feature CSV not found:\n"
            f"{CSV_PATH}"
        )

    print()
    print(
        "Loading 294-feature dataset..."
    )

    print(
        CSV_PATH
    )

    df = pd.read_csv(
        CSV_PATH
    )

    print(
        f"Loaded shape: {df.shape}"
    )

    # ---------------------------------------------------------------
    # TARGET
    # ---------------------------------------------------------------

    if "target" not in df.columns:

        raise RuntimeError(
            "CSV does not contain target."
        )

    y = df[
        "target"
    ].to_numpy(
        dtype=np.int32
    )

    X = df.drop(
        columns=["target"]
    ).to_numpy(
        dtype=np.float32
    )

    print(
        f"Feature count: {X.shape[1]}"
    )

    if X.shape[1] != 294:

        raise RuntimeError(
            f"Expected 294 features, "
            f"got {X.shape[1]}"
        )

    if not np.isfinite(X).all():

        raise RuntimeError(
            "Feature matrix contains "
            "NaN or infinite values."
        )

    # ---------------------------------------------------------------
    # SPLIT
    # ---------------------------------------------------------------

    X_train = X[
        :TRAIN_SIZE
    ]

    y_train = y[
        :TRAIN_SIZE
    ]

    X_val = X[
        TRAIN_SIZE:
    ]

    y_val = y[
        TRAIN_SIZE:
    ]

    print()
    print("=" * 70)
    print(
        "DATA SPLIT"
    )
    print("=" * 70)

    print(
        f"Train: {len(y_train)}"
    )

    print(
        f"Validation: {len(y_val)}"
    )

    print(
        f"Train cover/stego: "
        f"{np.sum(y_train == 0)} / "
        f"{np.sum(y_train == 1)}"
    )

    print(
        f"Val cover/stego: "
        f"{np.sum(y_val == 0)} / "
        f"{np.sum(y_val == 1)}"
    )

    # ---------------------------------------------------------------
    # MODELS
    # ---------------------------------------------------------------

    models = {

        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=600,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "RandomForest": RandomForestClassifier(
            n_estimators=600,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "HistGradientBoosting":
            HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.05,
                max_leaf_nodes=31,
                l2_regularization=1.0,
                random_state=RANDOM_STATE,
            ),

        "LogisticRegression":
            Pipeline(
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
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
    }

    # ---------------------------------------------------------------
    # BENCHMARK
    # ---------------------------------------------------------------

    results = []

    for name, model in models.items():

        result = evaluate_model(
            name,
            model,
            X_train,
            y_train,
            X_val,
            y_val,
        )

        results.append(
            result
        )

    # ---------------------------------------------------------------
    # RANK
    # ---------------------------------------------------------------

    results_sorted = sorted(
        results,
        key=lambda r: (
            r["auc"],
            r["balanced_accuracy"],
            r["f1"],
        ),
        reverse=True,
    )

    print()
    print("=" * 70)
    print(
        "FINAL 294-FEATURE MODEL RANKING"
    )
    print("=" * 70)

    for index, result in enumerate(
        results_sorted,
        start=1,
    ):

        print()
        print(
            f"{index}. "
            f"{result['name']}"
        )

        print(
            f"   AUC       : "
            f"{result['auc']:.6f}"
        )

        print(
            f"   Balanced  : "
            f"{result['balanced_accuracy']:.6f}"
        )

        print(
            f"   Accuracy  : "
            f"{result['accuracy']:.6f}"
        )

        print(
            f"   Precision : "
            f"{result['precision']:.6f}"
        )

        print(
            f"   Recall    : "
            f"{result['recall']:.6f}"
        )

        print(
            f"   F1        : "
            f"{result['f1']:.6f}"
        )

        print(
            f"   Threshold : "
            f"{result['threshold']:.4f}"
        )

    # ---------------------------------------------------------------
    # BEST
    # ---------------------------------------------------------------

    best = results_sorted[0]

    print()
    print("=" * 70)
    print(
        "BEST 294-FEATURE MODEL"
    )
    print("=" * 70)

    print(
        f"Model     : {best['name']}"
    )

    print(
        f"AUC       : "
        f"{best['auc']:.6f}"
    )

    print(
        f"Balanced  : "
        f"{best['balanced_accuracy']:.6f}"
    )

    print(
        f"Accuracy  : "
        f"{best['accuracy']:.6f}"
    )

    print(
        f"F1        : "
        f"{best['f1']:.6f}"
    )

    print(
        f"Threshold : "
        f"{best['threshold']:.4f}"
    )

    # ---------------------------------------------------------------
    # SAVE MODEL
    # ---------------------------------------------------------------

    model_path = (
        OUTPUT
        / "best_model_294.pkl"
    )

    metadata_path = (
        OUTPUT
        / "best_model_294_metadata.json"
    )

    with open(
        model_path,
        "wb",
    ) as file:

        pickle.dump(
            best["model"],
            file,
        )

    metadata = {
        "model_name": best["name"],
        "feature_count": 294,
        "threshold": best["threshold"],
        "validation_auc": best["auc"],
        "validation_balanced_accuracy":
            best["balanced_accuracy"],
        "validation_accuracy":
            best["accuracy"],
        "validation_precision":
            best["precision"],
        "validation_recall":
            best["recall"],
        "validation_f1":
            best["f1"],
        "random_state":
            RANDOM_STATE,
        "feature_dataset":
            str(CSV_PATH),
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )

    print()
    print("=" * 70)
    print(
        "294-FEATURE MODEL SAVED"
    )
    print("=" * 70)

    print(
        f"Model:"
    )

    print(
        model_path
    )

    print(
        f"Metadata:"
    )

    print(
        metadata_path
    )


if __name__ == "__main__":
    main()