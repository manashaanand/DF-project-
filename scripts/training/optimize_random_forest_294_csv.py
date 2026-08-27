"""
294-FEATURE RANDOM FOREST OPTIMIZATION
Uses the already-extracted CSV features.

IMPORTANT:
- Does NOT extract features from images.
- Does NOT use the 6,000-image test set.
- Uses train/validation data only.
- Saves the best RF model and metadata.
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


# ================================================================
# PATHS
# ================================================================

ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = (
    ROOT
    / "models"
    / "pycaret_benchmark"
    / "pycaret_development_features_294.csv"
)

OUTPUT_DIR = (
    ROOT
    / "models"
    / "classical_stego"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_PATH = (
    OUTPUT_DIR
    / "optimized_rf_294_csv.pkl"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "optimized_rf_294_csv_metadata.json"
)

RESULTS_PATH = (
    OUTPUT_DIR
    / "rf_294_csv_optimization_results.json"
)


# ================================================================
# CONFIGURATION
# ================================================================

RANDOM_STATE = 42

# Existing dataset is 33,984 samples:
# 27,988 train + 5,996 validation.
#
# The CSV itself does not contain a split column, so we recreate
# the same 82.35/17.65 split with stratification and random_state 42.

VALIDATION_SIZE = 5996


CONFIGURATIONS = [
    {
        "n_estimators": 500,
        "max_features": "sqrt",
        "min_samples_leaf": 1,
        "max_depth": None,
    },
    {
        "n_estimators": 500,
        "max_features": "sqrt",
        "min_samples_leaf": 2,
        "max_depth": None,
    },
    {
        "n_estimators": 500,
        "max_features": "sqrt",
        "min_samples_leaf": 4,
        "max_depth": None,
    },
    {
        "n_estimators": 500,
        "max_features": 0.3,
        "min_samples_leaf": 1,
        "max_depth": None,
    },
    {
        "n_estimators": 500,
        "max_features": 0.5,
        "min_samples_leaf": 2,
        "max_depth": None,
    },
    {
        "n_estimators": 500,
        "max_features": "sqrt",
        "min_samples_leaf": 2,
        "max_depth": 20,
    },
    {
        "n_estimators": 500,
        "max_features": "sqrt",
        "min_samples_leaf": 2,
        "max_depth": 40,
    },
    {
        "n_estimators": 1000,
        "max_features": "sqrt",
        "min_samples_leaf": 2,
        "max_depth": None,
    },
]


# ================================================================
# THRESHOLD
# ================================================================

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
            best_threshold = float(threshold)

    return best_threshold, best_score


# ================================================================
# EVALUATION
# ================================================================

def evaluate(
    model,
    X_val,
    y_val,
):

    probabilities = (
        model.predict_proba(X_val)[:, 1]
    )

    auc = roc_auc_score(
        y_val,
        probabilities,
    )

    threshold, balanced = (
        find_best_threshold(
            y_val,
            probabilities,
        )
    )

    predictions = (
        probabilities >= threshold
    ).astype(np.int32)

    return {
        "auc": float(auc),

        "threshold": float(
            threshold
        ),

        "balanced_accuracy": float(
            balanced
        ),

        "accuracy": float(
            accuracy_score(
                y_val,
                predictions,
            )
        ),

        "precision": float(
            precision_score(
                y_val,
                predictions,
                zero_division=0,
            )
        ),

        "recall": float(
            recall_score(
                y_val,
                predictions,
                zero_division=0,
            )
        ),

        "f1": float(
            f1_score(
                y_val,
                predictions,
                zero_division=0,
            )
        ),
    }


# ================================================================
# MAIN
# ================================================================

def main():

    print("=" * 75)
    print(
        "294-FEATURE RANDOM FOREST OPTIMIZATION"
    )
    print("=" * 75)

    # ------------------------------------------------------------
    # LOAD CSV
    # ------------------------------------------------------------

    if not CSV_PATH.exists():

        raise FileNotFoundError(
            f"Feature CSV not found:\n{CSV_PATH}"
        )

    print()
    print("Loading already-extracted features...")
    print(CSV_PATH)

    df = pd.read_csv(
        CSV_PATH
    )

    print()
    print("=" * 75)
    print("DATASET")
    print("=" * 75)

    print(
        "Rows       :",
        len(df),
    )

    print(
        "Columns    :",
        len(df.columns),
    )

    if "target" not in df.columns:

        raise RuntimeError(
            "CSV does not contain 'target' column."
        )

    feature_columns = [
        column
        for column in df.columns
        if column != "target"
    ]

    if len(feature_columns) != 294:

        raise RuntimeError(
            f"Expected 294 features, "
            f"got {len(feature_columns)}"
        )

    X = df[
        feature_columns
    ].to_numpy(
        dtype=np.float32
    )

    y = df[
        "target"
    ].to_numpy(
        dtype=np.int32
    )

    if not np.isfinite(X).all():

        raise RuntimeError(
            "Feature matrix contains NaN/Inf."
        )

    print(
        "Feature matrix:",
        X.shape,
    )

    print()
    print(
        "Class distribution:"
    )

    print(
        "Cover:",
        int(np.sum(y == 0)),
    )

    print(
        "Stego:",
        int(np.sum(y == 1)),
    )

    # ------------------------------------------------------------
    # TRAIN / VALIDATION SPLIT
    # ------------------------------------------------------------

    print()
    print("=" * 75)
    print(
        "TRAIN / VALIDATION SPLIT"
    )
    print("=" * 75)

    X_train, X_val, y_train, y_val = (
        train_test_split(
            X,
            y,
            test_size=VALIDATION_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )

    print(
        "Train:",
        X_train.shape,
    )

    print(
        "Validation:",
        X_val.shape,
    )

    print(
        "Train Cover:",
        int(np.sum(y_train == 0)),
    )

    print(
        "Train Stego:",
        int(np.sum(y_train == 1)),
    )

    print(
        "Val Cover:",
        int(np.sum(y_val == 0)),
    )

    print(
        "Val Stego:",
        int(np.sum(y_val == 1)),
    )

    # ------------------------------------------------------------
    # OPTIMIZATION
    # ------------------------------------------------------------

    results = []

    print()
    print("=" * 75)
    print(
        "VALIDATION EXPERIMENT"
    )
    print("=" * 75)

    for index, config in enumerate(
        CONFIGURATIONS,
        start=1,
    ):

        print()
        print(
            f"[{index}/{len(CONFIGURATIONS)}]"
        )

        print(
            json.dumps(
                config,
                indent=2,
            )
        )

        model = (
            RandomForestClassifier(
                n_estimators=config[
                    "n_estimators"
                ],

                max_features=config[
                    "max_features"
                ],

                min_samples_leaf=config[
                    "min_samples_leaf"
                ],

                max_depth=config[
                    "max_depth"
                ],

                class_weight="balanced",

                random_state=RANDOM_STATE,

                n_jobs=-1,
            )
        )

        start = time.time()

        model.fit(
            X_train,
            y_train,
        )

        training_time = (
            time.time()
            - start
        )

        metrics = evaluate(
            model,
            X_val,
            y_val,
        )

        result = {
            "configuration": config,
            "training_seconds": float(
                training_time
            ),
            **metrics,
        }

        results.append(
            result
        )

        print(
            f"AUC               : "
            f"{metrics['auc']:.6f}"
        )

        print(
            f"Balanced Accuracy : "
            f"{metrics['balanced_accuracy']:.6f}"
        )

        print(
            f"Accuracy          : "
            f"{metrics['accuracy']:.6f}"
        )

        print(
            f"Precision         : "
            f"{metrics['precision']:.6f}"
        )

        print(
            f"Recall            : "
            f"{metrics['recall']:.6f}"
        )

        print(
            f"F1                : "
            f"{metrics['f1']:.6f}"
        )

        print(
            f"Threshold         : "
            f"{metrics['threshold']:.4f}"
        )

        print(
            f"Training time     : "
            f"{training_time:.1f}s"
        )

    # ------------------------------------------------------------
    # SELECT BEST
    # ------------------------------------------------------------

    results.sort(
        key=lambda item: (
            item["auc"],
            item["balanced_accuracy"],
        ),
        reverse=True,
    )

    best = results[0]

    print()
    print("=" * 75)
    print(
        "BEST RANDOM FOREST"
    )
    print("=" * 75)

    print(
        json.dumps(
            best,
            indent=2,
        )
    )

    # ------------------------------------------------------------
    # SAVE RESULTS
    # ------------------------------------------------------------

    RESULTS_PATH.write_text(
        json.dumps(
            results,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ------------------------------------------------------------
    # TRAIN BEST MODEL
    # ------------------------------------------------------------

    print()
    print("=" * 75)
    print(
        "TRAINING SELECTED RF"
    )
    print("=" * 75)

    best_config = (
        best["configuration"]
    )

    best_model = (
        RandomForestClassifier(
            n_estimators=best_config[
                "n_estimators"
            ],

            max_features=best_config[
                "max_features"
            ],

            min_samples_leaf=best_config[
                "min_samples_leaf"
            ],

            max_depth=best_config[
                "max_depth"
            ],

            class_weight="balanced",

            random_state=RANDOM_STATE,

            n_jobs=-1,
        )
    )

    best_model.fit(
        X_train,
        y_train,
    )

    with open(
        MODEL_PATH,
        "wb",
    ) as file:

        pickle.dump(
            best_model,
            file,
        )

    # ------------------------------------------------------------
    # SAVE METADATA
    # ------------------------------------------------------------

    metadata = {
        "model_name": "RandomForest",
        "feature_count": 294,
        "feature_columns": feature_columns,
        "random_state": RANDOM_STATE,
        "train_samples": int(len(y_train)),
        "validation_samples": int(len(y_val)),
        "selected_threshold": best[
            "threshold"
        ],
        "validation_auc": best[
            "auc"
        ],
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
        "configuration": best_config,
        "feature_source":
            str(CSV_PATH),
        "test_set_used": False,
    }

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ------------------------------------------------------------
    # FINAL OUTPUT
    # ------------------------------------------------------------

    print()
    print("=" * 75)
    print(
        "OPTIMIZATION COMPLETE"
    )
    print("=" * 75)

    print(
        "Best validation AUC :",
        f"{best['auc']:.6f}",
    )

    print(
        "Best balanced acc   :",
        f"{best['balanced_accuracy']:.6f}",
    )

    print(
        "Best F1             :",
        f"{best['f1']:.6f}",
    )

    print(
        "Threshold           :",
        f"{best['threshold']:.4f}",
    )

    print()
    print(
        "Model saved:"
    )

    print(
        MODEL_PATH
    )

    print()
    print(
        "Metadata saved:"
    )

    print(
        METADATA_PATH
    )

    print()
    print(
        "Results saved:"
    )

    print(
        RESULTS_PATH
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "✓ 294 pre-extracted features used"
    )

    print(
        "✓ No image feature extraction"
    )

    print(
        "✓ Test set NOT used"
    )

    print(
        "✓ Logistic Regression NOT overwritten"
    )


if __name__ == "__main__":
    main()