"""
Random Forest Hyperparameter Optimization
AI Multimedia Steganography Detection

IMPORTANT:
- Uses 294-feature steganalysis extractor.
- Training split is used to fit models.
- Validation split is used to select the best configuration.
- Independent TEST split is NEVER used.
- Does NOT overwrite the current Logistic Regression model.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# =====================================================================
# PATHS
# =====================================================================

ROOT = Path(__file__).resolve().parents[2]

DATASET = (
    ROOT
    / "data"
    / "processed"
    / "images"
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
# IMPORT EXACT 294-FEATURE EXTRACTOR
# =====================================================================

BACKEND = ROOT / "backend"

if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from ml.features.stego_features import (
    extract_features_from_file,
)

# =====================================================================
# EXPECTED FEATURE COUNT
# =====================================================================

EXPECTED_FEATURES = 294

# =====================================================================
# COLLECT DATASET FILES
# =====================================================================


def collect_split(split: str):

    cover_dir = (
        DATASET
        / "cover"
        / split
    )

    stego_dir = (
        DATASET
        / "stego"
        / split
    )

    if not cover_dir.exists():
        raise FileNotFoundError(
            f"Cover directory not found:\n{cover_dir}"
        )

    if not stego_dir.exists():
        raise FileNotFoundError(
            f"Stego directory not found:\n{stego_dir}"
        )

    files = []

    # ---------------------------------------------------------------
    # COVER
    # ---------------------------------------------------------------

    for path in sorted(
        cover_dir.glob("*.png")
    ):

        files.append(
            (
                path,
                0,
            )
        )

    # ---------------------------------------------------------------
    # STEGO
    # ---------------------------------------------------------------

    for path in sorted(
        stego_dir.glob("*.png")
    ):

        files.append(
            (
                path,
                1,
            )
        )

    return files


# =====================================================================
# EXTRACT FEATURES
# =====================================================================


def extract_dataset(
    files,
    name: str,
):

    print()
    print("=" * 70)
    print(
        f"EXTRACTING FEATURES: {name}"
    )
    print("=" * 70)

    print(
        f"Samples: {len(files)}"
    )

    X = []
    y = []

    start = time.time()

    for i, (
        path,
        label,
    ) in enumerate(
        files,
        start=1,
    ):

        try:

            features = (
                extract_features_from_file(
                    path
                )
            )

        except Exception as exc:

            print()
            print(
                "ERROR processing:"
            )

            print(path)

            print(
                "Reason:",
                exc,
            )

            raise

        # -----------------------------------------------------------
        # CHECK EXACT FEATURE COUNT
        # -----------------------------------------------------------

        if features.shape != (
            EXPECTED_FEATURES,
        ):

            raise RuntimeError(
                f"Expected "
                f"{EXPECTED_FEATURES} features, "
                f"got {features.shape}\n"
                f"Image: {path}"
            )

        # -----------------------------------------------------------
        # CHECK FINITE
        # -----------------------------------------------------------

        if not np.isfinite(
            features
        ).all():

            raise RuntimeError(
                "Feature vector contains "
                f"NaN/Inf:\n{path}"
            )

        X.append(
            features
        )

        y.append(
            label
        )

        # -----------------------------------------------------------
        # PROGRESS
        # -----------------------------------------------------------

        if (
            i % 500 == 0
            or i == len(files)
        ):

            elapsed = (
                time.time()
                - start
            )

            rate = (
                i / elapsed
                if elapsed > 0
                else 0
            )

            remaining = (
                len(files)
                - i
            )

            eta = (
                remaining / rate
                if rate > 0
                else 0
            )

            print(
                f"Processed "
                f"{i}/{len(files)} "
                f"| {rate:.2f} img/s "
                f"| ETA {eta / 60:.1f} min"
            )

    X = np.asarray(
        X,
        dtype=np.float32,
    )

    y = np.asarray(
        y,
        dtype=np.int32,
    )

    return X, y


# =====================================================================
# THRESHOLD OPTIMIZATION
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
            probabilities
            >= threshold
        ).astype(
            np.int32
        )

        score = (
            balanced_accuracy_score(
                y_true,
                predictions,
            )
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
# VALIDATION EVALUATION
# =====================================================================


def evaluate_validation(
    model,
    X_val,
    y_val,
):

    probabilities = (
        model.predict_proba(
            X_val
        )[:, 1]
    )

    # ---------------------------------------------------------------
    # AUC
    # ---------------------------------------------------------------

    auc = roc_auc_score(
        y_val,
        probabilities,
    )

    # ---------------------------------------------------------------
    # FIND BEST VALIDATION THRESHOLD
    # ---------------------------------------------------------------

    (
        threshold,
        balanced_accuracy,
    ) = find_best_threshold(
        y_val,
        probabilities,
    )

    # ---------------------------------------------------------------
    # PREDICTIONS
    # ---------------------------------------------------------------

    predictions = (
        probabilities
        >= threshold
    ).astype(
        np.int32
    )

    # ---------------------------------------------------------------
    # METRICS
    # ---------------------------------------------------------------

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

    return {

        "auc": float(
            auc
        ),

        "threshold": float(
            threshold
        ),

        "balanced_accuracy": float(
            balanced_accuracy
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

        "f1": float(
            f1
        ),
    }


# =====================================================================
# MAIN
# =====================================================================


def main():

    print("=" * 70)
    print(
        "RANDOM FOREST HYPERPARAMETER OPTIMIZATION"
    )
    print("=" * 70)

    print()
    print(
        f"Expected feature count: "
        f"{EXPECTED_FEATURES}"
    )

    # =================================================================
    # COLLECT TRAIN / VALIDATION
    # =================================================================

    train_files = collect_split(
        "train"
    )

    val_files = collect_split(
        "val"
    )

    print()
    print(
        f"Training samples   : "
        f"{len(train_files)}"
    )

    print(
        f"Validation samples : "
        f"{len(val_files)}"
    )

    # =================================================================
    # EXTRACT TRAIN
    # =================================================================

    X_train, y_train = (
        extract_dataset(
            train_files,
            "TRAIN",
        )
    )

    # =================================================================
    # EXTRACT VALIDATION
    # =================================================================

    X_val, y_val = (
        extract_dataset(
            val_files,
            "VALIDATION",
        )
    )

    # =================================================================
    # FEATURE MATRIX CHECK
    # =================================================================

    print()
    print("=" * 70)
    print(
        "FEATURE MATRICES"
    )
    print("=" * 70)

    print(
        "Train shape:",
        X_train.shape,
    )

    print(
        "Val shape  :",
        X_val.shape,
    )

    if X_train.shape[1] != (
        EXPECTED_FEATURES
    ):

        raise RuntimeError(
            "Training feature mismatch: "
            f"{X_train.shape[1]}"
        )

    if X_val.shape[1] != (
        EXPECTED_FEATURES
    ):

        raise RuntimeError(
            "Validation feature mismatch: "
            f"{X_val.shape[1]}"
        )

    # =================================================================
    # CLASS DISTRIBUTION
    # =================================================================

    print()
    print("=" * 70)
    print(
        "CLASS DISTRIBUTION"
    )
    print("=" * 70)

    print(
        "Train Cover:",
        int(
            np.sum(
                y_train == 0
            )
        ),
    )

    print(
        "Train Stego:",
        int(
            np.sum(
                y_train == 1
            )
        ),
    )

    print(
        "Val Cover  :",
        int(
            np.sum(
                y_val == 0
            )
        ),
    )

    print(
        "Val Stego  :",
        int(
            np.sum(
                y_val == 1
            )
        ),
    )

    # =================================================================
    # CONTROLLED RANDOM FOREST SEARCH
    # =================================================================

    configurations = [

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
            "max_features": "sqrt",
            "min_samples_leaf": 8,
            "max_depth": None,
        },

        {
            "n_estimators": 500,
            "max_features": "log2",
            "min_samples_leaf": 1,
            "max_depth": None,
        },

        {
            "n_estimators": 500,
            "max_features": "log2",
            "min_samples_leaf": 2,
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

    results = []

    # =================================================================
    # VALIDATION EXPERIMENT
    # =================================================================

    print()
    print("=" * 70)
    print(
        "VALIDATION EXPERIMENT"
    )
    print("=" * 70)

    for index, config in enumerate(
        configurations,
        start=1,
    ):

        print()
        print(
            f"[{index}/{len(configurations)}]"
        )

        print(
            "Configuration:"
        )

        print(
            json.dumps(
                config,
                indent=2,
            )
        )

        # -------------------------------------------------------------
        # MODEL
        # -------------------------------------------------------------

        model = (
            RandomForestClassifier(

                n_estimators=
                    config[
                        "n_estimators"
                    ],

                max_features=
                    config[
                        "max_features"
                    ],

                min_samples_leaf=
                    config[
                        "min_samples_leaf"
                    ],

                max_depth=
                    config[
                        "max_depth"
                    ],

                class_weight="balanced",

                random_state=42,

                n_jobs=-1,
            )
        )

        # -------------------------------------------------------------
        # TRAIN
        # -------------------------------------------------------------

        start = time.time()

        model.fit(
            X_train,
            y_train,
        )

        training_time = (
            time.time()
            - start
        )

        # -------------------------------------------------------------
        # VALIDATION
        # -------------------------------------------------------------

        metrics = (
            evaluate_validation(
                model,
                X_val,
                y_val,
            )
        )

        result = {

            "configuration":
                config,

            "training_seconds":
                float(
                    training_time
                ),

            **metrics,
        }

        results.append(
            result
        )

        # -------------------------------------------------------------
        # DISPLAY
        # -------------------------------------------------------------

        print()
        print(
            f"AUC                : "
            f"{metrics['auc']:.6f}"
        )

        print(
            f"Balanced Accuracy  : "
            f"{metrics['balanced_accuracy']:.6f}"
        )

        print(
            f"Accuracy           : "
            f"{metrics['accuracy']:.6f}"
        )

        print(
            f"Precision          : "
            f"{metrics['precision']:.6f}"
        )

        print(
            f"Recall             : "
            f"{metrics['recall']:.6f}"
        )

        print(
            f"F1                 : "
            f"{metrics['f1']:.6f}"
        )

        print(
            f"Threshold          : "
            f"{metrics['threshold']:.4f}"
        )

        print(
            f"Training time      : "
            f"{training_time:.1f}s"
        )

    # =================================================================
    # SORT BY VALIDATION AUC
    # =================================================================

    results.sort(
        key=lambda x: x["auc"],
        reverse=True,
    )

    best = results[0]

    # =================================================================
    # BEST CONFIGURATION
    # =================================================================

    print()
    print("=" * 70)
    print(
        "BEST CONFIGURATION"
    )
    print("=" * 70)

    print(
        json.dumps(
            best,
            indent=2,
        )
    )

    # =================================================================
    # SAVE ALL RESULTS
    # =================================================================

    results_path = (
        OUTPUT
        / "rf_294_hyperparameter_results.json"
    )

    results_path.write_text(
        json.dumps(
            results,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "Results saved:"
    )

    print(
        results_path
    )

    # =================================================================
    # TRAIN BEST MODEL AGAIN
    # =================================================================

    best_config = (
        best["configuration"]
    )

    print()
    print("=" * 70)
    print(
        "TRAINING BEST VALIDATION MODEL"
    )
    print("=" * 70)

    print()
    print(
        "Best configuration:"
    )

    print(
        json.dumps(
            best_config,
            indent=2,
        )
    )

    best_model = (
        RandomForestClassifier(

            n_estimators=
                best_config[
                    "n_estimators"
                ],

            max_features=
                best_config[
                    "max_features"
                ],

            min_samples_leaf=
                best_config[
                    "min_samples_leaf"
                ],

            max_depth=
                best_config[
                    "max_depth"
                ],

            class_weight="balanced",

            random_state=42,

            n_jobs=-1,
        )
    )

    start = time.time()

    best_model.fit(
        X_train,
        y_train,
    )

    final_training_time = (
        time.time()
        - start
    )

    # =================================================================
    # SAVE MODEL
    # =================================================================

    best_model_path = (
        OUTPUT
        / "optimized_rf_294_validation_model.pkl"
    )

    with open(
        best_model_path,
        "wb",
    ) as f:

        pickle.dump(
            best_model,
            f,
        )

    # =================================================================
    # SAVE METADATA
    # =================================================================

    metadata = {

        "model_name":
            "RandomForestClassifier",

        "feature_count":
            EXPECTED_FEATURES,

        "selection_metric":
            "validation_auc",

        "validation_auc":
            best["auc"],

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

        "selected_threshold":
            best["threshold"],

        "random_state":
            42,

        "training_seconds":
            final_training_time,

        "configuration":
            best_config,

        "test_used":
            False,

        "model_path":
            str(
                best_model_path
            ),
    }

    metadata_path = (
        OUTPUT
        / "optimized_rf_294_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    # =================================================================
    # FINAL OUTPUT
    # =================================================================

    print()
    print(
        "Best model saved:"
    )

    print(
        best_model_path
    )

    print()
    print(
        "Metadata saved:"
    )

    print(
        metadata_path
    )

    # =================================================================
    # BASELINE COMPARISON
    # =================================================================

    baseline_auc = 0.873971

    print()
    print("=" * 70)
    print(
        "BASELINE COMPARISON"
    )
    print("=" * 70)

    print(
        "Current 294-feature "
        "Logistic Regression test AUC : "
        f"{baseline_auc:.6f}"
    )

    print(
        "Best Random Forest "
        "validation AUC               : "
        f"{best['auc']:.6f}"
    )

    print()
    print(
        "NOTE:"
    )

    print(
        "The comparison above is only "
        "a reference."
    )

    print(
        "The Logistic Regression number "
        "is from the untouched TEST set,"
    )

    print(
        "while the Random Forest number "
        "is from VALIDATION."
    )

    print(
        "Do NOT directly claim the RF is "
        "better until test evaluation."
    )

    # =================================================================
    # SAFETY MESSAGE
    # =================================================================

    print()
    print("=" * 70)
    print(
        "OPTIMIZATION COMPLETE"
    )
    print("=" * 70)

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "✓ 294 features used"
    )

    print(
        "✓ Training set used for fitting"
    )

    print(
        "✓ Validation set used for selection"
    )

    print(
        "✓ Test set NOT used"
    )

    print(
        "✓ Existing Logistic Regression "
        "model NOT overwritten"
    )

    print()
    print(
        "Next step:"
    )

    print(
        "Evaluate the selected RF model "
        "once on the untouched test set."
    )


# =====================================================================
# ENTRY POINT
# =====================================================================


if __name__ == "__main__":
    main()