"""
ENSEMBLE STEGANOGRAPHY DETECTOR

Uses the already-generated 99-feature development CSV.

NO IMAGE FEATURE EXTRACTION.
NO TEST SET USED FOR TRAINING OR TUNING.

Models:
    - Random Forest
    - Extra Trees
    - HistGradientBoosting
    - Logistic Regression

The ensemble combines validation probabilities and selects the
threshold ONLY from the validation set.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier,
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
    confusion_matrix,
)


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = (
    ROOT
    / "models"
    / "pycaret_benchmark"
    / "pycaret_development_features.csv"
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


# ---------------------------------------------------------------------
# DATA SETTINGS
# ---------------------------------------------------------------------

TRAIN_SIZE = 27988

RANDOM_STATE = 42


# ---------------------------------------------------------------------
# THRESHOLD SEARCH
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------

def evaluate(
    name,
    probabilities,
    y_true,
):

    auc = roc_auc_score(
        y_true,
        probabilities,
    )

    threshold, balanced = (
        find_best_threshold(
            y_true,
            probabilities,
        )
    )

    predictions = (
        probabilities >= threshold
    ).astype(np.int32)

    accuracy = accuracy_score(
        y_true,
        predictions,
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

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(
        f"AUC               : {auc:.6f}"
    )

    print(
        f"Threshold         : {threshold:.4f}"
    )

    print(
        f"Balanced Accuracy : {balanced:.6f}"
    )

    print(
        f"Accuracy          : {accuracy:.6f}"
    )

    print(
        f"Precision         : {precision:.6f}"
    )

    print(
        f"Recall            : {recall:.6f}"
    )

    print(
        f"F1                : {f1:.6f}"
    )

    return {
        "name": name,
        "auc": float(auc),
        "threshold": float(threshold),
        "balanced_accuracy": float(
            balanced
        ),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "probabilities": probabilities,
    }


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    print("=" * 70)
    print("ENSEMBLE STEGANOGRAPHY BENCHMARK")
    print("=" * 70)

    # ---------------------------------------------------------------
    # LOAD CACHED FEATURES
    # ---------------------------------------------------------------

    if not CSV_PATH.exists():

        raise FileNotFoundError(
            f"Cached feature file not found:\n"
            f"{CSV_PATH}"
        )

    print()
    print("Loading cached features...")

    df = pd.read_csv(
        CSV_PATH
    )

    print(
        f"Dataset shape: {df.shape}"
    )

    X = df.drop(
        columns=["target"]
    ).to_numpy(
        dtype=np.float32
    )

    y = df["target"].to_numpy(
        dtype=np.int32
    )

    if X.shape[1] != 99:

        raise RuntimeError(
            f"Expected 99 features, "
            f"got {X.shape[1]}"
        )

    # ---------------------------------------------------------------
    # DEVELOPMENT SPLIT
    # ---------------------------------------------------------------

    X_train = X[:TRAIN_SIZE]
    y_train = y[:TRAIN_SIZE]

    X_val = X[TRAIN_SIZE:]
    y_val = y[TRAIN_SIZE:]

    print()
    print(
        f"Train      : {len(y_train)}"
    )

    print(
        f"Validation : {len(y_val)}"
    )

    # ---------------------------------------------------------------
    # MODELS
    # ---------------------------------------------------------------

    rf = RandomForestClassifier(
        n_estimators=700,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    et = ExtraTreesClassifier(
        n_estimators=700,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    hgb = HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.04,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )

    lr = Pipeline(
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
    )

    models = [
        ("RandomForest", rf),
        ("ExtraTrees", et),
        ("HistGradientBoosting", hgb),
        ("LogisticRegression", lr),
    ]

    validation_probabilities = {}

    trained_models = {}

    # ---------------------------------------------------------------
    # TRAIN INDIVIDUAL MODELS
    # ---------------------------------------------------------------

    for name, model in models:

        print()
        print("=" * 70)
        print(
            f"TRAINING {name}"
        )
        print("=" * 70)

        model.fit(
            X_train,
            y_train,
        )

        probabilities = (
            model.predict_proba(
                X_val
            )[:, 1]
        )

        validation_probabilities[name] = (
            probabilities
        )

        trained_models[name] = model

        evaluate(
            name,
            probabilities,
            y_val,
        )

    # ---------------------------------------------------------------
    # ENSEMBLE WEIGHTS
    # ---------------------------------------------------------------
    #
    # RandomForest and ExtraTrees have been strongest.
    # Give them higher weight while retaining the diversity of HGB
    # and Logistic Regression.
    # ---------------------------------------------------------------

    ensemble_probability = (
        0.40
        * validation_probabilities[
            "RandomForest"
        ]
        +
        0.35
        * validation_probabilities[
            "ExtraTrees"
        ]
        +
        0.15
        * validation_probabilities[
            "HistGradientBoosting"
        ]
        +
        0.10
        * validation_probabilities[
            "LogisticRegression"
        ]
    )

    ensemble_result = evaluate(
        "WEIGHTED ENSEMBLE",
        ensemble_probability,
        y_val,
    )

    # ---------------------------------------------------------------
    # COMPARE
    # ---------------------------------------------------------------

    individual_results = []

    for name in validation_probabilities:

        result = evaluate(
            name,
            validation_probabilities[name],
            y_val,
        )

        individual_results.append(
            result
        )

    candidates = (
        individual_results
        + [ensemble_result]
    )

    best = max(
        candidates,
        key=lambda r: (
            r["auc"],
            r["balanced_accuracy"],
            r["f1"],
        ),
    )

    # ---------------------------------------------------------------
    # PRINT FINAL
    # ---------------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL DEVELOPMENT COMPARISON")
    print("=" * 70)

    for result in sorted(
        candidates,
        key=lambda r: (
            r["auc"],
            r["balanced_accuracy"],
        ),
        reverse=True,
    ):

        print(
            f"{result['name']:25s} "
            f"AUC={result['auc']:.6f} "
            f"BAL={result['balanced_accuracy']:.6f} "
            f"F1={result['f1']:.6f}"
        )

    print()
    print("=" * 70)
    print("SELECTED MODEL")
    print("=" * 70)

    print(
        f"Model     : {best['name']}"
    )

    print(
        f"AUC       : {best['auc']:.6f}"
    )

    print(
        f"Balanced  : "
        f"{best['balanced_accuracy']:.6f}"
    )

    print(
        f"Threshold : "
        f"{best['threshold']:.4f}"
    )

    # ---------------------------------------------------------------
    # SAVE INDIVIDUAL MODELS
    # ---------------------------------------------------------------

    for name, model in trained_models.items():

        filename = (
            name.lower()
            .replace(
                "gradientboosting",
                "gradient_boosting",
            )
            + ".pkl"
        )

        with open(
            OUTPUT_DIR / filename,
            "wb",
        ) as f:

            pickle.dump(
                model,
                f,
            )

    # ---------------------------------------------------------------
    # SAVE BEST MODEL
    # ---------------------------------------------------------------

    if best["name"] == "WEIGHTED ENSEMBLE":

        ensemble_path = (
            OUTPUT_DIR
            / "ensemble_models.pkl"
        )

        with open(
            ensemble_path,
            "wb",
        ) as f:

            pickle.dump(
                trained_models,
                f,
            )

        model_type = "weighted_ensemble"

    else:

        best_model = trained_models[
            best["name"]
        ]

        ensemble_path = (
            OUTPUT_DIR
            / "best_model_improved.pkl"
        )

        with open(
            ensemble_path,
            "wb",
        ) as f:

            pickle.dump(
                best_model,
                f,
            )

        model_type = best["name"]

    # ---------------------------------------------------------------
    # SAVE METADATA
    # ---------------------------------------------------------------

    metadata = {
        "model_type": model_type,
        "feature_count": 99,
        "threshold": best["threshold"],
        "validation_auc": best["auc"],
        "validation_balanced_accuracy": (
            best["balanced_accuracy"]
        ),
        "validation_accuracy": (
            best["accuracy"]
        ),
        "validation_precision": (
            best["precision"]
        ),
        "validation_recall": (
            best["recall"]
        ),
        "validation_f1": best["f1"],
        "random_state": RANDOM_STATE,
    }

    metadata_path = (
        OUTPUT_DIR
        / "improved_model_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    print()
    print("=" * 70)
    print("SAVED")
    print("=" * 70)

    print(
        f"Metadata: {metadata_path}"
    )


if __name__ == "__main__":
    main()