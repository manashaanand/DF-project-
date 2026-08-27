"""
CLI inference for AI Multimedia Steganography Detection.

Uses the validated classical Random Forest model trained on
99 statistical steganography features.

Model:
    models/classical_stego/best_model.pkl

Feature extractor:
    backend/ml/features/stego_features.py

Decision threshold:
    0.605
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np


# ================================================================
# PROJECT PATHS
# ================================================================

REPO_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = REPO_ROOT / "models" / "classical_stego" / "best_model.pkl"
METADATA_PATH = REPO_ROOT / "models" / "classical_stego" / "metadata.json"

BACKEND_PATH = REPO_ROOT / "backend"

if str(BACKEND_PATH) not in sys.path:
    sys.path.insert(0, str(BACKEND_PATH))


from ml.features.stego_features import extract_features_from_file


# ================================================================
# LOAD MODEL
# ================================================================

def load_model():
    """Load the trained Random Forest model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found:\n{MODEL_PATH}\n\n"
            "Run:\n"
            "python scripts\\training\\train_classical_model.py"
        )

    with MODEL_PATH.open("rb") as f:
        model = pickle.load(f)

    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "Loaded model does not support predict_proba()."
        )

    return model


# ================================================================
# LOAD METADATA
# ================================================================

def load_metadata() -> dict:
    """Load model metadata."""

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Model metadata not found:\n{METADATA_PATH}"
        )

    with METADATA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


# ================================================================
# CLASSIFICATION
# ================================================================

def classify_image(image_path: Path) -> dict:
    """
    Classify one image.

    Returns a dictionary containing:
        label
        stego_probability
        cover_probability
        confidence
        threshold
        feature_count
        model_type
        model_version
    """

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    # ------------------------------------------------------------
    # Load model and metadata
    # ------------------------------------------------------------

    model = load_model()
    metadata = load_metadata()

    threshold = float(
        metadata.get("selected_threshold", 0.605)
    )

    model_type = metadata.get(
        "model_type",
        "RandomForest"
    )

    # ------------------------------------------------------------
    # Extract EXACTLY the same 99 features used during training
    # ------------------------------------------------------------

    features = extract_features_from_file(image_path)

    if features.ndim != 1:
        raise ValueError(
            f"Expected 1-D feature vector, got shape {features.shape}"
        )

    expected_features = int(
        metadata.get("feature_count", 99)
    )

    if len(features) != expected_features:
        raise ValueError(
            f"Feature mismatch!\n"
            f"Model expects: {expected_features}\n"
            f"Extractor produced: {len(features)}"
        )

    if not np.isfinite(features).all():
        raise ValueError(
            "Feature vector contains NaN or infinite values."
        )

    # Random Forest expects shape:
    # (samples, features)
    X = features.reshape(1, -1)

    # ------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------

    probabilities = model.predict_proba(X)[0]

    # Class 1 = Stego
    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError(
            f"Model does not contain Stego class 1. "
            f"Classes found: {classes}"
        )

    stego_index = classes.index(1)

    stego_probability = float(
        probabilities[stego_index]
    )

    cover_probability = float(
        1.0 - stego_probability
    )

    # ------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------

    if stego_probability >= threshold:
        label = "STEGO"
    else:
        label = "COVER"

    # Confidence represents distance from decision threshold.
    #
    # It is NOT presented as "guaranteed probability of correctness".
    distance = abs(
        stego_probability - threshold
    )

    confidence = float(
        min(
            1.0,
            0.5 + distance
        )
    )

    # ------------------------------------------------------------
    # Result
    # ------------------------------------------------------------

    return {
        "label": label,
        "stego_probability": stego_probability,
        "cover_probability": cover_probability,
        "confidence": confidence,
        "decision_threshold": threshold,
        "feature_count": len(features),
        "model_type": model_type,
        "model_path": str(MODEL_PATH),
        "model_test_auc": metadata.get(
            "test_results", {}
        ).get("auc"),
        "model_test_accuracy": metadata.get(
            "test_results", {}
        ).get("accuracy"),
        "model_test_balanced_accuracy": metadata.get(
            "test_results", {}
        ).get("balanced_accuracy"),
        "image": str(image_path),
    }


# ================================================================
# MAIN
# ================================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "AI Multimedia Steganography Detector "
            "(99-feature Random Forest)"
        )
    )

    parser.add_argument(
        "image",
        type=Path,
        help="Path to image to classify"
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON"
    )

    args = parser.parse_args()

    try:
        result = classify_image(
            args.image
        )

    except Exception as exc:
        print(
            f"\nERROR: {exc}\n",
            file=sys.stderr
        )
        sys.exit(1)

    # ============================================================
    # JSON OUTPUT
    # ============================================================

    if args.json:
        print(
            json.dumps(
                result,
                indent=2
            )
        )
        return

    # ============================================================
    # NORMAL OUTPUT
    # ============================================================

    print("=" * 70)
    print("AI MULTIMEDIA STEGANOGRAPHY DETECTION")
    print("CLASSICAL RANDOM FOREST INFERENCE")
    print("=" * 70)

    print()
    print(f"Image              : {result['image']}")
    print(f"Model              : {result['model_type']}")
    print(f"Features           : {result['feature_count']}")
    print()

    print("-" * 70)
    print("PREDICTION")
    print("-" * 70)

    print(
        f"Label              : {result['label']}"
    )

    print(
        f"Stego probability  : "
        f"{result['stego_probability']:.6f}"
    )

    print(
        f"Cover probability  : "
        f"{result['cover_probability']:.6f}"
    )

    print(
        f"Decision threshold : "
        f"{result['decision_threshold']:.6f}"
    )

    print(
        f"Confidence indicator: "
        f"{result['confidence']:.6f}"
    )

    print()
    print("-" * 70)
    print("VALIDATED MODEL PERFORMANCE")
    print("-" * 70)

    if result["model_test_auc"] is not None:
        print(
            f"Independent test AUC          : "
            f"{result['model_test_auc']:.6f}"
        )

    if result["model_test_accuracy"] is not None:
        print(
            f"Independent test accuracy     : "
            f"{result['model_test_accuracy']:.6f}"
        )

    if result["model_test_balanced_accuracy"] is not None:
        print(
            f"Independent balanced accuracy : "
            f"{result['model_test_balanced_accuracy']:.6f}"
        )

    print()
    print("=" * 70)

    if result["label"] == "STEGO":
        print(
            "RESULT: STEGANOGRAPHY DETECTED"
        )
    else:
        print(
            "RESULT: NO STEGANOGRAPHY DETECTED"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()