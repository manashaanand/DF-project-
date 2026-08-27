"""
PRODUCTION STEGANOGRAPHY PREDICTOR

Pipeline:
    Image
      ↓
    294-feature extractor
      ↓
    Logistic Regression
      ↓
    Probability
      ↓
    Threshold
      ↓
    Cover / Stego

Frozen production model:
    models/classical_stego/best_model_294.pkl
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path


# ================================================================
# PROJECT PATH
# ================================================================

ROOT = Path(__file__).resolve().parents[1]

BACKEND = ROOT / "backend"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


# ================================================================
# IMPORT FEATURE EXTRACTOR
# ================================================================

from ml.features.stego_features import (
    extract_features_from_file,
)


# ================================================================
# MODEL PATHS
# ================================================================

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


# ================================================================
# PREDICTION
# ================================================================

def predict_image(
    image_path: str | Path,
):

    image_path = Path(
        image_path
    )

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    if not METADATA_PATH.exists():

        raise FileNotFoundError(
            f"Metadata not found:\n{METADATA_PATH}"
        )

    # ------------------------------------------------------------
    # LOAD MODEL
    # ------------------------------------------------------------

    with open(
        MODEL_PATH,
        "rb",
    ) as file:

        model = pickle.load(
            file
        )

    # ------------------------------------------------------------
    # LOAD METADATA
    # ------------------------------------------------------------

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

    expected_features = int(
        metadata["feature_count"]
    )

    # ------------------------------------------------------------
    # EXTRACT FEATURES
    # ------------------------------------------------------------

    features = (
        extract_features_from_file(
            image_path
        )
    )

    if len(features) != expected_features:

        raise RuntimeError(
            f"Feature mismatch.\n"
            f"Expected: {expected_features}\n"
            f"Got: {len(features)}"
        )

    # ------------------------------------------------------------
    # PREDICT
    # ------------------------------------------------------------

    probability = float(
        model.predict_proba(
            [features]
        )[0][1]
    )

    prediction = (
        "STEGO"
        if probability >= threshold
        else "COVER"
    )

    confidence = (
        probability
        if prediction == "STEGO"
        else 1.0 - probability
    )

    return {
        "image": str(image_path),
        "prediction": prediction,
        "stego_probability": probability,
        "confidence": confidence,
        "threshold": threshold,
        "model": metadata["model_name"],
        "features": expected_features,
    }


# ================================================================
# COMMAND LINE
# ================================================================

def main():

    if len(sys.argv) != 2:

        print()
        print(
            "Usage:"
        )
        print(
            "python scripts\\predict.py <image_path>"
        )
        print()

        raise SystemExit(1)

    image_path = sys.argv[1]

    result = predict_image(
        image_path
    )

    print()
    print("=" * 70)
    print(
        "STEGANOGRAPHY DETECTION RESULT"
    )
    print("=" * 70)

    print(
        f"Image       : "
        f"{result['image']}"
    )

    print(
        f"Prediction  : "
        f"{result['prediction']}"
    )

    print(
        f"Stego prob. : "
        f"{result['stego_probability']:.4f}"
    )

    print(
        f"Confidence  : "
        f"{result['confidence']:.4f}"
    )

    print(
        f"Threshold   : "
        f"{result['threshold']:.4f}"
    )

    print(
        f"Features    : "
        f"{result['features']}"
    )

    print(
        f"Model       : "
        f"{result['model']}"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()