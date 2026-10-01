"""
Production image predictor.

Production model:
    294-feature Logistic Regression

Model:
    models/classical_stego/best_model_294.pkl

Metadata:
    models/classical_stego/best_model_294_metadata.json

The predictor extracts the same 294 features used during
production model training and passes them to the frozen
Logistic Regression classifier.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from ml.features.stego_features import extract_features_from_file


class ImagePredictor:
    """
    Production 294-feature Logistic Regression predictor.
    """

    def __init__(
        self,
        model_path: Path,
        metadata_path: Path | None = None,
        preprocessing_config_path: Path | None = None,
    ):
        self.model_path = Path(model_path)

        self.metadata_path = (
            Path(metadata_path)
            if metadata_path
            else self.model_path.parent / "metadata.json"
        )

        self.preprocessing_config_path = (
            Path(preprocessing_config_path)
            if preprocessing_config_path
            else self.model_path.parent / "preprocessing_config.json"
        )

        self._model = None
        self._metadata: dict | None = None

    @property
    def metadata(self) -> dict | None:
        if self._metadata is None and self.metadata_path.is_file():
            self._metadata = json.loads(
                self.metadata_path.read_text(encoding="utf-8")
            )

        return self._metadata

    @property
    def expected_feature_count(self) -> int:
        metadata = self.metadata or {}
        return int(metadata.get("feature_count", 294))

    @property
    def model_version(self) -> str:
        metadata = self.metadata or {}
        return str(
            metadata.get(
                "model_name",
                "LogisticRegression",
            )
        )

    def is_available(self) -> bool:
        return self.model_path.is_file()

    def load(self) -> None:
        if not self.is_available():
            raise FileNotFoundError(
                "Production image model not found at:\n"
                f"{self.model_path}"
            )

        with open(self.model_path, "rb") as file:
            self._model = pickle.load(file)

    def ensure_loaded(self) -> None:
        if self._model is None:
            self.load()

    def predict_file(self, file_path: str | Path) -> dict:
        """
        Extract the production 294 features from an image and
        return both the prediction and the extracted features.

        Returning the features is important because the forensic
        layer uses them to report the actual feature count.
        """

        self.ensure_loaded()

        file_path = Path(file_path)

        if not file_path.is_file():
            raise FileNotFoundError(
                f"Image file not found:\n{file_path}"
            )

        # Extract the exact features used by the production model.
        features = extract_features_from_file(file_path)

        features = np.asarray(
            features,
            dtype=np.float64,
        )

        # Never silently continue if the feature vector does not
        # match the production model.
        if len(features) != self.expected_feature_count:
            raise RuntimeError(
                "Production feature mismatch.\n"
                f"Expected: {self.expected_feature_count}\n"
                f"Extracted: {len(features)}"
            )

        # Logistic Regression probability for the STEGO class.
        probability = float(
            self._model.predict_proba([features])[0][1]
        )

        probability = float(
            np.clip(
                probability,
                0.0,
                1.0,
            )
        )

        # Preserve the existing detector behaviour:
        # < 0.45       -> cover
        # 0.45-0.55    -> inconclusive
        # > 0.55       -> stego
        if probability > 0.55:
            label = "stego"
        elif probability < 0.45:
            label = "cover"
        else:
            label = "inconclusive"

        confidence = max(
            probability,
            1.0 - probability,
        )

        return {
            "label": label,
            "confidence": float(confidence),
            "classical_score": probability,
            "stego_probability": probability,
            "features": features.tolist(),
            "feature_count": int(len(features)),
            "model_loaded": True,
            "model_version": self.model_version,
        }

    def predict_array(self, image: np.ndarray) -> float:
        """
        The production classical model expects the full
        294-feature representation.

        Array-based prediction is intentionally unsupported
        because converting an arbitrary ndarray into the exact
        production feature vector would risk preprocessing mismatch.
        """

        raise NotImplementedError(
            "The production classical model expects "
            "294 extracted image features. "
            "Use predict_file() for image inference."
        )