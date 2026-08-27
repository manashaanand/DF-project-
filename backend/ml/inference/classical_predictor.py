"""
Production classical steganography predictor.

Uses:
    294 handcrafted image features
        ->
    Logistic Regression
        ->
    Stego probability
        ->
    Production threshold
        ->
    COVER / STEGO
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from ml.features.stego_features import extract_features_from_file


class ClassicalPredictor:
    """
    Production predictor based on the validated
    294-feature Logistic Regression model.
    """

    def __init__(
        self,
        model_path: Path,
        metadata_path: Path,
    ):

        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)

        self._model = None
        self._metadata: dict | None = None

    # ============================================================
    # AVAILABILITY
    # ============================================================

    def is_available(self) -> bool:

        return (
            self.model_path.is_file()
            and self.metadata_path.is_file()
        )

    # ============================================================
    # METADATA
    # ============================================================

    @property
    def metadata(self) -> dict:

        if self._metadata is None:

            if not self.metadata_path.is_file():

                raise FileNotFoundError(
                    f"Metadata not found:\n"
                    f"{self.metadata_path}"
                )

            with open(
                self.metadata_path,
                "r",
                encoding="utf-8",
            ) as file:

                self._metadata = json.load(file)

        return self._metadata

    # ============================================================
    # MODEL LOADING
    # ============================================================

    def load(self) -> None:

        if not self.model_path.is_file():

            raise FileNotFoundError(
                f"Classical model not found:\n"
                f"{self.model_path}"
            )

        with open(
            self.model_path,
            "rb",
        ) as file:

            self._model = pickle.load(file)

    def ensure_loaded(self) -> None:

        if self._model is None:

            self.load()

    # ============================================================
    # FEATURE COUNT
    # ============================================================

    @property
    def feature_count(self) -> int:

        return int(
            self.metadata["feature_count"]
        )

    # ============================================================
    # THRESHOLD
    # ============================================================

    @property
    def threshold(self) -> float:

        return float(
            self.metadata["threshold"]
        )

    # ============================================================
    # PREDICTION
    # ============================================================

    def predict_file(
        self,
        file_path: str | Path,
    ) -> float:

        self.ensure_loaded()

        file_path = Path(file_path)

        features = extract_features_from_file(
            file_path
        )

        features = np.asarray(
            features,
            dtype=np.float64,
        )

        if len(features) != self.feature_count:

            raise RuntimeError(
                "Feature count mismatch.\n"
                f"Expected: {self.feature_count}\n"
                f"Got: {len(features)}"
            )

        probability = float(
            self._model.predict_proba(
                [features]
            )[0][1]
        )

        return float(
            np.clip(
                probability,
                0.0,
                1.0,
            )
        )

    # ============================================================
    # COMPLETE RESULT
    # ============================================================

    def predict(
        self,
        file_path: str | Path,
    ) -> dict:

        probability = self.predict_file(
            file_path
        )

        threshold = self.threshold

        if probability >= threshold:

            label = "stego"

            confidence = probability

        else:

            label = "cover"

            confidence = 1.0 - probability

        return {
            "label": label,
            "probability": probability,
            "confidence": confidence,
            "threshold": threshold,
            "feature_count": self.feature_count,
            "model_name": self.metadata.get(
                "model_name",
                "LogisticRegression",
            ),
            "model_version": (
                f"294-feature-"
                f"{self.metadata.get('model_name', 'LogisticRegression')}"
            ),
        }