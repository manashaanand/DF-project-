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

from ml.features.stego_features import (
    extract_features_from_file,
)


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
            else (
                self.model_path.parent
                / "metadata.json"
            )
        )

        # Kept for compatibility with the existing
        # application configuration.
        self.preprocessing_config_path = (
            Path(preprocessing_config_path)
            if preprocessing_config_path
            else (
                self.model_path.parent
                / "preprocessing_config.json"
            )
        )

        self._model = None

        self._metadata: dict | None = None

    # =========================================================
    # MODEL AVAILABILITY
    # =========================================================

    def is_available(self) -> bool:
        """
        Return True when the production model exists.
        """

        return self.model_path.is_file()

    # =========================================================
    # METADATA
    # =========================================================

    @property
    def metadata(self) -> dict | None:
        """
        Lazily load model metadata.
        """

        if (
            self._metadata is None
            and self.metadata_path.is_file()
        ):

            self._metadata = json.loads(
                self.metadata_path.read_text(
                    encoding="utf-8"
                )
            )

        return self._metadata

    # =========================================================
    # FEATURE COUNT
    # =========================================================

    @property
    def expected_feature_count(self) -> int:
        """
        Number of features expected by the production model.
        """

        metadata = self.metadata or {}

        return int(
            metadata.get(
                "feature_count",
                294,
            )
        )

    # =========================================================
    # MODEL VERSION
    # =========================================================

    @property
    def model_version(self) -> str:
        """
        Return production model name.
        """

        metadata = self.metadata or {}

        return str(
            metadata.get(
                "model_name",
                "LogisticRegression",
            )
        )

    # =========================================================
    # LOAD MODEL
    # =========================================================

    def load(self) -> None:
        """
        Load the frozen production Logistic Regression model.
        """

        if not self.is_available():

            raise FileNotFoundError(
                "Production image model not found at:\n"
                f"{self.model_path}"
            )

        with open(
            self.model_path,
            "rb",
        ) as file:

            self._model = pickle.load(file)

    # =========================================================
    # ENSURE MODEL LOADED
    # =========================================================

    def ensure_loaded(self) -> None:
        """
        Lazily load the model.
        """

        if self._model is None:
            self.load()

    # =========================================================
    # PREDICT FILE
    # =========================================================

    def predict_file(
        self,
        file_path: str | Path,
    ) -> float:
        """
        Extract the 294 production features from an image
        and return the stego probability.
        """

        self.ensure_loaded()

        file_path = Path(file_path)

        if not file_path.is_file():

            raise FileNotFoundError(
                f"Image file not found:\n{file_path}"
            )

        features = extract_features_from_file(
            file_path
        )

        features = np.asarray(
            features,
            dtype=np.float64,
        )

        if len(features) != self.expected_feature_count:

            raise RuntimeError(
                "Production feature mismatch.\n"
                f"Expected: {self.expected_feature_count}\n"
                f"Extracted: {len(features)}"
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

    # =========================================================
    # PREDICT ARRAY
    # =========================================================

    def predict_array(
        self,
        image: np.ndarray,
    ) -> float:
        """
        This production model operates on IMAGE FILES because
        its input is the 294-feature steganalysis feature vector.

        Direct ndarray prediction is intentionally unsupported.
        """

        raise NotImplementedError(
            "The production classical model expects "
            "294 extracted image features. "
            "Use predict_file() for image inference."
        )