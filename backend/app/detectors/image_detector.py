"""
End-to-end image steganography detector.

Production pipeline:

    Uploaded Image
          |
          v
    294 Feature Extraction
          |
          v
    Logistic Regression
          |
          v
    Classical Stego Probability
          |
          +----------------------+
          |                      |
          v                      v
    Optional StegExpose     Feature Summary
          |
          v
       Fusion
          |
          v
    Cover / Stego /
    Inconclusive
"""

from __future__ import annotations

from pathlib import Path

from app.config import settings
from app.core.exceptions import (
    CorruptMediaError,
    ModelUnavailableError,
)
from app.core.logging import get_logger

from app.detectors.base import (
    DetectionResult,
    Detector,
)

from app.detectors.fusion import (
    fuse_scores,
)

from app.detectors.stegexpose_wrapper import (
    StegExposeWrapper,
)

from ml.features.image_features import (
    compute_feature_summary,
    load_grayscale_image,
)

from ml.inference.image_predictor import (
    ImagePredictor,
)


logger = get_logger(__name__)


class ImageDetector(Detector):
    """
    Production image steganography detector.

    Primary model:
        294-feature Logistic Regression

    Supplementary model:
        StegExpose, when configured.
    """

    def __init__(
        self,
        predictor: ImagePredictor | None = None,
        stegexpose: StegExposeWrapper | None = None,
    ):

        self.predictor = (
            predictor
            or ImagePredictor(
                model_path=(
                    settings.classical_model_path
                ),
                metadata_path=(
                    settings.classical_model_metadata_path
                ),
                preprocessing_config_path=(
                    settings.image_preprocessing_config_path
                ),
            )
        )

        self.stegexpose = (
            stegexpose
            or StegExposeWrapper(
                settings.stegexpose_jar_path
            )
        )

    # =========================================================
    # MODEL AVAILABILITY
    # =========================================================

    @property
    def model_available(self) -> bool:
        """
        Check whether the production model exists.
        """

        return self.predictor.is_available()

    # =========================================================
    # ANALYZE
    # =========================================================

    def analyze(
        self,
        file_path: str | Path,
    ) -> DetectionResult:

        path = Path(file_path)

        warnings: list[str] = []

        # -----------------------------------------------------
        # VALIDATE IMAGE
        # -----------------------------------------------------

        try:

            image = load_grayscale_image(
                path
            )

        except Exception as exc:

            logger.exception(
                "Could not load image: %s",
                exc,
            )

            raise CorruptMediaError(
                str(exc)
            ) from exc

        # -----------------------------------------------------
        # FEATURE SUMMARY
        # -----------------------------------------------------

        features = compute_feature_summary(
            image
        )

        # -----------------------------------------------------
        # PRODUCTION MODEL CHECK
        # -----------------------------------------------------

        if not self.model_available:

            raise ModelUnavailableError(
                "Production 294-feature Logistic Regression "
                "model is not available."
            )

        # -----------------------------------------------------
        # PRODUCTION CLASSICAL PREDICTION
        # -----------------------------------------------------
        #
        # IMPORTANT:
        #
        # This is NOT a CNN prediction.
        #
        # The production model expects the original image
        # file because it extracts 294 steganalysis features.
        #
        # -----------------------------------------------------

        classical_score = (
            self.predictor.predict_file(
                path
            )
        )

        # -----------------------------------------------------
        # MODEL METADATA
        # -----------------------------------------------------

        model_version = (
            self.predictor.model_version
        )

        # -----------------------------------------------------
        # STEGEXPOSE
        # -----------------------------------------------------

        steg_result = (
            self.stegexpose.analyze_file(
                path
            )
        )

        supplementary_score = (
            steg_result.score
        )

        supplementary_available = (
            steg_result.available
            and supplementary_score is not None
        )

        # -----------------------------------------------------
        # STEGEXPOSE WARNINGS
        # -----------------------------------------------------

        if (
            steg_result.available
            and supplementary_score is None
            and steg_result.error
        ):

            warnings.append(
                "StegExpose supplementary analysis "
                f"unavailable: {steg_result.error}"
            )

        elif not steg_result.available:

            warnings.append(
                "StegExpose not configured; using "
                "production Logistic Regression model only."
            )

        # -----------------------------------------------------
        # FUSION
        # -----------------------------------------------------

        confidence, label = fuse_scores(
            classical_score=classical_score,

            supplementary_score=(
                supplementary_score
                if supplementary_available
                else None
            ),

            classical_weight=(
                settings.classical_weight
            ),

            supplementary_weight=(
                settings.stegexpose_weight
            ),

            threshold=(
                settings.decision_threshold
            ),

            inconclusive_low=(
                settings.inconclusive_low
            ),

            inconclusive_high=(
                settings.inconclusive_high
            ),
        )

        # -----------------------------------------------------
        # RESULT
        # -----------------------------------------------------

        return DetectionResult(
            label=label,

            confidence=confidence,

            classical_score=classical_score,

            supplementary_score=(
                supplementary_score
                if supplementary_available
                else None
            ),

            supplementary_available=(
                supplementary_available
            ),

            features=features,

            model_loaded=True,

            warnings=warnings,

            model_version=model_version,
        )