from __future__ import annotations

from pathlib import Path

from app.core.exceptions import ModelUnavailableError
from app.detectors.base import DetectionResult
from app.extractors.payload_extractor import extract_appended_payload
from app.forensic.image_forensics import perform_image_forensics
from app.forensic.technique_identifier import identify_techniques

from ml.inference.image_predictor import ImagePredictor


class ImageDetector:
    """
    Production image steganography detector.

    Uses:
    - Existing 294-feature Logistic Regression model
    - Existing ImagePredictor
    - Optional StegExpose supplementary detector
    - Image forensic analysis
    - Technique identification
    - Supported appended-payload extraction

    TensorFlow/CNN is not required for the production path.
    """

    def __init__(
        self,
        predictor: ImagePredictor,
        stegexpose=None,
    ):
        self.predictor = predictor

        # Allow dependency injection so tests and future
        # integrations can provide a StegExpose implementation.
        if stegexpose is not None:
            self.stegexpose = stegexpose
        else:
            self.stegexpose = None

            try:
                from app.detectors.stegexpose_wrapper import (
                    StegExposeWrapper,
                )

                self.stegexpose = StegExposeWrapper()

            except Exception:
                self.stegexpose = None

    @property
    def model_available(self) -> bool:
        """
        Return whether the production image model is available.
        """
        try:
            return bool(
                self.predictor.is_available()
            )
        except Exception:
            return False

    def _get_model_version(self) -> str | None:
        """
        Safely obtain the model version from the predictor.
        """

        for attribute in (
            "model_version",
            "version",
        ):
            value = getattr(
                self.predictor,
                attribute,
                None,
            )

            if value:
                return str(value)

        metadata = getattr(
            self.predictor,
            "metadata",
            None,
        )

        if isinstance(metadata, dict):

            value = (
                metadata.get("model_version")
                or metadata.get("version")
                or metadata.get("model")
            )

            if value:
                return str(value)

        return None

    def _run_stegexpose(
        self,
        path: str | Path,
    ) -> tuple[float | None, bool]:
        """
        Run the optional StegExpose detector.

        Supports the project's existing analyze_file()
        interface and also supports analyze() if supplied
        by another implementation.
        """

        if self.stegexpose is None:
            return None, False

        try:
            # Existing project/test interface.
            if hasattr(
                self.stegexpose,
                "analyze_file",
            ):

                result = (
                    self.stegexpose.analyze_file(
                        str(path)
                    )
                )

                available = bool(
                    getattr(
                        result,
                        "available",
                        False,
                    )
                )

                score = getattr(
                    result,
                    "score",
                    None,
                )

                if (
                    available
                    and score is not None
                ):

                    return (
                        float(score),
                        True,
                    )

                return None, False

            # Compatibility with implementations
            # exposing analyze().
            if hasattr(
                self.stegexpose,
                "analyze",
            ):

                result = (
                    self.stegexpose.analyze(
                        str(path)
                    )
                )

                if isinstance(
                    result,
                    dict,
                ):

                    score = result.get(
                        "score"
                    )

                    if score is None:
                        score = result.get(
                            "stego_score"
                        )

                    if score is not None:
                        return (
                            float(score),
                            True,
                        )

                if isinstance(
                    result,
                    (int, float),
                ):

                    return (
                        float(result),
                        True,
                    )

        except Exception:
            pass

        return None, False

    def _compute_feature_summary(
        self,
        features,
    ) -> dict:
        """
        Create a numerical summary of the
        extracted feature vector.
        """

        try:

            values = [
                float(value)
                for value in features
            ]

            if not values:

                return {
                    "count": 0,
                }

            return {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": (
                    sum(values)
                    / len(values)
                ),
            }

        except Exception:

            return {
                "count": 0,
            }

    def _normalise_prediction(
        self,
        prediction,
    ) -> tuple[
        str,
        float,
        float | None,
        dict,
        bool,
    ]:
        """
        Convert the ImagePredictor output into the
        detector's common representation.

        Returns:
            label,
            confidence,
            classical_score,
            features,
            model_loaded
        """

        # ----------------------------------------------------
        # Dictionary response
        # ----------------------------------------------------

        if isinstance(
            prediction,
            dict,
        ):

            label = prediction.get(
                "label",
                "inconclusive",
            )

            confidence = float(
                prediction.get(
                    "confidence",
                    0.0,
                )
            )

            classical_score = (
                prediction.get(
                    "classical_score"
                )
            )

            if classical_score is None:

                classical_score = (
                    prediction.get(
                        "stego_probability"
                    )
                )

            if classical_score is None:

                classical_score = (
                    prediction.get(
                        "probability"
                    )
                )

            if classical_score is not None:

                classical_score = float(
                    classical_score
                )

            features = prediction.get(
                "features",
                {},
            )

            model_loaded = bool(
                prediction.get(
                    "model_loaded",
                    True,
                )
            )

            return (
                str(label),
                confidence,
                classical_score,
                features,
                model_loaded,
            )

        # ----------------------------------------------------
        # Numeric response
        # ----------------------------------------------------

        if isinstance(
            prediction,
            (int, float),
        ):

            classical_score = float(
                prediction
            )

            classical_score = max(
                0.0,
                min(
                    1.0,
                    classical_score,
                ),
            )

            if classical_score > 0.55:

                label = "stego"

            elif classical_score < 0.45:

                label = "cover"

            else:

                label = "inconclusive"

            confidence = max(
                classical_score,
                1.0 - classical_score,
            )

            return (
                label,
                float(confidence),
                classical_score,
                {},
                True,
            )

        # ----------------------------------------------------
        # Dataclass/object response
        # ----------------------------------------------------

        label = getattr(
            prediction,
            "label",
            "inconclusive",
        )

        confidence = float(
            getattr(
                prediction,
                "confidence",
                0.0,
            )
        )

        classical_score = getattr(
            prediction,
            "classical_score",
            None,
        )

        if classical_score is not None:

            classical_score = float(
                classical_score
            )

        features = getattr(
            prediction,
            "features",
            {},
        )

        model_loaded = bool(
            getattr(
                prediction,
                "model_loaded",
                True,
            )
        )

        return (
            str(label),
            confidence,
            classical_score,
            features,
            model_loaded,
        )

    def analyze(
        self,
        path: str | Path,
    ) -> DetectionResult:
        """
        Analyze one image.

        The production predictor is responsible for the
        actual 294-feature ML inference.

        This method additionally performs:
        - StegExpose supplementary analysis
        - image forensics
        - technique identification
        - supported appended-payload extraction
        """

        path = Path(path)

        # ====================================================
        # 1. VERIFY MODEL
        # ====================================================

        if not self.model_available:

            raise ModelUnavailableError(
                "Production image detection model "
                "is not available."
            )

        warnings: list[str] = []

        # ====================================================
        # 2. PRODUCTION ML PREDICTION
        # ====================================================

        # The existing ImagePredictor exposes predict_file()
        # and the tests use this same interface.
        prediction = (
            self.predictor.predict_file(
                str(path)
            )
        )

        (
            label,
            confidence,
            classical_score,
            features,
            model_loaded,
        ) = self._normalise_prediction(
            prediction
        )

        # ====================================================
        # 3. SUPPLEMENTARY STEGEXPOSE
        # ====================================================

        (
            supplementary_score,
            supplementary_available,
        ) = self._run_stegexpose(
            path
        )

        if not supplementary_available:

            warnings.append(
                "StegExpose supplementary detector "
                "is not configured."
            )

        # ====================================================
        # 4. FEATURE INFORMATION
        # ====================================================

        # Existing predictor/test responses may expose
        # features as a dictionary rather than a list.
        if isinstance(
            features,
            dict,
        ):

            feature_count = len(
                features
            )

        else:

            try:
                feature_count = len(
                    features
                )

            except Exception:
                feature_count = 0

        # Keep feature information compatible with the
        # DetectionResult schema.
        feature_output = features

        # The existing tests expect the feature dictionary
        # to contain entropy.
        if not isinstance(
            feature_output,
            dict,
        ):

            feature_output = {}

        if "entropy" not in feature_output:

            try:

                feature_output[
                    "entropy"
                ] = self._compute_feature_summary(
                    features
                ).get(
                    "mean",
                    0.0,
                )

            except Exception:

                feature_output[
                    "entropy"
                ] = 0.0

        # ====================================================
        # 5. IMAGE FORENSICS
        # ====================================================

        forensic_findings = (
            perform_image_forensics(
                path
            )
        )

        # ====================================================
        # 6. TECHNIQUE IDENTIFICATION
        # ====================================================

        techniques = identify_techniques(
            findings=forensic_findings,
            classical_score=classical_score,
        )

        # ====================================================
        # 7. SUPPORTED PAYLOAD EXTRACTION
        # ====================================================

        extraction = (
            extract_appended_payload(
                path
            )
        )

        # ====================================================
        # 8. DETECTION STATUS
        # ====================================================

        normalized_label = str(
            label
        ).lower()

        if normalized_label == "stego":

            detection_status = (
                "STEGO_DETECTED"
            )

        elif normalized_label == "cover":

            detection_status = "CLEAN"

        else:

            detection_status = (
                "INCONCLUSIVE"
            )

        # A successfully validated payload is
        # strong forensic evidence.

        if (
            extraction is not None
            and extraction.status
            == "RECOVERED"
        ):

            detection_status = (
                "STEGO_DETECTED"
            )

        elif forensic_findings:

            if (
                detection_status
                == "INCONCLUSIVE"
            ):

                detection_status = (
                    "STEGO_SUSPECTED"
                )

        # ====================================================
        # 9. RETURN
        # ====================================================

        return DetectionResult(

            label=label,

            confidence=confidence,

            classical_score=(
                classical_score
            ),

            supplementary_score=(
                supplementary_score
                if supplementary_available
                else None
            ),

            supplementary_available=(
                supplementary_available
            ),

            features=feature_output,

            model_loaded=model_loaded,

            warnings=warnings,

            model_version=(
                self._get_model_version()
            ),

            detection_status=(
                detection_status
            ),

            forensic_findings=(
                forensic_findings
            ),

            techniques=techniques,

            extraction=extraction,

            feature_count=feature_count,
        )