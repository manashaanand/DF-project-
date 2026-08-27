"""
Abstract detector interface and detection result models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DetectionResult:
    """
    Result returned by a media detector.

    The production image model is a
    294-feature Logistic Regression classifier.

    Therefore:
        classical_score = production stego probability

    `cnn_score` is intentionally NOT used because
    the production inference pipeline is not CNN-based.
    """

    label: str

    confidence: float

    classical_score: float | None

    supplementary_score: float | None

    supplementary_available: bool

    features: dict

    model_loaded: bool

    warnings: list[str] = field(
        default_factory=list
    )

    model_version: str | None = None


class Detector(ABC):

    @abstractmethod
    def analyze(
        self,
        file_path: str | Path,
    ) -> DetectionResult:
        """
        Analyze a media file and return
        a structured detection result.
        """
        raise NotImplementedError