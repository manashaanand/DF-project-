"""
Abstract detector interface and detection result models.

Production model:
    294-feature Logistic Regression classifier.

Result fields:
    classical_score = production stego probability
    supplementary_score = optional StegExpose score
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ============================================================
# FILE INFO
# ============================================================

@dataclass
class FileInfo:
    """Forensic file identification."""

    filename: str
    extension: str
    mime_type: str
    media_type: str  # image / audio / video
    file_size: int
    sha256: str


# ============================================================
# FORENSIC FINDING
# ============================================================

@dataclass
class ForensicFinding:
    """Individual forensic observation."""

    category: str  # e.g. "lsb_analysis", "metadata", "appended_data"
    description: str
    severity: str = "info"  # info / low / medium / high
    evidence: dict = field(default_factory=dict)


# ============================================================
# EXTRACTION RESULT
# ============================================================

@dataclass
class ExtractionResult:
    """Result of a payload extraction attempt."""

    status: str = "NOT_ATTEMPTED"
    # NOT_ATTEMPTED / RECOVERED / PARTIALLY_RECOVERED /
    # KEY_REQUIRED / UNSUPPORTED / NOT_RECOVERABLE / INVALID_PAYLOAD

    payload_type: str | None = None
    payload_size: int | None = None
    sha256: str | None = None
    download_id: str | None = None
    message: str | None = None


# ============================================================
# TECHNIQUE IDENTIFICATION
# ============================================================

@dataclass
class TechniqueResult:
    """Suspected steganography technique."""

    technique: str  # e.g. LIKELY_LSB, POSSIBLE_DCT, etc.
    confidence: str = "low"  # low / medium / high
    evidence: list[str] = field(default_factory=list)


# ============================================================
# DETECTION RESULT
# ============================================================

@dataclass
class DetectionResult:
    """
    Result returned by a media detector.

    The production image model is a
    294-feature Logistic Regression classifier.

    Therefore:
        classical_score = production stego probability
    """

    label: str  # cover / stego / inconclusive

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

    # --- Phase 1 additions ---

    detection_status: str = "INCONCLUSIVE"
    # CLEAN / STEGO_SUSPECTED / STEGO_DETECTED / INCONCLUSIVE

    file_info: FileInfo | None = None

    forensic_findings: list[ForensicFinding] = field(
        default_factory=list
    )

    techniques: list[TechniqueResult] = field(
        default_factory=list
    )

    extraction: ExtractionResult | None = None

    feature_count: int = 294


# ============================================================
# ABSTRACT DETECTOR
# ============================================================

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