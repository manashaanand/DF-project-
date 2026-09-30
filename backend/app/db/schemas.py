from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


MediaType = Literal["image", "video", "audio"]
AnalysisLabel = Literal["cover", "stego", "inconclusive"]
AnalysisStatus = Literal["completed", "failed", "processing"]

DetectionStatus = Literal[
    "CLEAN",
    "STEGO_SUSPECTED",
    "STEGO_DETECTED",
    "INCONCLUSIVE",
]

ExtractionStatus = Literal[
    "NOT_ATTEMPTED",
    "RECOVERED",
    "PARTIALLY_RECOVERED",
    "KEY_REQUIRED",
    "UNSUPPORTED",
    "NOT_RECOVERABLE",
    "INVALID_PAYLOAD",
]


# ============================================================
# HEALTH
# ============================================================

class HealthResponse(BaseModel):
    status: str = "ok"
    app_name: str
    database: str
    models_loaded: bool = False


# ============================================================
# MULTIMEDIA ANALYSIS RESPONSE
# ============================================================

class FileInfoResponse(BaseModel):
    filename: str
    extension: str
    mime_type: str
    media_type: str
    file_size: int
    sha256: str


class ForensicFindingResponse(BaseModel):
    category: str
    description: str
    severity: str = "info"
    evidence: dict = Field(default_factory=dict)


class TechniqueResponse(BaseModel):
    technique: str
    confidence: str = "low"
    evidence: list[str] = Field(default_factory=list)


class PayloadResponse(BaseModel):
    detected: bool = False
    type: str | None = None
    size_bytes: int | None = None
    extraction_status: str = "NOT_ATTEMPTED"
    sha256: str | None = None
    download_id: str | None = None
    message: str | None = None


class DetectorInfo(BaseModel):
    name: str
    score: float | None = None
    weight: float | None = None


class MultimediaAnalysisResponse(BaseModel):
    """Unified response for image/audio/video analysis."""

    file: FileInfoResponse
    media_type: str
    status: str  # DetectionStatus
    steganography_detected: bool
    confidence: float
    label: str

    detectors: list[DetectorInfo] = Field(default_factory=list)
    techniques: list[TechniqueResponse] = Field(default_factory=list)

    payload: PayloadResponse = Field(
        default_factory=PayloadResponse
    )

    forensic_findings: list[ForensicFindingResponse] = Field(
        default_factory=list
    )

    feature_count: int | None = None
    model_version: str | None = None
    warnings: list[str] = Field(default_factory=list)

    analysis_timestamp: str | None = None


# ============================================================
# LEGACY SCHEMAS (preserved for backward compatibility)
# ============================================================

class AnalysisSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    media_type: str
    label: str
    confidence: float
    status: str
    created_at: datetime


class AnalysisDetailResponse(AnalysisSummary):
    file_size: int | None = None
    error_message: str | None = None
    classical_score: float | None = None
    stegexpose_score: float | None = None
    features: dict | None = None
    frame_summary: dict | None = None
    model_version: str | None = None


class PaginatedAnalyses(BaseModel):
    items: list[AnalysisSummary]
    total: int
    page: int
    limit: int


class ErrorResponse(BaseModel):
    detail: str


class AnalyzeResponse(BaseModel):
    analysis_id: str
    media_type: MediaType
    label: AnalysisLabel
    confidence: float
    scores: dict[str, float] = Field(default_factory=dict)
    features: dict = Field(default_factory=dict)
    created_at: datetime
