from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


MediaType = Literal["image", "video", "audio"]
AnalysisLabel = Literal["cover", "stego", "inconclusive"]
AnalysisStatus = Literal["completed", "failed", "processing"]


class HealthResponse(BaseModel):
    status: str = "ok"
    app_name: str
    database: str
    models_loaded: bool = False


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
    cnn_score: float | None = None
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
