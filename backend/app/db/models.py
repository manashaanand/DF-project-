import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    media_type: Mapped[str] = mapped_column(String(32), nullable=False)
    file_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    file_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    label: Mapped[str] = mapped_column(String(32), nullable=False, default="inconclusive")
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="processing")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    details: Mapped["AnalysisDetail | None"] = relationship(
        "AnalysisDetail",
        back_populates="analysis",
        uselist=False,
        cascade="all, delete-orphan",
    )


class AnalysisDetail(Base):
    __tablename__ = "analysis_details"

    analysis_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("analyses.id", ondelete="CASCADE"),
        primary_key=True,
    )
    classical_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    stegexpose_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    features_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    frame_summary_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)

    analysis: Mapped["Analysis"] = relationship("Analysis", back_populates="details")
