from pathlib import Path

from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)


_REPO_ROOT = Path(
    __file__
).resolve().parents[2]


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=_REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ========================================================
    # APPLICATION
    # ========================================================

    app_name: str = (
        "Steganography Detection API"
    )

    debug: bool = False

    # ========================================================
    # PATHS
    # ========================================================

    repo_root: Path = _REPO_ROOT

    database_url: str = (
        f"sqlite:///"
        f"{_REPO_ROOT / 'data' / 'stegdetect.db'}"
    )

    uploads_dir: Path = (
        _REPO_ROOT / "data" / "uploads"
    )

    models_dir: Path = (
        _REPO_ROOT / "models"
    )

    reports_dir: Path = (
        _REPO_ROOT / "reports"
    )

    # ========================================================
    # PRODUCTION CLASSICAL MODEL
    # ========================================================

    classical_model_dir: Path = (
        _REPO_ROOT
        / "models"
        / "classical_stego"
    )

    @property
    def classical_model_path(self) -> Path:

        return (
            self.classical_model_dir
            / "best_model_294.pkl"
        )

    @property
    def classical_model_metadata_path(self) -> Path:

        return (
            self.classical_model_dir
            / "best_model_294_metadata.json"
        )

    # ========================================================
    # IMAGE MODEL DIRECTORY
    # ========================================================

    image_model_dir: Path = (
        _REPO_ROOT
        / "models"
        / "image_cnn"
    )

    @property
    def image_model_path(self) -> Path:

        return (
            self.image_model_dir
            / "model.keras"
        )

    @property
    def image_model_metadata_path(self) -> Path:

        return (
            self.image_model_dir
            / "metadata.json"
        )

    @property
    def image_preprocessing_config_path(self) -> Path:

        return (
            self.image_model_dir
            / "preprocessing_config.json"
        )

    # ========================================================
    # DETECTION THRESHOLDS
    # ========================================================

    decision_threshold: float = 0.505

    inconclusive_low: float = 0.45

    inconclusive_high: float = 0.55

    # Production model is the primary signal.
    classical_weight: float = 0.7

    # Optional StegExpose supplementary signal.
    stegexpose_weight: float = 0.3

    # ========================================================
    # UPLOAD LIMITS
    # ========================================================

    max_upload_size_mb: int = 50

    @property
    def max_upload_size_bytes(self) -> int:

        return (
            self.max_upload_size_mb
            * 1024
            * 1024
        )

    # ========================================================
    # STEGEXPOSE
    # ========================================================

    stegexpose_jar_path: Path | None = None

    # ========================================================
    # CORS
    # ========================================================

    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    # ========================================================
    # TEMPORARY / RECOVERED FILES
    # ========================================================

    @property
    def temp_dir(self) -> Path:

        path = self.repo_root / "data" / "temp"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def recovered_dir(self) -> Path:

        path = self.repo_root / "data" / "recovered"
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ========================================================
    # AUDIO SETTINGS
    # ========================================================

    audio_extensions: set[str] = {
        ".wav", ".mp3", ".flac", ".ogg",
    }

    # ========================================================
    # VIDEO SETTINGS
    # ========================================================

    video_extensions: set[str] = {
        ".mp4", ".avi", ".mov", ".mkv",
    }

    video_sample_frames: int = 10


settings = Settings()