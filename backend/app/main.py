from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
import tempfile

from fastapi import (
    FastAPI,
    Request,
    UploadFile,
    File,
    HTTPException,
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings

from app.core.exceptions import (
    AppError,
    app_error_to_http,
)

from app.core.logging import (
    get_logger,
    setup_logging,
)

from app.db.database import init_db
from app.db.schemas import (
    HealthResponse,
    MultimediaAnalysisResponse,
    FileInfoResponse,
    DetectorInfo,
    PayloadResponse,
    TechniqueResponse,
    ForensicFindingResponse,
)

from app.utils.file_utils import (
    classify_media_type,
    compute_sha256,
    detect_mime_type,
    ALL_SUPPORTED_EXTENSIONS,
)

from ml.inference.image_predictor import (
    ImagePredictor,
)

from app.detectors.image_detector import (
    ImageDetector,
)


# ============================================================
# LOGGING
# ============================================================

setup_logging()

logger = get_logger(__name__)


# ============================================================
# GLOBAL PREDICTOR / DETECTOR
# ============================================================

_image_predictor: ImagePredictor | None = None

_image_detector: ImageDetector | None = None


# ============================================================
# PRODUCTION PREDICTOR
# ============================================================

def get_image_predictor() -> ImagePredictor:

    global _image_predictor

    if _image_predictor is None:

        _image_predictor = ImagePredictor(
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

    return _image_predictor


# ============================================================
# IMAGE DETECTOR
# ============================================================

def get_image_detector() -> ImageDetector:

    global _image_detector

    if _image_detector is None:

        _image_detector = ImageDetector(
            predictor=get_image_predictor()
        )

    return _image_detector


# ============================================================
# LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info(
        "Starting %s",
        settings.app_name,
    )

    # --------------------------------------------------------
    # DATABASE
    # --------------------------------------------------------

    init_db()

    logger.info(
        "Database initialized"
    )

    # --------------------------------------------------------
    # PRODUCTION MODEL
    # --------------------------------------------------------

    predictor = get_image_predictor()

    if predictor.is_available():

        logger.info(
            "Production 294-feature Logistic Regression "
            "model available"
        )

    else:

        logger.warning(
            "Production 294-feature Logistic Regression "
            "model is not available"
        )

    yield

    logger.info(
        "Shutting down"
    )


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    lifespan=lifespan,
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# ============================================================
# APP ERROR HANDLER
# ============================================================

@app.exception_handler(AppError)
async def handle_app_error(
    _request: Request,
    exc: AppError,
) -> JSONResponse:

    http_exc = app_error_to_http(
        exc
    )

    return JSONResponse(
        status_code=http_exc.status_code,

        content={
            "detail": http_exc.detail
        },
    )


# ============================================================
# HEALTH
# ============================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["health"],
)
def health_check() -> HealthResponse:

    predictor = get_image_predictor()

    return HealthResponse(
        status="ok",

        app_name=settings.app_name,

        database="connected",

        models_loaded=(
            predictor.is_available()
        ),
    )


# ============================================================
# IMAGE ANALYSIS
# ============================================================

@app.post(
    "/analyze/image",
    tags=["analysis"],
)
async def analyze_image(
    file: UploadFile = File(...),
):
    """
    Upload an image and perform
    steganography detection using
    the production 294-feature
    Logistic Regression model.
    """

    # --------------------------------------------------------
    # FILENAME
    # --------------------------------------------------------

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No filename provided",
        )

    # --------------------------------------------------------
    # EXTENSIONS
    # --------------------------------------------------------

    allowed_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
    }

    extension = Path(
        file.filename
    ).suffix.lower()

    if extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,

            detail=(
                "Unsupported image format. "
                "Supported formats: "
                "JPG, JPEG, PNG, BMP, TIFF, WEBP"
            ),
        )

    # --------------------------------------------------------
    # DETECTOR
    # --------------------------------------------------------

    detector = get_image_detector()

    if not detector.model_available:

        raise HTTPException(
            status_code=503,

            detail=(
                "Production 294-feature "
                "Logistic Regression model "
                "is not available"
            ),
        )

    temp_path: Path | None = None

    try:

        # ----------------------------------------------------
        # READ FILE
        # ----------------------------------------------------

        file_bytes = await file.read()

        if len(file_bytes) == 0:

            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty",
            )

        # ----------------------------------------------------
        # SIZE
        # ----------------------------------------------------

        if (
            len(file_bytes)
            > settings.max_upload_size_bytes
        ):

            raise HTTPException(
                status_code=413,

                detail=(
                    f"File exceeds maximum size of "
                    f"{settings.max_upload_size_mb} MB"
                ),
            )

        # ----------------------------------------------------
        # TEMP FILE
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            suffix=extension,
            delete=False,
        ) as temp_file:

            temp_file.write(
                file_bytes
            )

            temp_path = Path(
                temp_file.name
            )

        # ----------------------------------------------------
        # ANALYZE
        # ----------------------------------------------------

        result = detector.analyze(
            temp_path
        )

        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        return {

            "filename": file.filename,

            "media_type": "image",

            "label": result.label,

            "confidence": result.confidence,

            "classical_score": (
                result.classical_score
            ),

            "stegexpose_score": (
                result.supplementary_score
            ),

            "stegexpose_available": (
                result.supplementary_available
            ),

            "features": result.features,

            "model_loaded": result.model_loaded,

            "model_version": result.model_version,

            "warnings": result.warnings,
        }

    except HTTPException:

        raise

    except Exception as exc:

        logger.exception(
            "Image analysis failed: %s",
            exc,
        )

        raise HTTPException(
            status_code=500,

            detail=(
                f"Image analysis failed: {exc}"
            ),
        )

    finally:

        if (
            temp_path is not None
            and temp_path.exists()
        ):

            try:

                temp_path.unlink()

            except Exception as cleanup_exc:

                logger.warning(
                    "Could not delete temporary "
                    "file %s: %s",
                    temp_path,
                    cleanup_exc,
                )

# ============================================================
# UNIFIED MULTIMEDIA ANALYSIS
# ============================================================

@app.post(
    "/multimedia/analyze",
    response_model=MultimediaAnalysisResponse,
    tags=["analysis"],
)
async def analyze_multimedia(
    file: UploadFile = File(...),
):
    """
    Unified endpoint for image, audio, and video steganalysis.
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided",
        )

    ext = Path(file.filename).suffix.lower()
    if ext not in ALL_SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Supported extensions: {', '.join(ALL_SUPPORTED_EXTENSIONS)}",
        )

    media_type = classify_media_type(file.filename)
    if not media_type:
        raise HTTPException(
            status_code=400,
            detail="Unable to classify media type.",
        )

    temp_path: Path | None = None
    try:
        file_bytes = await file.read()
        if len(file_bytes) == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty",
            )

        if len(file_bytes) > settings.max_upload_size_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds maximum size of {settings.max_upload_size_mb} MB",
            )

        # Write to temp directory safely
        temp_path = settings.temp_dir / f"temp_{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(file_bytes)

        # File forensics
        sha256_hash = compute_sha256(temp_path)
        mime_type = detect_mime_type(temp_path)

        file_info = FileInfoResponse(
            filename=file.filename,
            extension=ext,
            mime_type=mime_type,
            media_type=media_type,
            file_size=len(file_bytes),
            sha256=sha256_hash,
        )

        if media_type == "image":
            detector = get_image_detector()
            if not detector.model_available:
                raise HTTPException(
                    status_code=503,
                    detail="Production 294-feature Logistic Regression model is not available",
                )
            
            result = detector.analyze(temp_path)

            detectors = [
                DetectorInfo(
                    name=result.model_version or "LogisticRegression",
                    score=result.classical_score,
                    weight=settings.classical_weight,
                )
            ]
            if result.supplementary_available:
                detectors.append(
                    DetectorInfo(
                        name="StegExpose",
                        score=result.supplementary_score,
                        weight=settings.stegexpose_weight,
                    )
                )

            # Map the response
            return MultimediaAnalysisResponse(
                file=file_info,
                media_type="image",
                status="STEGO_DETECTED" if result.label == "stego" else "CLEAN" if result.label == "cover" else "INCONCLUSIVE",
                steganography_detected=(result.label == "stego"),
                confidence=result.confidence,
                label=result.label,
                detectors=detectors,
                techniques=[
                    TechniqueResponse(
                        technique=t.technique,
                        confidence=t.confidence,
                        evidence=t.evidence
                    ) for t in result.techniques
                ], 
                payload=PayloadResponse(), 
                forensic_findings=[
                    ForensicFindingResponse(
                        category=f.category,
                        description=f.description,
                        severity=f.severity,
                        evidence=f.evidence
                    ) for f in result.forensic_findings
                ], 
                feature_count=result.feature_count,
                model_version=result.model_version,
                warnings=result.warnings,
                analysis_timestamp=datetime.utcnow().isoformat() + "Z"
            )
        
        else:
            # Phase 1 only supports images; returning 501 for others until Phase 4/5
            raise HTTPException(
                status_code=501,
                detail=f"Analysis for {media_type} is not yet implemented (coming in subsequent phases).",
            )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Analysis failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {exc}",
        )
    finally:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except Exception as cleanup_exc:
                logger.warning("Could not delete temporary file %s: %s", temp_path, cleanup_exc)