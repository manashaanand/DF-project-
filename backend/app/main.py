from contextlib import asynccontextmanager
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
from app.db.schemas import HealthResponse

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