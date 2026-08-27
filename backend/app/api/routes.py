"""
API routes for steganography detection.
"""

from pathlib import Path
import tempfile

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException,
)

from app.config import settings
from app.detectors.image_detector import (
    ImageDetector,
)


router = APIRouter()


# ============================================================
# GLOBAL DETECTOR
# ============================================================

_detector: ImageDetector | None = None


def get_detector() -> ImageDetector:

    global _detector

    if _detector is None:

        _detector = ImageDetector()

    return _detector


# ============================================================
# IMAGE ANALYSIS
# ============================================================

@router.post(
    "/analyze/image",
    tags=["analysis"],
)
async def analyze_image(
    file: UploadFile = File(...),
):

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No filename provided",
        )

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

    detector = get_detector()

    if not detector.model_available:

        raise HTTPException(

            status_code=503,

            detail=(
                "Production 294-feature "
                "Logistic Regression model "
                "is not available."
            ),
        )

    temp_path: Path | None = None

    try:

        file_bytes = await file.read()

        if not file_bytes:

            raise HTTPException(

                status_code=400,

                detail="Uploaded file is empty",
            )

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

        result = detector.analyze(
            temp_path
        )

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

            "model_loaded": (
                result.model_loaded
            ),

            "model_version": (
                result.model_version
            ),

            "warnings": result.warnings,
        }

    except HTTPException:

        raise

    except Exception as exc:

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

            except Exception:

                pass