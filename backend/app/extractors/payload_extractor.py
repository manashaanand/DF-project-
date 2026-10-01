"""
Real payload extraction for supported image container structures.

Currently supported:
- PNG data appended after the IEND chunk
- JPEG data appended after the FF D9 end-of-image marker

The extractor never treats arbitrary bytes as a successful payload.
Recovered bytes are passed through payload_detector.identify_payload().
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from app.detectors.base import ExtractionResult

from app.extractors.payload_detector import identify_payload


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _extract_png_appended_data(data: bytes) -> bytes | None:
    """
    Return bytes appended after the PNG IEND chunk.

    PNG chunk structure:
        4 bytes length
        4 bytes type
        N bytes data
        4 bytes CRC

    IEND is therefore 12 bytes long.
    """
    png_signature = b"\x89PNG\r\n\x1a\n"

    if not data.startswith(png_signature):
        return None

    position = 8

    while position + 12 <= len(data):
        chunk_length = int.from_bytes(
            data[position:position + 4],
            "big",
        )

        chunk_type = data[position + 4:position + 8]

        chunk_end = position + 12 + chunk_length

        if chunk_end > len(data):
            return None

        if chunk_type == b"IEND":
            appended = data[chunk_end:]

            if appended:
                return appended

            return None

        position = chunk_end

    return None


def _extract_jpeg_appended_data(data: bytes) -> bytes | None:
    """
    Return bytes appended after the JPEG FF D9 end marker.
    """
    if not data.startswith(b"\xff\xd8"):
        return None

    end_marker = data.find(b"\xff\xd9", 2)

    if end_marker == -1:
        return None

    payload_start = end_marker + 2

    if payload_start >= len(data):
        return None

    return data[payload_start:]


def extract_appended_payload(
    file_path: str | Path,
) -> ExtractionResult:
    """
    Attempt real appended-data extraction.

    Returns:
        ExtractionResult
    """

    path = Path(file_path)

    try:
        data = path.read_bytes()
    except OSError as exc:
        return ExtractionResult(
            status="NOT_RECOVERABLE",
            message=f"Could not read source file: {exc}",
        )

    if not data:
        return ExtractionResult(
            status="INVALID_PAYLOAD",
            message="Source file is empty.",
        )

    suffix = path.suffix.lower()

    payload: bytes | None = None

    if suffix == ".png":
        payload = _extract_png_appended_data(data)

    elif suffix in {".jpg", ".jpeg"}:
        payload = _extract_jpeg_appended_data(data)

    else:
        return ExtractionResult(
            status="UNSUPPORTED",
            message=(
                "Appended payload extraction is currently "
                "supported only for PNG and JPEG."
            ),
        )

    if not payload:
        return ExtractionResult(
            status="NOT_RECOVERABLE",
            message=(
                "No verifiable appended payload was found "
                "after the image container."
            ),
        )

    payload_status, payload_type = identify_payload(payload)

    if payload_status == "RECOVERED":
        return ExtractionResult(
            status="RECOVERED",
            payload_type=payload_type,
            payload_size=len(payload),
            sha256=_sha256(payload),
            message=(
                "Payload bytes were recovered from appended "
                "image data and validated using file signatures."
            ),
        )

    if payload_status == "KEY_REQUIRED":
        return ExtractionResult(
            status="KEY_REQUIRED",
            payload_type=payload_type,
            payload_size=len(payload),
            sha256=_sha256(payload),
            message=(
                "Additional bytes were recovered, but their format "
                "could not be identified. The payload may be encrypted "
                "or compressed without a recognizable header."
            ),
        )

    return ExtractionResult(
        status="INVALID_PAYLOAD",
        payload_size=len(payload),
        sha256=_sha256(payload),
        message=(
            "Appended bytes were found, but they did not pass "
            "payload validation."
        ),
    )