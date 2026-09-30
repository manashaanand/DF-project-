"""
File utilities for forensic analysis.

Provides:
    - SHA-256 hashing
    - MIME type detection
    - File size formatting
    - Media type classification (image/audio/video)
"""

from __future__ import annotations

import hashlib
import mimetypes
from pathlib import Path


# ============================================================
# MEDIA TYPE CLASSIFICATION
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp", ".gif",
}

AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".flac", ".ogg",
    ".aac", ".wma", ".m4a",
}

VIDEO_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".mkv",
    ".wmv", ".flv", ".webm",
}

ALL_SUPPORTED_EXTENSIONS = (
    IMAGE_EXTENSIONS | AUDIO_EXTENSIONS | VIDEO_EXTENSIONS
)


def classify_media_type(
    filename: str,
) -> str | None:
    """
    Classify a filename into image/audio/video
    based on its extension.

    Returns None if unsupported.
    """

    ext = Path(filename).suffix.lower()

    if ext in IMAGE_EXTENSIONS:
        return "image"

    if ext in AUDIO_EXTENSIONS:
        return "audio"

    if ext in VIDEO_EXTENSIONS:
        return "video"

    return None


# ============================================================
# SHA-256
# ============================================================

def compute_sha256(
    file_path: str | Path,
) -> str:
    """
    Compute the SHA-256 hash of a file.
    """

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:

        for chunk in iter(
            lambda: f.read(8192),
            b"",
        ):
            sha256.update(chunk)

    return sha256.hexdigest()


def compute_sha256_bytes(
    data: bytes,
) -> str:
    """
    Compute the SHA-256 hash of bytes.
    """

    return hashlib.sha256(data).hexdigest()


# ============================================================
# MIME TYPE
# ============================================================

def detect_mime_type(
    file_path: str | Path,
) -> str:
    """
    Detect MIME type from filename extension.

    Uses Python's built-in mimetypes module.
    Falls back to 'application/octet-stream'.
    """

    mime_type, _ = mimetypes.guess_type(
        str(file_path)
    )

    return mime_type or "application/octet-stream"


# ============================================================
# FILE SIZE
# ============================================================

def format_file_size(
    size_bytes: int,
) -> str:
    """
    Human-readable file size.
    """

    if size_bytes < 1024:
        return f"{size_bytes} B"

    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"

    if size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"

    return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"
