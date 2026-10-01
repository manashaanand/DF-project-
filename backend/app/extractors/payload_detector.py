"""
Payload validation and identification based on magic bytes and heuristics.

This module validates recovered bytes. It does not perform extraction.
"""

from __future__ import annotations

import math
import string


MAGIC_BYTES = {
    b"\x50\x4B\x03\x04": "application/zip",
    b"\x52\x61\x72\x21\x1A\x07": "application/x-rar-compressed",
    b"\x25\x50\x44\x46": "application/pdf",
    b"\x89\x50\x4E\x47\x0D\x0A\x1A\x0A": "image/png",
    b"\xFF\xD8\xFF": "image/jpeg",
    b"\x47\x49\x46\x38": "image/gif",
    b"\x49\x44\x33": "audio/mpeg",
    b"\xFF\xFB": "audio/mpeg",
}


def is_printable_ascii(
    data: bytes,
    threshold: float = 0.95,
) -> bool:
    """Check whether payload is predominantly printable ASCII."""

    if not data:
        return False

    printable_chars = set(
        bytes(string.printable, "ascii")
    )

    printable_count = sum(
        1
        for byte in data
        if byte in printable_chars
    )

    return (
        printable_count / len(data)
    ) >= threshold


def _entropy(data: bytes) -> float:
    """Calculate Shannon entropy."""

    if not data:
        return 0.0

    counts = [0] * 256

    for byte in data:
        counts[byte] += 1

    entropy = 0.0

    for count in counts:
        if count == 0:
            continue

        probability = count / len(data)

        entropy -= (
            probability
            * math.log2(probability)
        )

    return entropy


def identify_payload(
    data: bytes,
) -> tuple[str, str | None]:
    """
    Identify recovered payload.

    Returns:
        (status, MIME type)

    Status:
        RECOVERED
        KEY_REQUIRED
        INVALID_PAYLOAD
    """

    if not data:
        return "INVALID_PAYLOAD", None

    # ---------------------------------------------------------
    # Standard magic bytes
    # ---------------------------------------------------------

    for magic, mime in MAGIC_BYTES.items():

        if data.startswith(magic):
            return "RECOVERED", mime

    # ---------------------------------------------------------
    # RIFF containers
    # ---------------------------------------------------------

    if data.startswith(b"RIFF") and len(data) >= 12:

        format_type = data[8:12]

        if format_type == b"WAVE":
            return "RECOVERED", "audio/wav"

        if format_type == b"AVI ":
            return "RECOVERED", "video/x-msvideo"

    # ---------------------------------------------------------
    # Plain text
    # ---------------------------------------------------------

    if is_printable_ascii(data):
        return "RECOVERED", "text/plain"

    # ---------------------------------------------------------
    # High entropy / potentially encrypted payload
    # ---------------------------------------------------------

    entropy = _entropy(data)

    if entropy > 7.5:
        return (
            "KEY_REQUIRED",
            "application/octet-stream",
        )

    return "INVALID_PAYLOAD", None