from pathlib import Path

from app.extractors.payload_extractor import (
    extract_appended_payload,
)


PNG_HEADER = (
    b"\x89PNG\r\n\x1a\n"
)

PNG_IEND = (
    b"\x00\x00\x00\x00"
    b"IEND"
    b"\xae\x42\x60\x82"
)


def test_png_zip_payload_is_recovered(
    tmp_path: Path,
):
    payload = (
        b"PK\x03\x04"
        + b"test-payload"
    )

    image_path = (
        tmp_path / "evidence.png"
    )

    image_path.write_bytes(
        PNG_HEADER
        + PNG_IEND
        + payload
    )

    result = extract_appended_payload(
        image_path
    )

    assert result.status == "RECOVERED"
    assert result.payload_type == "application/zip"
    assert result.payload_size == len(payload)
    assert result.sha256 is not None


def test_png_without_payload_is_not_recoverable(
    tmp_path: Path,
):
    image_path = (
        tmp_path / "clean.png"
    )

    image_path.write_bytes(
        PNG_HEADER
        + PNG_IEND
    )

    result = extract_appended_payload(
        image_path
    )

    assert result.status == "NOT_RECOVERABLE"
    assert result.sha256 is None


def test_png_invalid_appended_bytes_are_not_claimed_as_recovered(
    tmp_path: Path,
):
    image_path = (
        tmp_path / "invalid.png"
    )

    image_path.write_bytes(
        PNG_HEADER
        + PNG_IEND
        + b"\x01\x02\x03\x04"
    )

    result = extract_appended_payload(
        image_path
    )

    assert result.status == "INVALID_PAYLOAD"
    assert result.payload_size == 4
    assert result.sha256 is not None


def test_unsupported_image_format(
    tmp_path: Path,
):
    image_path = (
        tmp_path / "image.webp"
    )

    image_path.write_bytes(
        b"RIFF"
        + b"\x00" * 20
    )

    result = extract_appended_payload(
        image_path
    )

    assert result.status == "UNSUPPORTED"