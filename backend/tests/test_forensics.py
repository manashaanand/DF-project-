import pytest
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

from app.forensic.image_forensics import (
    analyze_eof_data,
    analyze_lsb_entropy,
    analyze_metadata,
    perform_image_forensics
)
from app.forensic.technique_identifier import identify_techniques
from app.detectors.base import ForensicFinding


@pytest.fixture
def dummy_png_path(tmp_path) -> Path:
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    p = tmp_path / "dummy.png"
    cv2.imwrite(str(p), img)
    return p


@pytest.fixture
def appended_png_path(tmp_path) -> Path:
    # 1. Write normal PNG
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    p = tmp_path / "appended.png"
    cv2.imwrite(str(p), img)
    
    # 2. Append data after IEND
    with open(p, "ab") as f:
        f.write(b"SECRET_DATA_APPENDED")
    return p


@pytest.fixture
def lsb_stego_png_path(tmp_path) -> Path:
    # Create image with high entropy LSB plane
    rng = np.random.default_rng(42)
    # Background
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    # Overwrite LSB with random bits to simulate high entropy
    random_bits = rng.integers(0, 2, size=(64, 64, 3), dtype=np.uint8)
    img = (img & ~1) | random_bits
    
    p = tmp_path / "lsb_stego.png"
    cv2.imwrite(str(p), img)
    return p


def test_analyze_eof_data_appended(appended_png_path):
    finding = analyze_eof_data(appended_png_path)
    assert finding is not None
    assert finding.category == "appended_data"
    assert "SECRET_DATA" in finding.evidence["preview"]
    assert finding.evidence["appended_length"] == 20


def test_analyze_eof_data_clean(dummy_png_path):
    finding = analyze_eof_data(dummy_png_path)
    assert finding is None


def test_analyze_lsb_entropy_high(lsb_stego_png_path):
    findings = analyze_lsb_entropy(lsb_stego_png_path)
    assert len(findings) > 0
    assert findings[0].category == "lsb_analysis"
    assert "channels" in findings[0].evidence


def test_analyze_lsb_entropy_clean(dummy_png_path):
    findings = analyze_lsb_entropy(dummy_png_path)
    assert len(findings) == 0


def test_identify_techniques_appended():
    findings = [
        ForensicFinding(
            category="appended_data",
            description="Found 20 bytes",
            evidence={"preview": "SECRET"}
        )
    ]
    techs = identify_techniques(findings)
    assert len(techs) == 1
    assert techs[0].technique == "APPENDED_DATA"


def test_identify_techniques_lsb_high_confidence():
    findings = [
        ForensicFinding(
            category="lsb_analysis",
            description="High entropy LSB",
        )
    ]
    # If ML score also high -> high confidence LSB replacement
    techs = identify_techniques(findings, classical_score=0.9)
    assert len(techs) == 1
    assert techs[0].technique == "LSB_REPLACEMENT"
    assert techs[0].confidence == "high"


def test_identify_techniques_ml_only():
    # High ML score but no hard forensic finding
    techs = identify_techniques([], classical_score=0.8)
    assert len(techs) == 1
    assert techs[0].technique == "UNKNOWN_STEGANOGRAPHY"
    assert techs[0].confidence == "medium"
