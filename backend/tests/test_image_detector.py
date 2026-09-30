import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from app.core.exceptions import ModelUnavailableError
from app.detectors.image_detector import ImageDetector
from ml.inference.image_predictor import ImagePredictor


@pytest.fixture
def sample_image(tmp_path) -> Path:
    rng = np.random.default_rng(1)
    image = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
    path = tmp_path / "sample.png"
    cv2.imwrite(str(path), image)
    return path


def test_predictor_unavailable_when_model_missing(tmp_path):
    predictor = ImagePredictor(tmp_path / "missing" / "model.pkl")
    assert predictor.is_available() is False
    with pytest.raises(FileNotFoundError):
        predictor.load()


def test_image_detector_raises_without_model(sample_image, tmp_path):
    predictor = ImagePredictor(tmp_path / "no_model.pkl")
    mock_stegexpose = MagicMock()
    mock_stegexpose.analyze_file.return_value = MagicMock(
        available=False, score=None, error="not configured"
    )
    detector = ImageDetector(predictor=predictor, stegexpose=mock_stegexpose)
    with pytest.raises(ModelUnavailableError):
        detector.analyze(sample_image)


def test_image_detector_inference_with_mock_predictor(sample_image):
    """Test detector using a mocked predictor (no TensorFlow required)."""
    mock_predictor = MagicMock(spec=ImagePredictor)
    mock_predictor.is_available.return_value = True
    mock_predictor.predict_file.return_value = 0.75
    mock_predictor.model_version = "LogisticRegression"

    mock_stegexpose = MagicMock()
    mock_stegexpose.analyze_file.return_value = MagicMock(
        available=False, score=None, error="not configured"
    )

    detector = ImageDetector(predictor=mock_predictor, stegexpose=mock_stegexpose)
    result = detector.analyze(sample_image)

    assert result.model_loaded is True
    assert result.label in {"cover", "stego", "inconclusive"}
    assert 0.0 <= result.confidence <= 1.0
    assert result.classical_score is not None
    assert 0.0 <= result.classical_score <= 1.0
    assert "entropy" in result.features
    assert result.supplementary_available is False
