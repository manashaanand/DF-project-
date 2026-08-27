import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from app.core.exceptions import ModelUnavailableError
from app.detectors.image_detector import ImageDetector
from ml.features.image_features import save_preprocessing_config
from ml.inference.image_predictor import ImagePredictor


@pytest.fixture
def sample_image(tmp_path) -> Path:
    rng = np.random.default_rng(1)
    image = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
    path = tmp_path / "sample.png"
    cv2.imwrite(str(path), image)
    return path


@pytest.fixture
def tiny_model_dir(tmp_path) -> Path:
    pytest.importorskip("tensorflow")
    from ml.models.image_cnn import build_image_cnn

    model_dir = tmp_path / "image_cnn"
    model_dir.mkdir()
    model = build_image_cnn(input_shape=(256, 256, 1))
    model.save(model_dir / "model.keras")
    save_preprocessing_config(model_dir / "preprocessing_config.json")
    (model_dir / "metadata.json").write_text(
        json.dumps({"model_version": "test_v0"}),
        encoding="utf-8",
    )
    return model_dir


def test_predictor_unavailable_when_model_missing(tmp_path):
    predictor = ImagePredictor(tmp_path / "missing" / "model.keras")
    assert predictor.is_available() is False
    with pytest.raises(FileNotFoundError):
        predictor.load()


def test_predictor_load_and_predict(sample_image, tiny_model_dir):
    predictor = ImagePredictor(
        tiny_model_dir / "model.keras",
        tiny_model_dir / "metadata.json",
        tiny_model_dir / "preprocessing_config.json",
    )
    assert predictor.is_available() is True
    predictor.load()
    score = predictor.predict_file(sample_image)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_image_detector_raises_without_model(sample_image, tmp_path):
    predictor = ImagePredictor(tmp_path / "no_model.keras")
    detector = ImageDetector(predictor=predictor, stegexpose=MagicMock(available=False))
    with pytest.raises(ModelUnavailableError):
        detector.analyze(sample_image)


@patch("app.detectors.image_detector.StegExposeWrapper")
def test_image_detector_inference_with_mock_stegexpose(mock_wrapper_cls, sample_image, tiny_model_dir):
    mock_wrapper = MagicMock()
    mock_wrapper.analyze_file.return_value = MagicMock(
        available=False,
        score=None,
        error="not configured",
    )
    mock_wrapper_cls.return_value = mock_wrapper

    predictor = ImagePredictor(
        tiny_model_dir / "model.keras",
        tiny_model_dir / "metadata.json",
        tiny_model_dir / "preprocessing_config.json",
    )
    detector = ImageDetector(predictor=predictor, stegexpose=mock_wrapper)
    result = detector.analyze(sample_image)

    assert result.model_loaded is True
    assert result.label in {"cover", "stego", "inconclusive"}
    assert 0.0 <= result.confidence <= 1.0
    assert result.cnn_score is not None
    assert 0.0 <= result.cnn_score <= 1.0
    assert "entropy" in result.features
    assert result.supplementary_available is False
