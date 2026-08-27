import numpy as np
import pytest

from ml.features.image_features import (
    DEFAULT_INPUT_SIZE,
    compute_feature_summary,
    compute_srm_residual,
    extract_cnn_input,
    get_preprocessing_config,
    preprocess_image,
    save_preprocessing_config,
)


@pytest.fixture
def sample_grayscale() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(128, 128), dtype=np.uint8)


def test_preprocess_image_resizes(sample_grayscale):
    out = preprocess_image(sample_grayscale, size=(256, 256))
    assert out.shape == (256, 256)
    assert out.dtype == np.uint8


def test_srm_residual_shape(sample_grayscale):
    residual = compute_srm_residual(sample_grayscale)
    assert residual.shape == sample_grayscale.shape
    assert residual.dtype == np.float32


def test_extract_cnn_input_shape_and_range(sample_grayscale):
    tensor = extract_cnn_input(sample_grayscale, size=DEFAULT_INPUT_SIZE)
    assert tensor.shape == (256, 256, 1)
    assert tensor.dtype == np.float32
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0


def test_feature_summary_keys_and_bounds(sample_grayscale):
    features = compute_feature_summary(sample_grayscale)
    expected_keys = {
        "entropy",
        "mean_residual",
        "std_residual",
        "chi_square_p",
        "lsb_balance_ratio",
        "width",
        "height",
    }
    assert expected_keys.issubset(features.keys())
    assert 0.0 <= features["entropy"] <= 8.0
    assert 0.0 <= features["chi_square_p"] <= 1.0
    assert 0.0 <= features["lsb_balance_ratio"] <= 1.0


def test_preprocessing_config_roundtrip(tmp_path):
    path = tmp_path / "preprocessing_config.json"
    save_preprocessing_config(path)
    from ml.features.image_features import load_preprocessing_config

    cfg = load_preprocessing_config(path)
    assert cfg == get_preprocessing_config()
