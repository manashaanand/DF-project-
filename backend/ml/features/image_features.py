"""OpenCV-based image preprocessing and steganalysis-oriented feature extraction."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

DEFAULT_INPUT_SIZE = (256, 256)

# SRM-inspired 3x3 high-pass kernel (KV filter subset used in steganalysis literature)
_SRM_KERNEL = np.array(
    [[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, -1.0, 2.0, -1.0, 0.0], [0.0, 2.0, -4.0, 2.0, 0.0], [0.0, -1.0, 2.0, -1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]],
    dtype=np.float32,
)


def load_grayscale_image(path: str | Path) -> np.ndarray:
    """Load image as single-channel uint8 array."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    return image


def preprocess_image(
    image: np.ndarray,
    size: tuple[int, int] = DEFAULT_INPUT_SIZE,
) -> np.ndarray:
    """Resize grayscale image to target size."""
    if image.ndim != 2:
        raise ValueError("Expected grayscale image with shape (H, W)")
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def compute_srm_residual(image: np.ndarray) -> np.ndarray:
    """Apply SRM-style high-pass filtering to obtain a residual map."""
    if image.ndim != 2:
        raise ValueError("Expected grayscale image with shape (H, W)")
    filtered = cv2.filter2D(image.astype(np.float32), -1, _SRM_KERNEL)
    return filtered


def _entropy(values: np.ndarray) -> float:
    hist, _ = np.histogram(values.flatten(), bins=256, range=(0, 256))
    total = hist.sum()
    if total == 0:
        return 0.0
    probs = hist[hist > 0] / total
    return float(-np.sum(probs * np.log2(probs)))


def _chi_square_lsb_pvalue(image: np.ndarray) -> float:
    """
    Chi-square test on LSB pairs (classic LSB steganalysis statistic).
    Returns p-value in [0, 1]; lower values suggest LSB embedding.
    """
    pairs = (image.astype(np.uint8) >> 1).flatten()
    hist = np.bincount(pairs, minlength=128)
    observed = hist.astype(np.float64)
    expected = (observed[:-1] + observed[1:]) / 2.0
    observed_pairs = observed[1:]
    mask = expected > 0
    if not np.any(mask):
        return 1.0
    chi2 = np.sum(((observed_pairs[mask] - expected[mask]) ** 2) / expected[mask])
    # Approximate p-value via simplified survival; clamp to [0, 1]
    # Uses chi2 with ~50 dof heuristic for 128 bins
    dof = max(int(np.sum(mask)) - 1, 1)
    # Wilson-Hilferty approximation for upper tail
    z = ((chi2 / dof) ** (1 / 3) - (1 - 2 / (9 * dof))) / np.sqrt(2 / (9 * dof))
    from math import erfc, sqrt

    p_value = 0.5 * erfc(z / sqrt(2))
    return float(np.clip(p_value, 0.0, 1.0))


def _lsb_balance_ratio(image: np.ndarray) -> float:
    """Ratio of 1-bits in LSB plane; ~0.5 expected for natural images."""
    lsb = image.astype(np.uint8) & 1
    return float(lsb.mean())


def compute_feature_summary(image: np.ndarray) -> dict[str, float]:
    """Extract interpretable steganalysis features from a grayscale image."""
    resized = preprocess_image(image)
    residual = compute_srm_residual(resized)
    return {
        "entropy": _entropy(resized),
        "mean_residual": float(residual.mean()),
        "std_residual": float(residual.std()),
        "chi_square_p": _chi_square_lsb_pvalue(resized),
        "lsb_balance_ratio": _lsb_balance_ratio(resized),
        "width": float(resized.shape[1]),
        "height": float(resized.shape[0]),
    }


def extract_cnn_input(
    image: np.ndarray,
    size: tuple[int, int] = DEFAULT_INPUT_SIZE,
) -> np.ndarray:
    """
    Build normalized residual map input for the CNN: shape (H, W, 1), float32.
    Residual is scaled to approximately [0, 1].
    """
    resized = preprocess_image(image, size=size)
    residual = compute_srm_residual(resized)
    residual = residual - residual.min()
    denom = residual.max() - residual.min()
    if denom > 0:
        residual = residual / denom
    return residual.astype(np.float32)[..., np.newaxis]


def get_preprocessing_config(size: tuple[int, int] = DEFAULT_INPUT_SIZE) -> dict:
    """Serializable preprocessing configuration saved alongside the model."""
    return {
        "input_size": list(size),
        "channels": 1,
        "normalization": "min_max_residual",
        "filter": "srm_5x5_highpass",
    }


def save_preprocessing_config(path: Path, size: tuple[int, int] = DEFAULT_INPUT_SIZE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(get_preprocessing_config(size), indent=2), encoding="utf-8")


def load_preprocessing_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
