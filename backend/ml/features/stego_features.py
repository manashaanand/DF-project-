"""
Improved statistical/residual feature extraction for
image steganography detection.

Designed for:
    - Training
    - Validation
    - Testing
    - Real-image inference

Input:
    Grayscale image

Output:
    1-D numerical feature vector

IMPORTANT:
    This extractor is intentionally focused on
    steganalysis-related residual/noise information.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


# =====================================================================
# SRM-STYLE HIGH-PASS FILTERS
# =====================================================================

SRM_FILTERS = [

    # 3x3 Laplacian
    np.array(
        [
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0],
        ],
        dtype=np.float32,
    ),

    # 3x3 diagonal Laplacian
    np.array(
        [
            [1, 0, 1],
            [0, -4, 0],
            [1, 0, 1],
        ],
        dtype=np.float32,
    ),

    # Horizontal edge residual
    np.array(
        [
            [0, 0, 0],
            [-1, 2, -1],
            [0, 0, 0],
        ],
        dtype=np.float32,
    ),

    # Vertical edge residual
    np.array(
        [
            [0, -1, 0],
            [0, 2, 0],
            [0, -1, 0],
        ],
        dtype=np.float32,
    ),

    # Diagonal residual
    np.array(
        [
            [-1, 0, 0],
            [0, 2, 0],
            [0, 0, -1],
        ],
        dtype=np.float32,
    ),

    # Opposite diagonal
    np.array(
        [
            [0, 0, -1],
            [0, 2, 0],
            [-1, 0, 0],
        ],
        dtype=np.float32,
    ),

    # 5x5 SRM-style filter
    np.array(
        [
            [0, 0, 0, 0, 0],
            [0, -1, 2, -1, 0],
            [0, 2, -4, 2, 0],
            [0, -1, 2, -1, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    ),

    # 5x5 directional filter
    np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 0, -1, 0],
            [0, 0, 0, 0, 0],
            [0, -1, 0, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    ),

]


# =====================================================================
# IMAGE LOADING
# =====================================================================

def load_grayscale(
    path: str | Path,
) -> np.ndarray:

    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise ValueError(
            f"Unable to read image: {path}"
        )

    if image.dtype != np.uint8:

        image = np.clip(
            image,
            0,
            255,
        ).astype(np.uint8)

    return image


# =====================================================================
# FAST ROBUST STATISTICS
# =====================================================================

def _fast_stats(
    values: np.ndarray,
) -> list[float]:
    """
    Fast statistics.

    Avoids expensive np.percentile calls over
    full-resolution arrays.

    Returns:
        mean
        std
        mean absolute value
        std absolute value
        minimum
        maximum
        median approximation
        upper-tail statistics
        zero/sign statistics
    """

    values = np.asarray(
        values,
        dtype=np.float32,
    )

    values = values[
        np.isfinite(values)
    ]

    if values.size == 0:

        return [0.0] * 14

    mean = float(
        np.mean(values)
    )

    std = float(
        np.std(values)
    )

    abs_values = np.abs(values)

    abs_mean = float(
        np.mean(abs_values)
    )

    abs_std = float(
        np.std(abs_values)
    )

    minimum = float(
        np.min(values)
    )

    maximum = float(
        np.max(values)
    )

    # Median using partition instead of full sort.
    middle = values.size // 2

    partitioned = np.partition(
        values,
        middle,
    )

    median = float(
        partitioned[middle]
    )

    # Tail thresholds.
    abs_p90 = float(
        np.mean(
            abs_values >=
            np.percentile(
                abs_values,
                90,
            )
        )
    )

    abs_p95 = float(
        np.mean(
            abs_values >=
            np.percentile(
                abs_values,
                95,
            )
        )
    )

    # Histogram of residual magnitudes.
    magnitude_hist, _ = np.histogram(
        abs_values,
        bins=8,
        range=(
            0,
            max(
                float(np.max(abs_values)),
                1e-6,
            ),
        ),
    )

    magnitude_hist = (
        magnitude_hist.astype(
            np.float32
        )
    )

    total = float(
        magnitude_hist.sum()
    )

    if total > 0:

        magnitude_hist /= total

    # Sign balance.
    positive_ratio = float(
        np.mean(values > 0)
    )

    negative_ratio = float(
        np.mean(values < 0)
    )

    zero_ratio = float(
        np.mean(values == 0)
    )

    return [
        mean,
        std,
        abs_mean,
        abs_std,
        minimum,
        maximum,
        median,
        abs_p90,
        abs_p95,
        positive_ratio,
        negative_ratio,
        zero_ratio,
        float(magnitude_hist[0]),
        float(magnitude_hist[1]),
    ]


# =====================================================================
# HISTOGRAM FEATURES
# =====================================================================

def _histogram_features(
    image: np.ndarray,
) -> list[float]:

    histogram, _ = np.histogram(
        image,
        bins=32,
        range=(0, 256),
    )

    histogram = histogram.astype(
        np.float32
    )

    total = float(
        histogram.sum()
    )

    if total > 0:

        histogram /= total

    return histogram.tolist()


# =====================================================================
# LSB FEATURES
# =====================================================================

def _lsb_features(
    image: np.ndarray,
) -> list[float]:

    lsb = (
        image & 1
    ).astype(
        np.float32
    )

    features = [

        float(
            np.mean(lsb)
        ),

        float(
            np.std(lsb)
        ),

    ]

    # Horizontal transitions.
    if image.shape[1] > 1:

        horizontal = np.abs(
            lsb[:, 1:]
            -
            lsb[:, :-1]
        )

        features.append(
            float(
                np.mean(horizontal)
            )
        )

    else:

        features.append(0.0)

    # Vertical transitions.
    if image.shape[0] > 1:

        vertical = np.abs(
            lsb[1:, :]
            -
            lsb[:-1, :]
        )

        features.append(
            float(
                np.mean(vertical)
            )
        )

    else:

        features.append(0.0)

    # Diagonal transitions.
    if (
        image.shape[0] > 1
        and image.shape[1] > 1
    ):

        diagonal = np.abs(
            lsb[1:, 1:]
            -
            lsb[:-1, :-1]
        )

        features.append(
            float(
                np.mean(diagonal)
            )
        )

    else:

        features.append(0.0)

    # 2x2 consistency.
    if (
        image.shape[0] > 1
        and image.shape[1] > 1
    ):

        block_mean = (

            lsb[:-1, :-1]
            +
            lsb[1:, :-1]
            +
            lsb[:-1, 1:]
            +
            lsb[1:, 1:]

        ) / 4.0

        features.append(
            float(
                np.std(block_mean)
            )
        )

    else:

        features.append(0.0)

    return features


# =====================================================================
# PIXEL / DIFFERENCE FEATURES
# =====================================================================

def _pixel_features(
    image: np.ndarray,
) -> list[float]:

    image_float = image.astype(
        np.float32
    )

    features = []

    features.extend(
        _fast_stats(
            image_float
        )
    )

    # Horizontal differences.
    if image.shape[1] > 1:

        horizontal = (
            image_float[:, 1:]
            -
            image_float[:, :-1]
        )

        features.extend(
            _fast_stats(horizontal)
        )

    # Vertical differences.
    if image.shape[0] > 1:

        vertical = (
            image_float[1:, :]
            -
            image_float[:-1, :]
        )

        features.extend(
            _fast_stats(vertical)
        )

    # Diagonal differences.
    if (
        image.shape[0] > 1
        and image.shape[1] > 1
    ):

        diagonal = (
            image_float[1:, 1:]
            -
            image_float[:-1, :-1]
        )

        features.extend(
            _fast_stats(diagonal)
        )

    return features


# =====================================================================
# RESIDUAL FEATURES
# =====================================================================

def _residual_features(
    image: np.ndarray,
) -> list[float]:

    image_float = image.astype(
        np.float32
    )

    features = []

    for kernel in SRM_FILTERS:

        residual = cv2.filter2D(
            image_float,
            cv2.CV_32F,
            kernel,
            borderType=cv2.BORDER_REFLECT,
        )

        features.extend(
            _fast_stats(
                residual
            )
        )

        absolute = np.abs(
            residual
        )

        features.extend(
            [
                float(
                    np.mean(absolute)
                ),
                float(
                    np.std(absolute)
                ),
                float(
                    np.mean(
                        absolute > 1.0
                    )
                ),
                float(
                    np.mean(
                        absolute > 2.0
                    )
                ),
            ]
        )

    return features


# =====================================================================
# LOCAL NOISE FEATURES
# =====================================================================

def _local_noise_features(
    image: np.ndarray,
) -> list[float]:

    image_float = image.astype(
        np.float32
    )

    # Gaussian residual.
    blur = cv2.GaussianBlur(
        image_float,
        (5, 5),
        0,
    )

    residual = (
        image_float
        -
        blur
    )

    features = []

    features.extend(
        _fast_stats(
            residual
        )
    )

    # Median residual.
    median = cv2.medianBlur(
        image,
        5,
    ).astype(
        np.float32
    )

    median_residual = (
        image_float
        -
        median
    )

    features.extend(
        _fast_stats(
            median_residual
        )
    )

    return features


# =====================================================================
# TEXTURE FEATURES
# =====================================================================

def _texture_features(
    image: np.ndarray,
) -> list[float]:

    image_float = image.astype(
        np.float32
    )

    features = []

    # Local variance using mean of squares.
    mean = cv2.GaussianBlur(
        image_float,
        (7, 7),
        0,
    )

    mean_squared = cv2.GaussianBlur(
        image_float * image_float,
        (7, 7),
        0,
    )

    variance = np.maximum(
        mean_squared
        -
        mean * mean,
        0,
    )

    features.extend(
        _fast_stats(
            variance
        )
    )

    # Sobel gradient magnitude.
    gx = cv2.Sobel(
        image_float,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gy = cv2.Sobel(
        image_float,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    magnitude = cv2.magnitude(
        gx,
        gy,
    )

    features.extend(
        _fast_stats(
            magnitude
        )
    )

    return features


# =====================================================================
# COMPLETE FEATURE EXTRACTION
# =====================================================================

def extract_features(
    image: np.ndarray,
) -> np.ndarray:

    if image is None:

        raise ValueError(
            "Image cannot be None."
        )

    if image.ndim != 2:

        raise ValueError(
            "Expected grayscale image "
            "with shape (H, W)."
        )

    if image.dtype != np.uint8:

        image = np.clip(
            image,
            0,
            255,
        ).astype(
            np.uint8
        )

    features = []

    # ---------------------------------------------------------------
    # 1. Pixel distribution
    # ---------------------------------------------------------------

    features.extend(
        _histogram_features(
            image
        )
    )

    # ---------------------------------------------------------------
    # 2. LSB statistics
    # ---------------------------------------------------------------

    features.extend(
        _lsb_features(
            image
        )
    )

    # ---------------------------------------------------------------
    # 3. Pixel/difference statistics
    # ---------------------------------------------------------------

    features.extend(
        _pixel_features(
            image
        )
    )

    # ---------------------------------------------------------------
    # 4. SRM residuals
    # ---------------------------------------------------------------

    features.extend(
        _residual_features(
            image
        )
    )

    # ---------------------------------------------------------------
    # 5. Local noise
    # ---------------------------------------------------------------

    features.extend(
        _local_noise_features(
            image
        )
    )

    # ---------------------------------------------------------------
    # 6. Texture
    # ---------------------------------------------------------------

    features.extend(
        _texture_features(
            image
        )
    )

    result = np.asarray(
        features,
        dtype=np.float32,
    )

    result = np.nan_to_num(
        result,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return result


# =====================================================================
# FILE EXTRACTION
# =====================================================================

def extract_features_from_file(
    path: str | Path,
) -> np.ndarray:

    image = load_grayscale(
        path
    )

    return extract_features(
        image
    )


# =====================================================================
# DIRECT TEST
# =====================================================================

if __name__ == "__main__":

    import sys

    if len(sys.argv) != 2:

        print(
            "Usage:"
        )

        print(
            "python stego_features.py <image>"
        )

        raise SystemExit(1)

    path = Path(
        sys.argv[1]
    )

    features = (
        extract_features_from_file(
            path
        )
    )

    print("=" * 70)
    print("IMPROVED STEGO FEATURE EXTRACTION TEST")
    print("=" * 70)

    print(
        f"Image   : {path}"
    )

    print(
        f"Features: {len(features)}"
    )

    print(
        f"Dtype   : {features.dtype}"
    )

    print(
        f"Shape   : {features.shape}"
    )

    print(
        f"Finite  : "
        f"{np.isfinite(features).all()}"
    )

    print(
        f"Minimum : "
        f"{features.min():.6f}"
    )

    print(
        f"Maximum : "
        f"{features.max():.6f}"
    )

    print("=" * 70)