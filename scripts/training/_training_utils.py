"""Shared utilities for training and evaluation scripts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Repo root: scripts/training -> parents[2]
REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "backend"


def ensure_backend_on_path() -> None:
    import sys

    backend_str = str(BACKEND_ROOT)
    if backend_str not in sys.path:
        sys.path.insert(0, backend_str)


def load_split_files(data_dir: Path, split: str) -> list[tuple[Path, int]]:
    """Return (path, label) pairs where label 0=cover, 1=stego."""
    pairs: list[tuple[Path, int]] = []
    for label_name, label in (("cover", 0), ("stego", 1)):
        split_dir = data_dir / label_name / split
        if not split_dir.is_dir():
            continue
        for path in sorted(split_dir.glob("*")):
            if path.is_file():
                pairs.append((path, label))
    return pairs


def build_arrays(
    pairs: list[tuple[Path, int]],
    input_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    from ml.features.image_features import extract_cnn_input, load_grayscale_image

    xs: list[np.ndarray] = []
    ys: list[int] = []
    for path, label in pairs:
        image = load_grayscale_image(path)
        xs.append(extract_cnn_input(image, size=input_size))
        ys.append(label)
    return np.stack(xs, axis=0), np.array(ys, dtype=np.float32)


def save_metrics(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
