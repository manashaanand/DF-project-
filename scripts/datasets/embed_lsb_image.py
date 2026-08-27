"""Random LSB embedding for synthetic stego image generation."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def embed_lsb_random(image: np.ndarray, payload_rate: float, seed: int | None = None) -> np.ndarray:
    """
    Embed random bits into LSBs of pixel values.

    Args:
        image: Grayscale uint8 array (H, W) or RGB (H, W, 3).
        payload_rate: Fraction of embeddable LSB positions to modify (0.0–1.0).
        seed: Optional RNG seed for reproducibility.

    Returns:
        Stego image with same dtype/shape as input.
    """
    if not 0.0 <= payload_rate <= 1.0:
        raise ValueError("payload_rate must be between 0.0 and 1.0")

    rng = np.random.default_rng(seed)
    stego = image.copy()

    if stego.ndim == 2:
        flat = stego.reshape(-1)
        n_bits = int(flat.size * payload_rate)
        if n_bits == 0:
            return stego
        indices = rng.choice(flat.size, size=n_bits, replace=False)
        bits = rng.integers(0, 2, size=n_bits, dtype=np.uint8)
        flat[indices] = (flat[indices] & 0xFE) | bits
        return flat.reshape(stego.shape)

    if stego.ndim == 3 and stego.shape[2] == 3:
        flat = stego.reshape(-1)
        n_bits = int(flat.size * payload_rate)
        if n_bits == 0:
            return stego
        indices = rng.choice(flat.size, size=n_bits, replace=False)
        bits = rng.integers(0, 2, size=n_bits, dtype=np.uint8)
        flat[indices] = (flat[indices] & 0xFE) | bits
        return flat.reshape(stego.shape)

    raise ValueError("Unsupported image shape; expected grayscale or RGB")


def embed_file(
    input_path: Path,
    output_path: Path,
    payload_rate: float,
    seed: int | None = None,
) -> None:
    image = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Unable to read image: {input_path}")
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    stego = embed_lsb_random(image, payload_rate=payload_rate, seed=seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), stego)


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed random LSB payload into an image")
    parser.add_argument("input", type=Path, help="Cover image path")
    parser.add_argument("output", type=Path, help="Output stego image path")
    parser.add_argument("--rate", type=float, default=0.3, help="Payload rate (0.0–1.0)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()
    embed_file(args.input, args.output, payload_rate=args.rate, seed=args.seed)
    print(f"Wrote stego image to {args.output}")


if __name__ == "__main__":
    main()
