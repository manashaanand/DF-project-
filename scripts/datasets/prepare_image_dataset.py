"""
Prepare image dataset with cover/stego classes and leakage-free train/val/test splits.

Splitting is performed at the **source cover image** level: all stego variants derived
from the same cover image are assigned to the same split, preventing data leakage.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

# Allow importing embedder from sibling module
sys.path.insert(0, str(Path(__file__).resolve().parent))
from embed_lsb_image import embed_lsb_random  # noqa: E402

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pgm"}


def _list_images(directory: Path) -> list[Path]:
    return sorted(
        p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def _generate_synthetic_covers(output_dir: Path, count: int, size: int, seed: int) -> list[Path]:
    """Generate simple synthetic grayscale covers for demo/testing when no dataset is available."""
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i in range(count):
        # Structured noise + gradients to mimic varied texture
        base = rng.integers(0, 256, size=(size, size), dtype=np.uint8)
        gradient = np.linspace(0, 255, size, dtype=np.uint8)
        image = np.clip(base.astype(np.int16) + gradient[:, None], 0, 255).astype(np.uint8)
        path = output_dir / f"synthetic_{i:04d}.png"
        cv2.imwrite(str(path), image)
        paths.append(path)
    return paths


def _assign_splits(source_ids: list[str], train_ratio: float, val_ratio: float, seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    ids = source_ids[:]
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    mapping: dict[str, str] = {}
    for idx, source_id in enumerate(ids):
        if idx < n_train:
            mapping[source_id] = "train"
        elif idx < n_train + n_val:
            mapping[source_id] = "val"
        else:
            mapping[source_id] = "test"
    return mapping


def prepare_dataset(
    covers_dir: Path | None,
    output_dir: Path,
    payload_rates: list[float],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    synthetic_count: int = 0,
    synthetic_size: int = 256,
) -> dict:
    raw_covers_dir = output_dir / "raw" / "covers"
    if covers_dir is not None:
        cover_paths = _list_images(covers_dir)
        if not cover_paths:
            raise FileNotFoundError(f"No images found in {covers_dir}")
    elif synthetic_count > 0:
        cover_paths = _generate_synthetic_covers(raw_covers_dir, synthetic_count, synthetic_size, seed)
    else:
        raise ValueError("Provide --covers-dir or --synthetic-count > 0")

    source_ids = [p.stem for p in cover_paths]
    split_map = _assign_splits(source_ids, train_ratio, val_ratio, seed)

    manifest: dict = {
        "seed": seed,
        "payload_rates": payload_rates,
        "split_policy": "by_source_cover_image",
        "splits": {"train": [], "val": [], "test": []},
        "files": [],
    }

    for cover_path in cover_paths:
        source_id = cover_path.stem
        split = split_map[source_id]

        cover_out = output_dir / "cover" / split / f"{source_id}.png"
        cover_out.parent.mkdir(parents=True, exist_ok=True)
        image = cv2.imread(str(cover_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Unable to read cover image: {cover_path}")
        cv2.imwrite(str(cover_out), image)
        manifest["splits"][split].append(source_id)
        manifest["files"].append({"path": str(cover_out.relative_to(output_dir)), "class": "cover", "split": split, "source_id": source_id})

        for rate in payload_rates:
            stego = embed_lsb_random(image, payload_rate=rate, seed=seed + int(rate * 1000) + hash(source_id) % 10000)
            stego_name = f"{source_id}_rate{rate:.2f}.png"
            stego_out = output_dir / "stego" / split / stego_name
            stego_out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(stego_out), stego)
            manifest["files"].append(
                {
                    "path": str(stego_out.relative_to(output_dir)),
                    "class": "stego",
                    "split": split,
                    "source_id": source_id,
                    "payload_rate": rate,
                }
            )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare cover/stego image dataset with leakage-free splits")
    parser.add_argument("--covers-dir", type=Path, default=None, help="Directory of cover images (e.g. BOSSbase subset)")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/images"), help="Output dataset root")
    parser.add_argument("--rates", type=float, nargs="+", default=[0.1, 0.3, 0.5], help="LSB payload rates")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic-count", type=int, default=0, help="Generate N synthetic covers if no --covers-dir")
    parser.add_argument("--synthetic-size", type=int, default=256)
    args = parser.parse_args()

    manifest = prepare_dataset(
        covers_dir=args.covers_dir,
        output_dir=args.output_dir,
        payload_rates=args.rates,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        synthetic_count=args.synthetic_count,
        synthetic_size=args.synthetic_size,
    )
    for split in ("train", "val", "test"):
        n_cover = sum(1 for f in manifest["files"] if f["split"] == split and f["class"] == "cover")
        n_stego = sum(1 for f in manifest["files"] if f["split"] == split and f["class"] == "stego")
        print(f"{split}: {n_cover} cover, {n_stego} stego")
    print(f"Manifest written to {args.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
