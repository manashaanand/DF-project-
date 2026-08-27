"""
Remove exact duplicate images that occur across different dataset splits.

For duplicates across train/val/test:
- Keep the image in the earliest split according to:
    train -> val -> test
- Remove the duplicate source from the later split.
- Remove its corresponding stego variants as well.

Duplicates entirely within the same split are also removed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2


DATA_DIR = Path("data/processed/images")

SPLIT_ORDER = {
    "train": 0,
    "val": 1,
    "test": 2,
}

CLASSES = ("cover", "stego")


def image_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []

    return sorted(
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() == ".png"
    )


def image_hash(path: Path) -> str:
    """
    Hash the actual decoded pixel data rather than the PNG file bytes.
    This catches visually/pixel-identical images saved with different
    PNG metadata or compression settings.
    """

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if image is None:
        raise ValueError(f"Unable to read image: {path}")

    return hashlib.sha256(image.tobytes()).hexdigest()


def source_id_from_cover(path: Path) -> str:
    return path.stem


def remove_source_from_split(source_id: str, split: str) -> list[Path]:
    removed = []

    # Remove cover
    cover_path = DATA_DIR / "cover" / split / f"{source_id}.png"

    if cover_path.exists():
        cover_path.unlink()
        removed.append(cover_path)

    # Remove all stego variants belonging to the source
    stego_dir = DATA_DIR / "stego" / split

    if stego_dir.exists():
        for path in stego_dir.glob(f"{source_id}_rate*.png"):
            path.unlink()
            removed.append(path)

    return removed


def main() -> None:
    print("=" * 70)
    print("EXACT DUPLICATE CLEANUP")
    print("=" * 70)

    print()
    print("Split priority:")
    print("  train -> val -> test")
    print()

    # ------------------------------------------------------------------
    # Build hash records for COVER images.
    # We use cover images as the source identity because all corresponding
    # stego variants are generated from those covers.
    # ------------------------------------------------------------------

    records = []

    for split in ("train", "val", "test"):
        cover_dir = DATA_DIR / "cover" / split

        for path in image_files(cover_dir):
            digest = image_hash(path)

            records.append(
                {
                    "hash": digest,
                    "split": split,
                    "source_id": source_id_from_cover(path),
                    "path": path,
                }
            )

    # ------------------------------------------------------------------
    # Group by image hash
    # ------------------------------------------------------------------

    groups = {}

    for record in records:
        groups.setdefault(record["hash"], []).append(record)

    duplicate_groups = [
        group for group in groups.values()
        if len(group) > 1
    ]

    print(f"Cover images checked : {len(records)}")
    print(f"Duplicate groups     : {len(duplicate_groups)}")
    print()

    if not duplicate_groups:
        print("No duplicate cover images found.")
        print("=" * 70)
        return

    total_removed = 0

    # ------------------------------------------------------------------
    # Process each duplicate group.
    # ------------------------------------------------------------------

    for index, group in enumerate(duplicate_groups, start=1):

        print("-" * 70)
        print(f"DUPLICATE GROUP {index}")
        print("-" * 70)

        for record in sorted(
            group,
            key=lambda x: SPLIT_ORDER[x["split"]]
        ):
            print(
                f"  {record['split']:5s} | "
                f"{record['path']}"
            )

        # Keep the copy in the earliest split.
        keep = min(
            group,
            key=lambda x: SPLIT_ORDER[x["split"]]
        )

        print()
        print(f"KEEP : {keep['path']}")

        # Remove every later copy.
        for record in group:

            if record is keep:
                continue

            print(
                f"REMOVE: {record['path']}"
            )

            removed = remove_source_from_split(
                record["source_id"],
                record["split"],
            )

            for removed_path in removed:
                print(
                    f"        deleted: {removed_path}"
                )

            total_removed += len(removed)

    print()
    print("=" * 70)
    print("CLEANUP COMPLETE")
    print("=" * 70)
    print(f"Files removed: {total_removed}")
    print()
    print("IMPORTANT:")
    print("Run validate_dataset.py again before training.")
    print("=" * 70)


if __name__ == "__main__":
    main()