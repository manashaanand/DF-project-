"""
Validate the prepared image steganography dataset.

Checks:
- Required directory structure
- Cover/stego counts
- Train/validation/test split counts
- File readability
- Duplicate filenames
- Source-cover leakage using manifest.json
- Class distribution
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2


DATA_DIR = Path("data/processed/images")

EXPECTED_SPLITS = ("train", "val", "test")
EXPECTED_CLASSES = ("cover", "stego")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pgm"}


def get_images(directory: Path) -> list[Path]:
    if not directory.exists():
        return []

    return sorted(
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def file_hash(path: Path) -> str:
    hasher = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            hasher.update(chunk)

    return hasher.hexdigest()


def validate_structure() -> bool:
    print("=" * 70)
    print("DATASET STRUCTURE VALIDATION")
    print("=" * 70)

    if not DATA_DIR.exists():
        print(f"ERROR: Dataset directory does not exist:")
        print(f"       {DATA_DIR}")
        return False

    print(f"Dataset directory: {DATA_DIR.resolve()}")
    print()

    valid = True

    for class_name in EXPECTED_CLASSES:
        for split in EXPECTED_SPLITS:
            directory = DATA_DIR / class_name / split

            if not directory.is_dir():
                print(f"ERROR: Missing directory: {directory}")
                valid = False

    if valid:
        print("Directory structure: OK")

    print()

    return valid


def validate_counts() -> dict:
    print("=" * 70)
    print("CLASS DISTRIBUTION")
    print("=" * 70)

    counts = {}

    for split in EXPECTED_SPLITS:
        counts[split] = {}

        for class_name in EXPECTED_CLASSES:
            directory = DATA_DIR / class_name / split
            images = get_images(directory)

            counts[split][class_name] = len(images)

            print(
                f"{split:5s} | "
                f"{class_name:5s} : "
                f"{len(images):6d}"
            )

    print()

    return counts


def validate_readability() -> bool:
    print("=" * 70)
    print("IMAGE READABILITY CHECK")
    print("=" * 70)

    unreadable = []

    for class_name in EXPECTED_CLASSES:
        for split in EXPECTED_SPLITS:
            directory = DATA_DIR / class_name / split

            for path in get_images(directory):
                image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

                if image is None:
                    unreadable.append(path)

    if unreadable:
        print(f"ERROR: {len(unreadable)} unreadable images found.")

        for path in unreadable[:20]:
            print(f"  {path}")

        if len(unreadable) > 20:
            print(f"  ... and {len(unreadable) - 20} more")

        return False

    print("All images readable: OK")
    print()

    return True


def validate_duplicates() -> bool:
    print("=" * 70)
    print("DUPLICATE IMAGE CHECK")
    print("=" * 70)

    hashes = {}
    duplicates = []

    for class_name in EXPECTED_CLASSES:
        for split in EXPECTED_SPLITS:
            directory = DATA_DIR / class_name / split

            for path in get_images(directory):
                digest = file_hash(path)

                if digest in hashes:
                    duplicates.append(
                        (path, hashes[digest])
                    )
                else:
                    hashes[digest] = path

    if duplicates:
        print(f"WARNING: {len(duplicates)} duplicate image pairs found.")

        for current, previous in duplicates[:20]:
            print(f"  {current}")
            print(f"  == {previous}")

        if len(duplicates) > 20:
            print(f"  ... and {len(duplicates) - 20} more")

        print()
        print(
            "Duplicate images can cause leakage if the duplicates "
            "are located in different splits."
        )

        return False

    print("No exact duplicate images found: OK")
    print()

    return True


def validate_manifest() -> bool:
    print("=" * 70)
    print("MANIFEST / SOURCE SPLIT CHECK")
    print("=" * 70)

    manifest_path = DATA_DIR / "manifest.json"

    if not manifest_path.exists():
        print("WARNING: manifest.json not found.")
        print("Cannot independently verify source-level splitting.")
        print()

        return False

    try:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        print(f"ERROR: Could not read manifest.json: {exc}")
        return False

    if manifest.get("split_policy") != "by_source_cover_image":
        print(
            "ERROR: Manifest does not declare "
            "'by_source_cover_image' splitting."
        )
        return False

    source_to_split = {}

    for file_info in manifest.get("files", []):
        source_id = file_info.get("source_id")
        split = file_info.get("split")

        if not source_id or not split:
            continue

        previous = source_to_split.get(source_id)

        if previous is not None and previous != split:
            print(
                "ERROR: SOURCE LEAKAGE DETECTED!"
            )
            print(
                f"Source '{source_id}' appears in both "
                f"'{previous}' and '{split}'."
            )

            return False

        source_to_split[source_id] = split

    print(
        f"Unique source images checked: {len(source_to_split)}"
    )

    print("Source-level split policy: OK")
    print("No source ID appears across multiple splits: OK")
    print()

    return True


def validate_class_balance(counts: dict) -> bool:
    print("=" * 70)
    print("CLASS BALANCE CHECK")
    print("=" * 70)

    valid = True

    for split in EXPECTED_SPLITS:
        cover = counts[split]["cover"]
        stego = counts[split]["stego"]

        total = cover + stego

        if total == 0:
            print(f"ERROR: {split} split is empty.")
            valid = False
            continue

        cover_ratio = cover / total
        stego_ratio = stego / total

        print(
            f"{split:5s} | "
            f"Cover: {cover:6d} ({cover_ratio:.2%}) | "
            f"Stego: {stego:6d} ({stego_ratio:.2%})"
        )

    print()

    return valid


def main() -> None:
    structure_ok = validate_structure()

    if not structure_ok:
        print("=" * 70)
        print("VALIDATION FAILED")
        print("=" * 70)
        return

    counts = validate_counts()

    readability_ok = validate_readability()

    duplicate_ok = validate_duplicates()

    manifest_ok = validate_manifest()

    balance_ok = validate_class_balance(counts)

    print("=" * 70)
    print("FINAL VALIDATION RESULT")
    print("=" * 70)

    print(
        f"Directory structure : "
        f"{'PASS' if structure_ok else 'FAIL'}"
    )

    print(
        f"Image readability   : "
        f"{'PASS' if readability_ok else 'FAIL'}"
    )

    print(
        f"Duplicate check     : "
        f"{'PASS' if duplicate_ok else 'WARNING'}"
    )

    print(
        f"Source split check  : "
        f"{'PASS' if manifest_ok else 'WARNING'}"
    )

    print(
        f"Class distribution  : "
        f"{'PASS' if balance_ok else 'FAIL'}"
    )

    print()

    if (
        structure_ok
        and readability_ok
        and duplicate_ok
        and manifest_ok
        and balance_ok
    ):
        print("DATASET VALIDATION: PASS")
        print("The dataset is ready for the next stage.")
    else:
        print("DATASET VALIDATION: REVIEW REQUIRED")
        print("Do NOT train the final model yet.")

    print("=" * 70)


if __name__ == "__main__":
    main()