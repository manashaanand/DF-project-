"""
IMPROVED 294-FEATURE STEGANALYSIS DATASET EXTRACTION

Creates a NEW feature cache using the improved stego_features.py.

Existing 99-feature dataset is NOT modified.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd


# =====================================================================
# PROJECT ROOT / IMPORT PATH
# =====================================================================

ROOT = Path(__file__).resolve().parents[2]

# Make both the project root and backend available to Python.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BACKEND = ROOT / "backend"

if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


from ml.features.stego_features import (
    extract_features_from_file,
)


# =====================================================================
# PATHS
# =====================================================================

IMAGE_ROOT = (
    ROOT
    / "data"
    / "processed"
    / "images"
)

OUTPUT_DIR = (
    ROOT
    / "models"
    / "pycaret_benchmark"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "pycaret_development_features_294.csv"
)


# =====================================================================
# SETTINGS
# =====================================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}

WORKERS = 6

PRINT_EVERY = 500


# =====================================================================
# FIND IMAGES
# =====================================================================

def get_images(directory: Path) -> list[Path]:

    if not directory.exists():
        raise FileNotFoundError(
            f"Directory not found:\n{directory}"
        )

    files = [
        path
        for path in directory.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    ]

    return sorted(files)


# =====================================================================
# PROCESS ONE IMAGE
# =====================================================================

def process_image(item):

    path, label = item

    features = extract_features_from_file(
        path
    )

    return features, label


# =====================================================================
# EXTRACT GROUP
# =====================================================================

def extract_group(
    name: str,
    directory: Path,
    label: int,
):

    images = get_images(directory)

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(
        f"Images: {len(images)}"
    )

    if not images:
        raise RuntimeError(
            f"No images found:\n{directory}"
        )

    items = [
        (path, label)
        for path in images
    ]

    features = []
    labels = []

    start = time.time()

    # Prevent OpenCV from creating too many
    # additional internal threads.
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    with ThreadPoolExecutor(
        max_workers=WORKERS
    ) as executor:

        for index, (
            feature_vector,
            image_label,
        ) in enumerate(
            executor.map(
                process_image,
                items,
            ),
            start=1,
        ):

            features.append(
                feature_vector
            )

            labels.append(
                image_label
            )

            if (
                index % PRINT_EVERY == 0
                or index == len(items)
            ):

                elapsed = (
                    time.time()
                    - start
                )

                rate = (
                    index / elapsed
                    if elapsed > 0
                    else 0
                )

                remaining = (
                    len(items)
                    - index
                )

                eta = (
                    remaining / rate
                    if rate > 0
                    else 0
                )

                print(
                    f"Processed "
                    f"{index}/{len(items)} "
                    f"| {rate:.2f} img/s "
                    f"| ETA {eta / 60:.1f} min"
                )

    X = np.asarray(
        features,
        dtype=np.float32,
    )

    y = np.asarray(
        labels,
        dtype=np.int32,
    )

    return X, y


# =====================================================================
# MAIN
# =====================================================================

def main():

    print("=" * 70)
    print(
        "IMPROVED 294-FEATURE EXTRACTION"
    )
    print("=" * 70)

    print()
    print(
        f"Project root:"
    )
    print(ROOT)

    print()
    print(
        f"Image root:"
    )
    print(IMAGE_ROOT)

    print()
    print(
        f"Output:"
    )
    print(OUTPUT_CSV)

    print()
    print(
        "Existing 99-feature cache will NOT be modified."
    )

    # ---------------------------------------------------------------
    # DIRECTORIES
    # ---------------------------------------------------------------

    cover_train = (
        IMAGE_ROOT
        / "cover"
        / "train"
    )

    stego_train = (
        IMAGE_ROOT
        / "stego"
        / "train"
    )

    cover_val = (
        IMAGE_ROOT
        / "cover"
        / "val"
    )

    stego_val = (
        IMAGE_ROOT
        / "stego"
        / "val"
    )

    directories = [
        cover_train,
        stego_train,
        cover_val,
        stego_val,
    ]

    # ---------------------------------------------------------------
    # CHECK DATASET
    # ---------------------------------------------------------------

    print()
    print(
        "Checking dataset structure..."
    )

    for directory in directories:

        print(
            f"  {directory}"
        )

        if not directory.exists():

            raise FileNotFoundError(
                f"\nExpected directory does not exist:\n"
                f"{directory}"
            )

    # ---------------------------------------------------------------
    # TRAIN COVER
    # ---------------------------------------------------------------

    X_cover_train, y_cover_train = (
        extract_group(
            "TRAIN COVER",
            cover_train,
            0,
        )
    )

    # ---------------------------------------------------------------
    # TRAIN STEGO
    # ---------------------------------------------------------------

    X_stego_train, y_stego_train = (
        extract_group(
            "TRAIN STEGO",
            stego_train,
            1,
        )
    )

    # ---------------------------------------------------------------
    # VALIDATION COVER
    # ---------------------------------------------------------------

    X_cover_val, y_cover_val = (
        extract_group(
            "VALIDATION COVER",
            cover_val,
            0,
        )
    )

    # ---------------------------------------------------------------
    # VALIDATION STEGO
    # ---------------------------------------------------------------

    X_stego_val, y_stego_val = (
        extract_group(
            "VALIDATION STEGO",
            stego_val,
            1,
        )
    )

    # ---------------------------------------------------------------
    # COMBINE TRAIN
    # ---------------------------------------------------------------

    X_train = np.concatenate(
        [
            X_cover_train,
            X_stego_train,
        ],
        axis=0,
    )

    y_train = np.concatenate(
        [
            y_cover_train,
            y_stego_train,
        ],
        axis=0,
    )

    # ---------------------------------------------------------------
    # COMBINE VALIDATION
    # ---------------------------------------------------------------

    X_val = np.concatenate(
        [
            X_cover_val,
            X_stego_val,
        ],
        axis=0,
    )

    y_val = np.concatenate(
        [
            y_cover_val,
            y_stego_val,
        ],
        axis=0,
    )

    # ---------------------------------------------------------------
    # DEVELOPMENT DATASET
    # ---------------------------------------------------------------

    X = np.concatenate(
        [
            X_train,
            X_val,
        ],
        axis=0,
    )

    y = np.concatenate(
        [
            y_train,
            y_val,
        ],
        axis=0,
    )

    # ---------------------------------------------------------------
    # VERIFY
    # ---------------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "FEATURE EXTRACTION COMPLETE"
    )
    print("=" * 70)

    print(
        f"Train samples      : {len(y_train)}"
    )

    print(
        f"Validation samples : {len(y_val)}"
    )

    print(
        f"Total samples      : {len(y)}"
    )

    print(
        f"Feature count      : {X.shape[1]}"
    )

    print(
        f"Train cover/stego  : "
        f"{np.sum(y_train == 0)} / "
        f"{np.sum(y_train == 1)}"
    )

    print(
        f"Val cover/stego    : "
        f"{np.sum(y_val == 0)} / "
        f"{np.sum(y_val == 1)}"
    )

    if X.shape[1] != 294:

        raise RuntimeError(
            f"Expected 294 features, "
            f"got {X.shape[1]}"
        )

    if not np.isfinite(X).all():

        raise RuntimeError(
            "Feature matrix contains "
            "NaN or infinite values."
        )

    # ---------------------------------------------------------------
    # CREATE DATAFRAME
    # ---------------------------------------------------------------

    columns = [
        f"feature_{i:03d}"
        for i in range(
            X.shape[1]
        )
    ]

    df = pd.DataFrame(
        X,
        columns=columns,
    )

    df["target"] = y

    # ---------------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------------

    print()
    print(
        "Saving 294-feature dataset..."
    )

    df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print()
    print("=" * 70)
    print(
        "294-FEATURE DATASET SAVED"
    )
    print("=" * 70)

    print(
        f"File: {OUTPUT_CSV}"
    )

    print(
        f"Shape: {df.shape}"
    )

    print()
    print(
        "Class distribution:"
    )

    print(
        df["target"].value_counts()
    )

    print()
    print(
        "Existing 99-feature dataset remains untouched."
    )


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()