"""
PyCaret benchmark for the 99-feature steganography detector.

IMPORTANT:
- Uses the existing 99-feature extractor.
- TRAIN + VAL are used for model selection.
- TEST is NEVER used here.
- Does not overwrite the existing classical model.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"

sys.path.insert(0, str(BACKEND))

from ml.features.stego_features import extract_features_from_file


DATASET = ROOT / "data" / "processed" / "images"
OUTPUT = ROOT / "models" / "pycaret_benchmark"

OUTPUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# COLLECT IMAGES
# ---------------------------------------------------------------------

def collect_split(split: str):
    cover_dir = DATASET / "cover" / split
    stego_dir = DATASET / "stego" / split

    cover = sorted(cover_dir.glob("*.png"))
    stego = sorted(stego_dir.glob("*.png"))

    files = [(p, 0) for p in cover]
    files.extend((p, 1) for p in stego)

    return files


# ---------------------------------------------------------------------
# EXTRACT 99 FEATURES
# ---------------------------------------------------------------------

def extract_dataset(files, name: str):

    print()
    print("=" * 70)
    print(f"EXTRACTING FEATURES: {name}")
    print("=" * 70)

    print(f"Samples: {len(files)}")

    X = []
    y = []

    start = time.time()

    for index, (path, label) in enumerate(files, start=1):

        features = extract_features_from_file(path)

        if features.shape != (294,):
            raise RuntimeError(
                f"Unexpected feature shape for {path}: "
                f"{features.shape}. Expected (294,)."
            )

        if not np.isfinite(features).all():
            raise RuntimeError(
                f"Non-finite feature values found in: {path}"
            )

        X.append(features)
        y.append(label)

        if index % 500 == 0 or index == len(files):
            elapsed = time.time() - start
            print(
                f"\rProcessed {index}/{len(files)} "
                f"({elapsed:.1f}s)",
                end="",
                flush=True,
            )

    print()

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)

    print(f"Feature matrix: {X.shape}")
    print(f"Cover         : {np.sum(y == 0)}")
    print(f"Stego         : {np.sum(y == 1)}")

    return X, y


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    print("=" * 70)
    print("PYCARET CLASSICAL MODEL BENCHMARK")
    print("=" * 70)

    # -------------------------------------------------------------
    # LOAD TRAIN
    # -------------------------------------------------------------

    train_files = collect_split("train")
    val_files = collect_split("val")

    print()
    print(f"Train images: {len(train_files)}")
    print(f"Validation images: {len(val_files)}")

    # -------------------------------------------------------------
    # EXTRACT TRAIN
    # -------------------------------------------------------------

    X_train, y_train = extract_dataset(
        train_files,
        "TRAIN"
    )

    # -------------------------------------------------------------
    # EXTRACT VALIDATION
    # -------------------------------------------------------------

    X_val, y_val = extract_dataset(
        val_files,
        "VALIDATION"
    )

    # -------------------------------------------------------------
    # CREATE DATAFRAMES
    # -------------------------------------------------------------

    feature_names = [
        f"feature_{i:03d}"
        for i in range(294)
    ]

    train_df = pd.DataFrame(
        X_train,
        columns=feature_names
    )

    train_df["target"] = y_train

    val_df = pd.DataFrame(
        X_val,
        columns=feature_names
    )

    val_df["target"] = y_val

    # -------------------------------------------------------------
    # COMBINE TRAIN + VALIDATION
    #
    # PyCaret will create its own internal training/CV split.
    # TEST IS NOT TOUCHED.
    # -------------------------------------------------------------

    development_df = pd.concat(
        [train_df, val_df],
        ignore_index=True
    )

    development_df["target"] = (
        development_df["target"]
        .astype(int)
        .astype(str)
    )

    csv_path = OUTPUT / "pycaret_development_features.csv"

    development_df.to_csv(
        csv_path,
        index=False
    )

    print()
    print("=" * 70)
    print("DEVELOPMENT DATASET")
    print("=" * 70)

    print(f"Shape: {development_df.shape}")
    print(f"Saved: {csv_path}")

    print()
    print("Class distribution:")
    print(
        development_df["target"]
        .value_counts()
        .sort_index()
    )

    # -------------------------------------------------------------
    # PYCARET
    # -------------------------------------------------------------

    from pycaret.classification import (
        setup,
        compare_models,
        pull,
    )

    print()
    print("=" * 70)
    print("STARTING PYCARET")
    print("=" * 70)

    setup(
        data=development_df,
        target="target",
        session_id=42,

        # Cross-validation
        fold=5,

        # Metrics
        normalize=True,

        # Avoid unnecessary UI
        verbose=False,

        # Prevent test leakage
        data_split_stratify=True,
    )

    # -------------------------------------------------------------
    # COMPARE MODELS
    # -------------------------------------------------------------

    print()
    print("=" * 70)
    print("COMPARING CLASSIFIERS")
    print("=" * 70)

    best_model = compare_models(
        sort="AUC",
        n_select=10,
        turbo=False,
    )

    results = pull()

    print()
    print("=" * 70)
    print("PYCARET RESULTS")
    print("=" * 70)

    print(results.to_string(index=False))

    results_path = OUTPUT / "pycaret_model_comparison.csv"

    results.to_csv(
        results_path,
        index=False
    )

    print()
    print(f"Results saved to:")
    print(results_path)

    print()
    print("=" * 70)
    print("BEST MODEL")
    print("=" * 70)

    print(best_model)

    print()
    print("Benchmark completed successfully.")


if __name__ == "__main__":
    main()
