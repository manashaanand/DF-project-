"""Train the image CNN on prepared cover/stego residual maps."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from _training_utils import (
    BACKEND_ROOT,
    build_arrays,
    ensure_backend_on_path,
    load_split_files,
)

ensure_backend_on_path()

from ml.features.image_features import (  # noqa: E402
    DEFAULT_INPUT_SIZE,
    save_preprocessing_config,
)
from ml.models.image_cnn import build_image_cnn  # noqa: E402


def calculate_class_weights(y_train: np.ndarray) -> dict[int, float]:
    """Calculate balanced class weights for binary classification.

    Class 0 = Cover
    Class 1 = Stego
    """

    labels = y_train.astype(np.int32)

    class_counts = np.bincount(labels, minlength=2)
    total = len(labels)
    num_classes = 2

    if class_counts[0] == 0 or class_counts[1] == 0:
        raise ValueError(
            f"Both classes must be present in the training data. "
            f"Cover={class_counts[0]}, Stego={class_counts[1]}"
        )

    class_weight = {
        0: total / (num_classes * class_counts[0]),
        1: total / (num_classes * class_counts[1]),
    }

    print()
    print("=" * 60)
    print("CLASS DISTRIBUTION")
    print("=" * 60)
    print(f"Cover samples : {class_counts[0]}")
    print(f"Stego samples : {class_counts[1]}")
    print(f"Total samples : {total}")
    print()
    print("CLASS WEIGHTS")
    print("=" * 60)
    print(f"Cover weight  : {class_weight[0]:.6f}")
    print(f"Stego weight  : {class_weight[1]:.6f}")
    print("=" * 60)
    print()

    return class_weight


def train(
    data_dir: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    input_size: tuple[int, int],
) -> dict:
    """Train the CNN and save the best model."""

    print("=" * 60)
    print("AI MULTIMEDIA STEGANOGRAPHY DETECTION")
    print("IMAGE CNN TRAINING")
    print("=" * 60)
    print()

    # ---------------------------------------------------------
    # 1. Load train and validation file lists
    # ---------------------------------------------------------

    print("Loading dataset...")

    train_pairs = load_split_files(data_dir, "train")
    val_pairs = load_split_files(data_dir, "val")

    if not train_pairs:
        raise FileNotFoundError(
            f"No training files found under "
            f"{data_dir / 'cover' / 'train'} and "
            f"{data_dir / 'stego' / 'train'}"
        )

    print(f"Training samples   : {len(train_pairs)}")
    print(f"Validation samples : {len(val_pairs)}")
    print()

    # ---------------------------------------------------------
    # 2. Build SRM residual-map arrays
    # ---------------------------------------------------------

    print("Preparing SRM residual-map inputs...")
    print("This may take some time on CPU.")
    print()

    x_train, y_train = build_arrays(
        train_pairs,
        input_size,
    )

    if val_pairs:
        x_val, y_val = build_arrays(
            val_pairs,
            input_size,
        )
    else:
        x_val, y_val = None, None

    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    if x_val is not None:
        print(f"x_val shape  : {x_val.shape}")
        print(f"y_val shape  : {y_val.shape}")

    print()

    # ---------------------------------------------------------
    # 3. Calculate balanced class weights
    # ---------------------------------------------------------

    class_weight = calculate_class_weights(y_train)

    # ---------------------------------------------------------
    # 4. Build CNN
    # ---------------------------------------------------------

    print("Building CNN...")

    model = build_image_cnn(
        input_shape=(*input_size, 1)
    )

    model.summary()

    # ---------------------------------------------------------
    # 5. TensorFlow callbacks
    # ---------------------------------------------------------

    import tensorflow as tf

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = output_dir / "checkpoint.keras"

    callbacks = []

    # Save the model with the best validation AUC.
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_auc" if val_pairs else "auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        )
    )

    # Stop training when validation AUC stops improving.
    callbacks.append(
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc" if val_pairs else "auc",
            mode="max",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        )
    )

    # Reduce learning rate when validation loss stops improving.
    callbacks.append(
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss" if val_pairs else "loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        )
    )

    # ---------------------------------------------------------
    # 6. Training configuration
    # ---------------------------------------------------------

    fit_kwargs: dict = {
        "x": x_train,
        "y": y_train,
        "epochs": epochs,
        "batch_size": batch_size,
        "callbacks": callbacks,
        "verbose": 1,

        # IMPORTANT:
        # Compensates for the 1:3 Cover/Stego imbalance.
        "class_weight": class_weight,
    }

    if val_pairs:
        fit_kwargs["validation_data"] = (
            x_val,
            y_val,
        )

    # ---------------------------------------------------------
    # 7. Train
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    print()

    history = model.fit(
        **fit_kwargs
    )

    # ---------------------------------------------------------
    # 8. Save final model
    # ---------------------------------------------------------

    model_path = output_dir / "model.keras"

    model.save(
        model_path
    )

    # Save preprocessing configuration.
    preprocessing_path = (
        output_dir / "preprocessing_config.json"
    )

    save_preprocessing_config(
        preprocessing_path,
        size=input_size,
    )

    # ---------------------------------------------------------
    # 9. Save training metadata
    # ---------------------------------------------------------

    metadata = {
        "model_version": "image_cnn_v2_balanced",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir),

        "train_samples": int(len(train_pairs)),
        "val_samples": int(len(val_pairs)),

        "input_size": list(input_size),

        "epochs_requested": int(epochs),
        "batch_size": int(batch_size),

        "class_counts": {
            "cover": int(np.sum(y_train == 0)),
            "stego": int(np.sum(y_train == 1)),
        },

        "class_weights": {
            "cover": float(class_weight[0]),
            "stego": float(class_weight[1]),
        },

        "final_loss": float(
            history.history["loss"][-1]
        ),

        "final_accuracy": float(
            history.history["accuracy"][-1]
        ),

        "final_auc": float(
            history.history["auc"][-1]
        ),
    }

    if val_pairs and "val_auc" in history.history:
        metadata["final_val_auc"] = float(
            history.history["val_auc"][-1]
        )

    if val_pairs and "val_accuracy" in history.history:
        metadata["final_val_accuracy"] = float(
            history.history["val_accuracy"][-1]
        )

    metadata_path = (
        output_dir / "metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # 10. Training summary
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(f"Model saved       : {model_path}")
    print(f"Checkpoint saved  : {checkpoint_path}")
    print(f"Metadata saved    : {metadata_path}")
    print(
        f"Preprocessing     : {preprocessing_path}"
    )

    print()
    print(
        f"Final train AUC   : "
        f"{metadata['final_auc']:.4f}"
    )

    if "final_val_auc" in metadata:
        print(
            f"Final validation AUC : "
            f"{metadata['final_val_auc']:.4f}"
        )

    print("=" * 60)
    print()

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train balanced image CNN "
            "for steganography detection"
        )
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(
            "data/processed/images"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "models/image_cnn_balanced"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--input-size",
        type=int,
        default=DEFAULT_INPUT_SIZE[0],
    )

    args = parser.parse_args()

    size = (
        args.input_size,
        args.input_size,
    )

    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        input_size=size,
    )


if __name__ == "__main__":
    main()