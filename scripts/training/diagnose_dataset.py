import cv2
import numpy as np
from pathlib import Path

root = Path("data/processed/images")

cover = sorted((root / "cover" / "test").glob("*.png"))[:500]
stego = sorted((root / "stego" / "test").glob("*.png"))[:1500]

kernel = np.array([
    [0, 0, 0, 0, 0],
    [0, -1, 2, -1, 0],
    [0, 2, -4, 2, 0],
    [0, -1, 2, -1, 0],
    [0, 0, 0, 0, 0],
], dtype=np.float32)


def get_stats(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Could not read: {path}")

    lsb_balance = float((image & 1).mean())
    pixel_std = float(image.std())

    residual = cv2.filter2D(
        image.astype(np.float32),
        -1,
        kernel
    )

    residual_std = float(residual.std())

    return lsb_balance, pixel_std, residual_std


print("=" * 60)
print("STEGANOGRAPHY DATASET DIAGNOSTIC")
print("=" * 60)

print(f"Cover samples : {len(cover)}")
print(f"Stego samples : {len(stego)}")

cover_stats = np.array([get_stats(p) for p in cover])
stego_stats = np.array([get_stats(p) for p in stego])

print()
print(f"{'Feature':<25}{'Cover mean':>15}{'Stego mean':>15}")
print("-" * 55)

print(
    f"{'LSB balance':<25}"
    f"{cover_stats[:,0].mean():>15.6f}"
    f"{stego_stats[:,0].mean():>15.6f}"
)

print(
    f"{'Pixel std':<25}"
    f"{cover_stats[:,1].mean():>15.6f}"
    f"{stego_stats[:,1].mean():>15.6f}"
)

print(
    f"{'SRM residual std':<25}"
    f"{cover_stats[:,2].mean():>15.6f}"
    f"{stego_stats[:,2].mean():>15.6f}"
)

print()
print("Absolute differences")
print("-" * 55)

print(
    "LSB difference          :",
    abs(cover_stats[:,0].mean() - stego_stats[:,0].mean())
)

print(
    "Pixel std difference    :",
    abs(cover_stats[:,1].mean() - stego_stats[:,1].mean())
)

print(
    "SRM difference          :",
    abs(cover_stats[:,2].mean() - stego_stats[:,2].mean())
)

print("=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
