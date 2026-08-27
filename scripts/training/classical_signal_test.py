import cv2
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

root = Path("data/processed/images")

kernel = np.array([
    [0, 0, 0, 0, 0],
    [0, -1, 2, -1, 0],
    [0, 2, -4, 2, 0],
    [0, -1, 2, -1, 0],
    [0, 0, 0, 0, 0],
], dtype=np.float32)


def features(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    residual = cv2.filter2D(
        image.astype(np.float32),
        -1,
        kernel
    )

    return [
        float((image & 1).mean()),
        float(image.std()),
        float(residual.mean()),
        float(residual.std()),
        float(np.mean(np.abs(residual))),
        float(np.percentile(np.abs(residual), 90)),
        float(np.percentile(np.abs(residual), 95)),
        float(np.percentile(np.abs(residual), 99)),
    ]


print("=" * 70)
print("CLASSICAL STEGANOGRAPHY SIGNAL TEST")
print("=" * 70)

cover = sorted((root / "cover" / "test").glob("*.png"))

# Use an equal number of cover/stego samples.
# Keep the test entirely separate from the CNN training.
stego_all = sorted((root / "stego" / "test").glob("*.png"))

# Take up to 1500 each.
cover = cover[:1500]
stego_all = stego_all[:1500]

X_cover = np.array([features(p) for p in cover])
X_stego = np.array([features(p) for p in stego_all])

X = np.vstack([X_cover, X_stego])
y = np.array([0] * len(X_cover) + [1] * len(X_stego))

print("Cover samples :", len(X_cover))
print("Stego samples :", len(X_stego))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y,
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)

model.fit(X_train, y_train)

prob = model.predict_proba(X_test)[:, 1]
pred = (prob >= 0.5).astype(int)

print()
print("RESULT")
print("-" * 70)
print("Accuracy :", accuracy_score(y_test, pred))
print("AUC      :", roc_auc_score(y_test, prob))

print()
print("Feature importance")
print("-" * 70)

names = [
    "LSB balance",
    "Pixel std",
    "SRM mean",
    "SRM std",
    "Mean abs SRM",
    "SRM P90",
    "SRM P95",
    "SRM P99",
]

for name, importance in sorted(
    zip(names, model.feature_importances_),
    key=lambda x: x[1],
    reverse=True
):
    print(f"{name:<20} {importance:.6f}")

print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)
