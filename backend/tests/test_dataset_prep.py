import json
from pathlib import Path

import pytest


def test_prepare_dataset_no_leakage(tmp_path):
    pytest.importorskip("cv2")
    import sys

    scripts_dir = Path(__file__).resolve().parents[2] / "scripts" / "datasets"
    sys.path.insert(0, str(scripts_dir))
    from prepare_image_dataset import prepare_dataset  # noqa: E402

    output_dir = tmp_path / "processed"
    manifest = prepare_dataset(
        covers_dir=None,
        output_dir=output_dir,
        payload_rates=[0.3],
        train_ratio=0.6,
        val_ratio=0.2,
        seed=7,
        synthetic_count=10,
        synthetic_size=64,
    )

    train_ids = set(manifest["splits"]["train"])
    val_ids = set(manifest["splits"]["val"])
    test_ids = set(manifest["splits"]["test"])

    assert len(train_ids & val_ids) == 0
    assert len(train_ids & test_ids) == 0
    assert len(val_ids & test_ids) == 0
    assert len(train_ids) + len(val_ids) + len(test_ids) == 10

    # Each source cover and its stego variants share one split
    for entry in manifest["files"]:
        source_id = entry["source_id"]
        split = entry["split"]
        assert source_id in manifest["splits"][split]

    assert (output_dir / "manifest.json").is_file()
