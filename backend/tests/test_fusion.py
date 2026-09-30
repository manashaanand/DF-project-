import pytest

from app.detectors.fusion import fuse_scores, label_from_score


@pytest.mark.parametrize(
    "score,expected",
    [
        (0.9, "stego"),
        (0.1, "cover"),
        (0.5, "inconclusive"),
        (0.45, "inconclusive"),
        (0.55, "inconclusive"),
    ],
)
def test_label_from_score(score, expected):
    assert label_from_score(score) == expected


def test_fuse_classical_only():
    fused, label = fuse_scores(classical_score=0.8, supplementary_score=None)
    assert fused == pytest.approx(0.8)
    assert label == "stego"


def test_fuse_weighted_average():
    fused, label = fuse_scores(classical_score=0.8, supplementary_score=0.2, classical_weight=0.7, supplementary_weight=0.3)
    assert fused == pytest.approx(0.7 * 0.8 + 0.3 * 0.2)
    assert label == "stego"


def test_fuse_clamps_to_unit_interval():
    fused, _ = fuse_scores(classical_score=1.5, supplementary_score=1.5)
    assert 0.0 <= fused <= 1.0
