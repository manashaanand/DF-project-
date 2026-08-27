"""
Confidence fusion for image steganography detection.

Primary signal:
    294-feature Logistic Regression classical steganalysis model.

Supplementary signal:
    StegExpose statistical steganalysis score (optional).

The production model is NOT a CNN.
Therefore all production prediction terminology uses:
    classical_score

Fusion:
    fused =
        classical_weight * classical_score
        +
        stegexpose_weight * supplementary_score

If StegExpose is unavailable:
    fused = classical_score

Decision:
    score >= threshold -> stego
    score < threshold  -> cover

Low-confidence band:
    inconclusive_low <= score <= inconclusive_high
"""

from __future__ import annotations


def label_from_score(
    score: float,
    threshold: float = 0.5,
    inconclusive_low: float = 0.45,
    inconclusive_high: float = 0.55,
) -> str:
    """
    Convert a fused steganography score into a label.
    """

    score = float(score)

    if (
        inconclusive_low <= score <= inconclusive_high
    ):
        return "inconclusive"

    if score >= threshold:
        return "stego"

    return "cover"


def fuse_scores(
    classical_score: float,
    supplementary_score: float | None,
    classical_weight: float = 0.7,
    supplementary_weight: float = 0.3,
    threshold: float = 0.5,
    inconclusive_low: float = 0.45,
    inconclusive_high: float = 0.55,
) -> tuple[float, str]:
    """
    Fuse the production classical model score with
    the optional StegExpose supplementary score.

    Parameters
    ----------
    classical_score:
        Stego probability produced by the 294-feature
        Logistic Regression model.

    supplementary_score:
        Optional StegExpose score.

    classical_weight:
        Weight assigned to the production model.

    supplementary_weight:
        Weight assigned to StegExpose.

    Returns
    -------
    tuple[float, str]
        (fused_score, label)
    """

    classical_score = float(
        max(0.0, min(1.0, classical_score))
    )

    if supplementary_score is None:

        fused = classical_score

    else:

        supplementary_score = float(
            max(0.0, min(1.0, supplementary_score))
        )

        total_weight = (
            classical_weight
            + supplementary_weight
        )

        if total_weight <= 0:
            raise ValueError(
                "Fusion weights must sum to a positive value"
            )

        fused = (
            classical_weight * classical_score
            + supplementary_weight * supplementary_score
        ) / total_weight

    fused = float(
        max(0.0, min(1.0, fused))
    )

    label = label_from_score(
        score=fused,
        threshold=threshold,
        inconclusive_low=inconclusive_low,
        inconclusive_high=inconclusive_high,
    )

    return fused, label