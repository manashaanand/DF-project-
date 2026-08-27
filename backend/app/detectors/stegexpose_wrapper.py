"""
StegExpose JAR wrapper — supplementary statistical steganalysis only.

StegExpose is NOT the primary AI classifier. It provides an optional LSB-oriented
statistical score that is fused with the CNN prediction when available.

Requires Java and the StegExpose JAR (see third_party/stegexpose/README.md).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class StegExposeResult:
    available: bool
    score: float | None  # normalized stego probability in [0, 1]
    raw_output: str | None = None
    error: str | None = None


class StegExposeWrapper:
    def __init__(self, jar_path: Path | None, java_binary: str = "java"):
        self.jar_path = Path(jar_path) if jar_path else None
        self.java_binary = java_binary

    def is_available(self) -> bool:
        if self.jar_path is None or not self.jar_path.is_file():
            return False
        return shutil.which(self.java_binary) is not None

    def analyze_file(self, image_path: Path) -> StegExposeResult:
        if not self.is_available():
            return StegExposeResult(
                available=False,
                score=None,
                error="StegExpose JAR or Java runtime not configured",
            )

        assert self.jar_path is not None
        try:
            with tempfile.TemporaryDirectory(prefix="stegexpose_") as tmp:
                tmp_dir = Path(tmp)
                # StegExpose expects a directory of images
                link_path = tmp_dir / image_path.name
                shutil.copy2(image_path, link_path)
                cmd = [
                    self.java_binary,
                    "-jar",
                    str(self.jar_path),
                    str(tmp_dir),
                ]
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
                output = (proc.stdout or "") + "\n" + (proc.stderr or "")
                if proc.returncode != 0:
                    logger.warning("StegExpose exited with code %s: %s", proc.returncode, output[:500])
                    return StegExposeResult(
                        available=True,
                        score=None,
                        raw_output=output,
                        error=f"StegExpose process failed (exit {proc.returncode})",
                    )
                score = self._parse_score(output)
                return StegExposeResult(available=True, score=score, raw_output=output)
        except subprocess.TimeoutExpired:
            return StegExposeResult(available=True, score=None, error="StegExpose timed out")
        except OSError as exc:
            return StegExposeResult(available=True, score=None, error=str(exc))

    @staticmethod
    def _parse_score(output: str) -> float | None:
        """
        Parse StegExpose output for a suspicion/score value and normalize to [0, 1].
        Supports common numeric patterns in StegExpose CSV/text output.
        """
        # Look for explicit suspicion/score labels
        patterns = [
            r"(?i)suspicion[:\s]+([0-9]*\.?[0-9]+)",
            r"(?i)score[:\s]+([0-9]*\.?[0-9]+)",
            r"(?i)probability[:\s]+([0-9]*\.?[0-9]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, output)
            if match:
                value = float(match.group(1))
                return float(max(0.0, min(1.0, value)))

        # Fallback: last float on a line with the image name or highest suspicion-like value
        floats = [float(x) for x in re.findall(r"\b0\.\d+\b|\b1\.0+\b", output)]
        if floats:
            return float(max(0.0, min(1.0, max(floats))))

        return None
