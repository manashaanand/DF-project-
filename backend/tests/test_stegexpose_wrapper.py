from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.detectors.stegexpose_wrapper import StegExposeWrapper


def test_unavailable_when_jar_missing(tmp_path):
    wrapper = StegExposeWrapper(jar_path=tmp_path / "missing.jar")
    assert wrapper.is_available() is False
    result = wrapper.analyze_file(tmp_path / "image.png")
    assert result.available is False
    assert result.score is None


def test_parse_score_from_output():
    output = "Image,suspicion\nsample.png,0.73\n"
    score = StegExposeWrapper._parse_score(output)
    assert score == pytest.approx(0.73)


@patch("app.detectors.stegexpose_wrapper.subprocess.run")
@patch("app.detectors.stegexpose_wrapper.shutil.which", return_value="/usr/bin/java")
def test_analyze_file_parses_subprocess_output(mock_which, mock_run, tmp_path):
    jar = tmp_path / "StegExpose.jar"
    jar.write_text("placeholder", encoding="utf-8")
    image = tmp_path / "test.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")

    mock_run.return_value = MagicMock(
        returncode=0,
        stdout="Suspicion: 0.62\n",
        stderr="",
    )

    wrapper = StegExposeWrapper(jar_path=jar)
    result = wrapper.analyze_file(image)
    assert result.available is True
    assert result.score == pytest.approx(0.62)
