"""The pinned library versions and the installed ones."""

from __future__ import annotations

import os
import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest

REQUIREMENTS = Path(__file__).resolve().parents[1] / "requirements.txt"
# Only the packages pinned in requirements.txt. A version is pinned after it
# has been read from the Colab runtime (torchaudio, torchcodec, scipy and
# soundfile: the first run of notebooks/01_environment_check.ipynb).
EXPECTED = {"torch": "2.11.0", "transformers": "5.17.0", "numpy": "2.1.3",
            "scikit-learn": "1.6.1", "pandas": "2.2.3", "librosa": "0.11.0",
            "torchaudio": "2.11.0", "torchcodec": "0.17.0", "scipy": "1.16.3",
            "soundfile": "0.14.0", "praat-parselmouth": "0.4.7",
            "webrtcvad-wheels": "2.0.14.post1"}
INSTALLED = tuple(EXPECTED)
# torchcodec needs FFmpeg and is not required on a development machine
# (without it every file is read with soundfile); where it is installed, and
# always on Colab (SSL_SILENCE_REQUIRE_REFERENCE=1), it must be the pinned version.
OPTIONAL = ("torchcodec",)


def pins() -> dict[str, str]:
    found = {}
    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([A-Za-z0-9_.\-]+)==(\S+)", line.strip())
        if match:
            found[match.group(1)] = match.group(2)
    return found


def test_requirements_pins():
    assert pins() == EXPECTED


@pytest.mark.skipif(os.environ.get("SSL_SILENCE_SKIP_VERSION_CHECK") == "1",
                    reason="version check disabled")
@pytest.mark.parametrize("package", INSTALLED)
def test_installed_versions(package):
    try:
        installed = version(package)
    except PackageNotFoundError:
        if package in OPTIONAL and os.environ.get("SSL_SILENCE_REQUIRE_REFERENCE") != "1":
            pytest.skip(f"{package} is not installed on this machine")
        raise
    assert installed.split("+")[0] == EXPECTED[package]
