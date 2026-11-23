"""The pre-registered files are the files recorded in docs/preregistration.md."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RECORD = REPO / "docs" / "preregistration.md"
FILES = ("docs/analysis_plan_hubert.md", "docs/replication_units.csv", "docs/replication_units_summary.csv")


def recorded() -> dict[str, str]:
    rows = re.findall(r"^\| `([^`]+)` \| `([0-9a-f]{64})` \|$", RECORD.read_text(encoding="utf-8"), flags=re.M)
    return dict(rows)


def sha256_lf(path: Path) -> str:
    """sha256 of a file with LF line endings, whatever the checkout wrote."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def test_the_record_names_the_three_files_and_the_commit():
    assert tuple(recorded()) == FILES
    assert "1ecf7324dfd61369a1b941cc2ce32109784150d6" in RECORD.read_text(encoding="utf-8")


def test_the_plan_and_the_units_are_unchanged_since_the_pre_registration():
    for name, digest in recorded().items():
        assert sha256_lf(REPO / name) == digest, f"{name} differs from the pre-registered file"


def test_the_units_file_has_the_sha256_of_its_meta_file():
    import json

    meta = json.loads((REPO / "docs" / "replication_units.meta.json").read_text(encoding="utf-8"))
    assert meta["sha256"] == recorded()["docs/replication_units.csv"] and meta["rows"] == 830
