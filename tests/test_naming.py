"""No version or development-stage label may appear anywhere in the repository."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from ssl_silence import config as cfg

REPO = Path(__file__).resolve().parents[1]

TEXT_SUFFIXES = {".py", ".md", ".txt", ".toml", ".yaml", ".yml", ".cff", ".csv", ".ipynb",
                 ".tex", ".json", ".cfg", ".ini", ""}

# A "v" directly followed by digits that is not part of a longer word or number.
VERSION_LABEL = re.compile(r"(?<![A-Za-z0-9])[vV][0-9]+(?![A-Za-z0-9])")
STAGE_LABEL = re.compile(r"(?i)\bphase[ _-]?\d")
NOTEBOOK_NUMBER = re.compile(r"(?i)\b(?:notebooks?|nb)[ _-]?\d")
SCRATCH_NAME = re.compile("(?i)unti" + r"tled\d*")
PATTERNS = {"version label": VERSION_LABEL, "stage label": STAGE_LABEL,
            "numbered notebook reference": NOTEBOOK_NUMBER, "scratch name": SCRATCH_NAME}

# A notebook of this repository is referred to by its full file name only,
# and only if the name is in the planned list (config.NOTEBOOKS) or the file
# exists. A bare stem such as the one in "notebooks/<stem>.ipynb" is not
# allowed on its own.
NOTEBOOK_FILE = re.compile(r"notebooks/[0-9A-Za-z_]+\.ipynb")

# Files whose content is exempt from the rule. The only entry is the list of
# function names of the original analysis code that the parity tests import;
# those names must be written literally to call the original functions. The
# path of an exempt file is still checked.
ALLOWED_FILES = frozenset({"tests/original_names.py"})


def problems_in(text: str) -> list[str]:
    """Human-readable problems found in ``text``."""
    found = []
    for name, pattern in PATTERNS.items():
        for match in pattern.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            found.append(f"{name} {match.group(0)!r} at line {line}")
    return found


def notebook_reference_problems(text: str, known=cfg.NOTEBOOKS) -> list[str]:
    """Notebook references of ``text`` that are not full, known file names."""
    found = []
    for match in NOTEBOOK_FILE.finditer(text):
        name = match.group(0)
        if name not in known and not (REPO / name).is_file():
            line = text.count("\n", 0, match.start()) + 1
            found.append(f"unknown notebook {name!r} at line {line}")
    remainder = NOTEBOOK_FILE.sub("", text)
    for name in known:
        stem = Path(name).stem
        for match in re.finditer(re.escape(stem), remainder):
            found.append(f"notebook {stem!r} is not written as its full file name")
    return found


def repository_files() -> list[Path]:
    """Tracked and untracked (not ignored) files of the repository."""
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                         cwd=REPO, capture_output=True, text=True, check=True).stdout
    return [REPO / line for line in out.splitlines() if line and (REPO / line).is_file()]


def notebook_text(path: Path) -> str:
    """Cell sources and outputs of a notebook as one string."""
    nb = json.loads(path.read_text(encoding="utf-8"))
    parts = []
    for cell in nb.get("cells", []):
        parts.append("".join(cell.get("source", [])))
        parts.append(json.dumps(cell.get("outputs", [])))
    return "\n".join(parts)


@pytest.mark.parametrize("bad", ["results_" + "v" + "3/x.csv", "the " + "v" + "2 pipeline",
                                 "V" + "1 cache", "pha" + "se 6 tables", "Pha" + "se9",
                                 "note" + "book 14", "n" + "b10", "Unti" + "tled24"])
def test_patterns_flag_labels(bad):
    assert problems_in(bad)


@pytest.mark.parametrize("good", ["wav2vec 2.0", "conv1d", "L4 stats", "trim40", "hubert-base",
                                  "0a266cbbd2271988becb6b6b86db5fcdb05bd3929207988692527f410610ba72",
                                  "transformers==5.17.0", "emphasis 3", "notebooks/", "A07"])
def test_patterns_accept_plain_names(good):
    assert not problems_in(good)


def test_repository_has_no_version_labels():
    problems = []
    for path in repository_files():
        rel = path.relative_to(REPO).as_posix()
        problems += [f"{rel}: path has a {p}" for p in problems_in(rel)]
        if rel in ALLOWED_FILES or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = notebook_text(path) if path.suffix == ".ipynb" else path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        problems += [f"{rel}: {p}" for p in problems_in(text)]
        problems += [f"{rel}: {p}" for p in notebook_reference_problems(text)]
    assert not problems, "\n".join(problems)


def test_planned_notebooks_are_numbered_in_order():
    assert len(cfg.NOTEBOOKS) == len(set(cfg.NOTEBOOKS))
    for position, name in enumerate(cfg.NOTEBOOKS, start=1):
        assert NOTEBOOK_FILE.fullmatch(name)
        assert Path(name).name.startswith(f"{position:02d}_")


@pytest.mark.parametrize("name", cfg.NOTEBOOKS)
def test_full_file_name_of_a_planned_notebook_is_allowed(name):
    text = f"run {name} on Colab; see `{name}`."
    assert not problems_in(text) and not notebook_reference_problems(text)


def test_notebook_reference_by_number_stays_forbidden():
    assert problems_in("run note" + "book 01 on Colab")
    assert problems_in("see N" + "B01")


def test_unknown_or_bare_notebook_names_are_flagged():
    unknown = "notebooks/" + "10_something_else" + ".ipynb"
    assert notebook_reference_problems(f"run {unknown}")
    stem = Path(cfg.NOTEBOOKS[0]).stem
    assert notebook_reference_problems("run " + stem + " first")
    assert notebook_reference_problems("`" + stem + ".ipynb`")
    assert not notebook_reference_problems("the notebooks/ folder and config.NOTEBOOKS")


def test_allow_list_is_the_original_names_file_only():
    assert ALLOWED_FILES == {"tests/original_names.py"}
    path = REPO / "tests" / "original_names.py"
    assert path.is_file()
    # The exemption is needed: the file does contain names the rule forbids.
    assert problems_in(path.read_text(encoding="utf-8"))
    # No other test file may spell those names.
    from original_names import ORIGINAL_NAMES
    different = {old for names in ORIGINAL_NAMES.values() for new, old in names.items() if old != new}
    for other in (REPO / "tests").glob("*.py"):
        if other.name != "original_names.py":
            text = other.read_text(encoding="utf-8")
            assert not [name for name in different if name in text], other.name


def test_no_large_files():
    large = [p.relative_to(REPO).as_posix() for p in repository_files()
             if p.stat().st_size > 5 * 1024 * 1024]
    assert not large, large
