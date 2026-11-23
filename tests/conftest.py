"""Shared fixtures: access to the original analysis code and the parity record.

The parity tests compare this package with the original analysis code on the
same inputs. They run only when the environment variable
SSL_SILENCE_ORIGINAL_SRC points to the ``src`` folder of that code, which is
read and never modified. Their outcome is written to
results/checks/parity_original_code.csv.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

from original_names import ORIGINAL_PACKAGE, original_name

REPO = Path(__file__).resolve().parents[1]
PARITY_FILE = REPO / "results" / "checks" / "parity_original_code.csv"
PARITY_COLUMNS = ["module", "function", "n_cases", "max_abs_diff", "identical"]
ORIGINAL_SRC = os.environ.get("SSL_SILENCE_ORIGINAL_SRC")

_records: list[dict] = []


def pytest_addoption(parser):
    parser.addoption("--manuscript", default=None,
                     help="manuscript .tex file: the rebuilt tables and the in-text numbers are "
                          "compared with it")
    parser.addoption("--tables", default=str(REPO / "results"),
                     help="folder of the result tables that the paper modules read")


@pytest.fixture(scope="session")
def manuscript(request) -> str:
    """The manuscript file given with --manuscript; the test is skipped without it."""
    path = request.config.getoption("--manuscript")
    if not path:
        pytest.skip("pass --manuscript <file.tex> (and --tables <folder>) to compare with the manuscript")
    return path


@pytest.fixture(scope="session")
def result_tables(request) -> str:
    """The folder given with --tables (default: results/ of the repository)."""
    return request.config.getoption("--tables")


@pytest.fixture(scope="session")
def original_src() -> Path:
    """Folder of the original analysis code; the test is skipped without it."""
    if not ORIGINAL_SRC:
        pytest.skip("SSL_SILENCE_ORIGINAL_SRC is not set")
    path = Path(ORIGINAL_SRC)
    if not (path / ORIGINAL_PACKAGE).is_dir():
        pytest.skip(f"no {ORIGINAL_PACKAGE} package under SSL_SILENCE_ORIGINAL_SRC")
    return path


@pytest.fixture(scope="session")
def original(original_src):
    """Importer of a module of the original analysis code, e.g. original("metrics")."""
    if str(original_src) not in sys.path:
        sys.path.insert(0, str(original_src))

    def load(name: str):
        return importlib.import_module(f"{ORIGINAL_PACKAGE}.{name}")

    return load


@pytest.fixture(scope="session")
def original_function(original):
    """Function of the original code by this package's name, e.g. ("metrics", "eer")."""
    def get(module: str, name: str):
        return getattr(original(module), original_name(module, name))

    return get


@pytest.fixture(scope="session")
def parity_record():
    """Function that records one compared function: (module, function, n_cases, max_abs_diff)."""
    def record(module: str, function: str, n_cases: int, max_abs_diff: float) -> None:
        _records.append({"module": module, "function": function, "n_cases": int(n_cases),
                         "max_abs_diff": float(max_abs_diff),
                         "identical": bool(max_abs_diff == 0.0)})
    return record


def pytest_sessionfinish(session, exitstatus):
    """Write the parity record; rows of modules compared in this session replace old ones.

    Nothing is written in a session without the original code.
    """
    if not _records or not ORIGINAL_SRC:
        return
    new = pd.DataFrame(_records, columns=PARITY_COLUMNS)
    if PARITY_FILE.exists():
        old = pd.read_csv(PARITY_FILE)
        new = pd.concat([old[~old["module"].isin(new["module"].unique())], new],
                        ignore_index=True)
    new = new.sort_values(["module", "function"], kind="mergesort").reset_index(drop=True)
    PARITY_FILE.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(PARITY_FILE, index=False, lineterminator="\n")
