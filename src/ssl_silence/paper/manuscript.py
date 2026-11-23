"""Reading the manuscript source and comparing its tables with rebuilt ones.

``compare_tables`` rebuilds every table from the saved CSV files
(``paper.tables``) and compares its data cells number by number with the
``tabular`` of the same LaTeX label in a manuscript ``.tex`` file. Label
cells are skipped; ``$-$`` is read as a minus sign; the argument of a macro
such as ``\\textbf{...}`` is kept; a letter followed by digits (L4) is a
label, not a number.
"""

from __future__ import annotations

import re
from typing import Sequence

import pandas as pd

from . import tables as T

NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?")
REPORT_COLUMNS = ["table", "row", "row_label", "number", "built", "manuscript", "problem"]


def read_tex(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def strip_comments(tex: str) -> str:
    """The source without LaTeX comments (an escaped percent sign is kept)."""
    return re.sub(r"(?<!\\)%.*", "", tex)


def numbers_in(cells: Sequence[str]) -> list[float]:
    """The numbers of the data cells of a table row."""
    text = " ".join(cells).replace("{,}", "")                      # thousands separator
    text = text.replace("$-$", " -").replace("$+$", " +").replace("--", " ").replace("\\%", " ")
    text = re.sub(r"\\cmidrule(\([^)]*\))?\{[^}]*\}", " ", text)
    text = re.sub(r"\\[A-Za-z]+", " ", text)
    text = re.sub(r"[${}~]", " ", text)
    text = re.sub(r"(?<![\w.])[A-Za-z]+\d+(?:\.\d+)?\b", " ", text)
    return [float(m) for m in NUMBER.findall(text)]


def split_cells(row: str) -> list[str]:
    """The cells of a table row (an escaped ampersand does not split)."""
    return [cell.strip() for cell in re.split(r"(?<!\\)&", row)]


def tabular_rows(tex: str, label: str) -> list[str]:
    """The data rows of the tabular that belongs to ``\\label{<label>}``: the
    rows after the first ``\\midrule`` (every row when there is none)."""
    position = tex.find("\\label{%s}" % label)
    if position < 0:
        raise KeyError(f"no \\label{{{label}}} in the manuscript")
    start = tex.rfind("\\begin{table", 0, position)
    end = tex.find("\\end{table", position)
    if start < 0 or end < 0:
        raise ValueError(f"{label}: not inside a table environment")
    found = re.search(r"\\begin\{tabular\*?\}.*?\n(.*?)\\end\{tabular\*?\}", tex[start:end], re.S)
    if not found:
        raise ValueError(f"{label}: no tabular in its table")
    body = strip_comments(found.group(1))
    if "\\midrule" in body:
        body = body.split("\\midrule", 1)[1]
    rows = [row.strip() for row in re.split(r"\\\\", body)]
    rows = [re.sub(r"\\(top|mid|bottom)rule|\\hline|\\cmidrule(\([^)]*\))?\{[^}]*\}", "", row).strip()
            for row in rows]
    return [row for row in rows if row and "&" in row]


def compare_rows(label: str, built: Sequence[Sequence[str]], manuscript_rows: Sequence[str],
                 label_cells: int) -> list[dict]:
    """Differences between the rows of a rebuilt table and the rows of the manuscript.

    Rows without a number are left out on both sides. Returns one record per
    difference: a different number of rows, a different number of numbers in
    a row, or a different number.
    """
    ours = [(row, numbers_in(row[label_cells:])) for row in built]
    theirs = [(row, numbers_in(split_cells(row)[label_cells:])) for row in manuscript_rows]
    ours = [(row, n) for row, n in ours if n]
    theirs = [(row, n) for row, n in theirs if n]
    problems = []

    def add(row, row_label, number, a, b, problem):
        problems.append({"table": label, "row": row, "row_label": row_label, "number": number,
                         "built": a, "manuscript": b, "problem": problem})

    if len(ours) != len(theirs):
        add(0, "", 0, len(ours), len(theirs), "number of rows")
    for j, ((row, a), (_, b)) in enumerate(zip(ours, theirs), start=1):
        row_label = " & ".join(row[:label_cells])
        if len(a) != len(b):
            add(j, row_label, 0, len(a), len(b), "number of numbers in the row")
            continue
        for i, (x, y) in enumerate(zip(a, b), start=1):
            if x != y:
                add(j, row_label, i, x, y, "different number")
    return problems


def compare_tables(tables_dir: str, tex_path: str, labels: Sequence[str] | None = None,
                   encoder: str = "wavlm-base") -> pd.DataFrame:
    """Rebuild the tables and compare them with the manuscript.

    Returns one row per difference (columns REPORT_COLUMNS); an empty table
    means that every number of every compared table is identical. A table
    whose label is missing from the manuscript is one difference.
    """
    tex = read_tex(tex_path)
    problems: list[dict] = []
    for label in (labels or T.TABLES):
        _, built = T.build(label, tables_dir, encoder)
        try:
            manuscript_rows = tabular_rows(tex, label)
        except (KeyError, ValueError) as error:
            problems.append({"table": label, "row": 0, "row_label": "", "number": 0, "built": len(built),
                             "manuscript": float("nan"), "problem": str(error.args[0])})
            continue
        problems += compare_rows(label, built, manuscript_rows, T.TABLES[label]["label_cells"])
    return pd.DataFrame(problems, columns=REPORT_COLUMNS)


def summary(report: pd.DataFrame, labels: Sequence[str] | None = None) -> pd.DataFrame:
    """One row per table: table, n_differences, identical."""
    counts = report.groupby("table").size() if len(report) else pd.Series(dtype=int)
    rows = [{"table": label, "n_differences": int(counts.get(label, 0)),
             "identical": bool(counts.get(label, 0) == 0)} for label in (labels or T.TABLES)]
    return pd.DataFrame(rows)
