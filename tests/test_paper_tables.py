"""Table bodies of the manuscript and their comparison with a manuscript source.

The builders are tested on invented tables with values that can be stated
by hand. ``pytest --manuscript <file.tex> --tables <folder>`` additionally
compares every rebuilt table with the real manuscript.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest

import paper_tables as PT
from ssl_silence.paper import manuscript as M
from ssl_silence.paper import tables as T


@pytest.fixture(scope="module")
def folder(tmp_path_factory) -> str:
    return PT.write_tables(str(tmp_path_factory.mktemp("results")))


def rows_of(label: str, folder: str) -> list[list[str]]:
    return T.build(label, folder)[1]


# Formatting ----------------------------------------------------------------------------------

def test_numbers_are_rounded_once_and_written_as_in_the_manuscript():
    assert T.num(1.4849, 2) == "1.48" and T.num(1.485001, 2) == "1.49"
    assert T.num(-0.076, 2) == "$-$0.08" and T.num(-0.004, 2) == "$-$0.00"
    assert T.num(0.11, 2, signed=True) == "+0.11" and T.num(-5.65, 2, signed=True) == "$-$5.65"
    assert T.num(1147, 0) == "1{,}147" and T.num(49.4, 0) == "49" and T.num(0.817, 0, scale=100) == "82"
    assert T.num(float("nan"), 2) == "--" and T.num(None, 2) == "--" and T.num("x", 2) == "--"
    assert T.num(0.0091, 3, fmt="pvalue") == "0.009" and T.num(0.0004988, 3, fmt="pvalue") == "0.0005"
    assert T.num(0.0625, 3, fmt="pvalue") == "0.062" and T.num(4.2e-9, 3, fmt="pvalue") == "4e-09"
    assert T.interval(-6.55, -4.78, 2) == "[$-$6.55, $-$4.78]"
    assert T.interval(0.02, 3.14, 2, signed=True) == "[+0.02, +3.14]"


def test_a_limit_that_rounds_to_zero_is_printed_to_its_first_significant_digit():
    # The cell of the manuscript: the upper limit 0.002211 is printed 0.002, not 0.00.
    assert T.value_ci(-4.058205, -7.928107, 0.002211, 2) == "$-$4.06 [$-$7.93, 0.002]"
    assert T.limit(0.002211, 2) == "0.002" and T.limit(-0.0004, 2) == "$-$0.0004"
    assert T.limit(0.0049, 2) == "0.005" and T.limit(0.00096, 2) == "0.001"
    assert T.limit(0.0031, 2, signed=True) == "+0.003" and T.limit(0.00002, 3) == "0.00002"
    # A limit that does not round to zero, and one that is exactly zero, are unchanged.
    assert T.limit(0.005001, 2) == "0.01" and T.limit(0.29, 2) == "0.29" and T.limit(0.0, 2) == "0.00"
    assert T.limit(float("nan"), 2) == "--" and T.limit(None, 2) == "--" and T.limit(0.4, 0) == "0"
    assert T.interval(-0.004, 0.004, 2) == "[$-$0.004, 0.004]"
    # The estimate before the interval is not a limit: it keeps its two decimals.
    assert T.value_ci(0.003, -0.97, 0.72, 2) == "0.00 [$-$0.97, 0.72]"
    assert T.num(-0.004, 2) == "$-$0.00"
    assert T.pm(1.6012, 0.0149, 2) == "1.60 $\\pm$ 0.01"
    assert T.best_layer(5, 0.82) == "L5 (82\\%)" and T.best_layer(float("nan"), 0.5) == "--"


def test_interval_cells_are_bold_only_when_they_exclude_zero():
    assert T.value_ci(4.51, 3.73, 5.38, 2) == "4.51 [3.73, 5.38]"
    assert T.value_ci(-1.23, -1.93, -0.48, 2, signed=True, bold_if_excludes_zero=True) == (
        "\\textbf{$-$1.23 [$-$1.93, $-$0.48]}")
    assert T.value_ci(-3.18, -6.53, 0.29, 2, signed=True, bold_if_excludes_zero=True) == (
        "$-$3.18 [$-$6.53, +0.29]")
    assert T.value_ci(1.0, 0.0, 2.0, 2, bold_if_excludes_zero=True) == "1.00 [0.00, 2.00]"   # zero is inside
    assert T.value_ci(float("nan"), 0, 1, 2) == "--"


def test_ranges_and_group_labels():
    assert T.magnitude_range([24.3, 16.7, 20.0], 1) == "+16.7 to +24.3"
    assert T.magnitude_range([-1.27, -0.76], 2) == "$-$0.76 to $-$1.27"       # smaller magnitude first
    assert T.magnitude_range([6.3, 10.2], 1, signed=False) == "6.3 to 10.2"
    assert T.magnitude_range([float("nan")], 1) == "--"
    rows = [["Eval", "1"], ["Eval", "2"], ["ITW", "1"], ["ITW", "2"]]
    assert T.group_labels(rows) == [["Eval", "1"], ["", "2"], ["ITW", "1"], ["", "2"]]
    assert rows[1][0] == "Eval"                                               # the input is not changed


# Builders ----------------------------------------------------------------------------------------

def test_every_table_has_its_shape(folder):
    shapes = {label: (len(T.build(label, folder)[1]), len(T.build(label, folder)[0])) for label in T.TABLES}
    assert shapes == {"tab:silence": (4, 5), "tab:main": (4, 8), "tab:effects": (24, 6),
                      "tab:rank": (4, 6), "tab:rankdetail": (7, 5), "tab:duration": (7, 3),
                      "tab:attack": (5, 3), "tab:threshold": (15, 4), "tab:probes": (5, 5),
                      "tab:effects-mean": (24, 6), "tab:effects-other": (24, 6), "tab:bca": (12, 10)}
    for label in T.TABLES:
        header, rows = T.build(label, folder)
        assert all(len(row) == len(header) for row in rows), label
        assert not any("--" == cell for row in rows for cell in row), label   # nothing is missing


def test_silence_and_main_tables(folder):
    assert rows_of("tab:silence", folder) == [["sil", "30~dB", "33.75", "21.88", "50.00"],
                                              ["sil", "40~dB", "35.00", "22.50", "50.00"],
                                              ["sil+dur", "30~dB", "34.75", "22.88", "51.00"],
                                              ["sil+dur", "40~dB", "36.00", "23.50", "51.00"]]
    rows = rows_of("tab:main", folder)
    assert [row[:2] for row in rows] == [["clean/stats/L3", "dev"], ["clean/stats/L4", "ref"],
                                         ["trimmed/mean/L4", "ref"], ["trimmed/stats/L5", "dev"]]
    # clean/stats/L4: evaluation EER 1 + 0.25 * 4 = 2; the matched unpadded cell is read.
    assert rows[1][2:] == ["0.50 $\\pm$ 0.01", "2.00 $\\pm$ 0.04", "[1.50, 2.50]",
                           "0.0500 $\\pm$ 0.0003", "28.00 $\\pm$ 0.25", "[19.50, 20.50]"]
    assert rows[2][3] == "5.12 $\\pm$ 0.04"                    # trimmed/mean/L4: 1 + 1 + 3 + 0.125


def test_effects_tables(folder):
    rows = rows_of("tab:effects", folder)
    assert [row[:2] for row in rows[:2]] == [["Eval", "1"], ["", "2"]] and rows[12][:2] == ["ITW", "1"]
    # Evaluation, layer 1, statistics: delta_test = 1 + 1/8, interaction = -(6 + 1/8).
    assert rows[0][2] == "\\textbf{+1.12 [+0.62, +1.62]}"
    assert rows[0][5] == "\\textbf{$-$6.12 [$-$6.62, $-$5.62]}"
    # In-the-Wild, delta_train: the interval includes zero, so the cell is not bold.
    assert rows[12][3] == "+2.62 [$-$1.00, +6.25]"
    mean = rows_of("tab:effects-mean", folder)
    assert mean[0][2] == "\\textbf{+1.38 [+0.88, +1.88]}"
    other = rows_of("tab:effects-other", folder)
    assert other[0][2:] == ["\\textbf{+4.12 [+3.62, +4.62]}", "\\textbf{+4.38 [+3.88, +4.88]}",
                            "\\textbf{+5.12 [+4.62, +5.62]}", "\\textbf{+5.38 [+4.88, +5.88]}"]


def test_rank_tables(folder):
    assert rows_of("tab:rank", folder)[0] == ["Evaluation", "stats", "0.75 [0.50, 0.88]", "0.0004",
                                              "0.88 [0.75, 0.94]", "L5 (88\\%) / L5 (62\\%)"]
    assert rows_of("tab:rank", folder)[3][:2] == ["In-the-Wild", "mean"]
    detail = rows_of("tab:rankdetail", folder)
    # Layers 1 and 2 get 20 points when trimmed: ranks 1 and 2 become 11 and 12.
    assert detail[0] == ["1", "1.25", "24.25", "$\\times$19.4", "1 $\\rightarrow$ 11"]
    assert detail[1] == ["2", "1.50", "24.50", "$\\times$16.3", "2 $\\rightarrow$ 12"]
    assert detail[2] == ["3", "1.75", "4.75", "$\\times$2.7", "3 $\\rightarrow$ 1"]
    assert detail[6] == ["7--12", "2.75--4.00", "5.75--7.00", "$\\times$1.8--2.1", "same relative order"]
    bca = rows_of("tab:bca", folder)
    assert bca[0] == ["40~dB", "Eval", "stats", "0.75", "[0.50, 0.88]", "[0.62, 0.94]", "0.50",
                      "0.013", "12\\%", "0.0004"]
    assert [row[0] for row in bca] == ["40~dB", "", "", "", "20~dB", "", "", "", "15~dB", "", "", ""]
    assert bca[5][:3] == ["", "Eval", "mean"] and bca[6][1] == "ITW"     # only the first label is grouped


def test_duration_attack_and_probe_tables(folder):
    duration = rows_of("tab:duration", folder)
    assert duration[0] == ["$E(\\mathrm{C},\\mathrm{C})$", "19.04", "18.04"]       # layer 4, statistics
    assert duration[4] == ["$\\Delta_{\\mathrm{test}}$", "$-$0.34 $\\pm$ 0.12", "$-$0.46 $\\pm$ 0.12"]
    attack = rows_of("tab:attack", folder)
    assert attack[0] == ["Most affected", "A10", "+5.0 to +7.2"]
    assert attack[1][2] == "+5.5 to +7.8; +7.5 to +9.8"
    assert attack[4] == ["Decrease", "A17", "$-$0.88 to $-$1.12"]
    probes = rows_of("tab:probes", folder)
    assert probes[0] == ["Mean F0", "0.88", "0.84", "0.78", "0.71"]              # clean, speaker folds
    assert probes[4][0] == "Length in the 4~s window" and probes[4][1] == "0.57"


def test_threshold_table(folder):
    header, rows = T.build("tab:threshold", folder)
    assert header == ["", "40~dB", "20~dB", "15~dB"]
    by_label = {row[0]: row[1:] for row in rows}
    assert by_label["Fallback files, eval / ITW"] == ["21 / 49", "546 / 1{,}274", "1{,}071 / 2{,}499"]
    assert by_label["Median removed (s), eval, bona / spoof"] == ["1.00 / 0.50", "1.25 / 0.75", "1.50 / 1.00"]
    assert by_label["Median removed (s), ITW, bona / spoof"][0] == "1.12 / 0.62"
    assert by_label["Dev-selected layer, trimmed"] == ["L5", "L5", "L5"]
    assert by_label["Eval EER, stats, L5 (\\%)"] == ["5.25", "8.25", "11.25"]
    assert by_label["Kendall's $\\tau$, eval, stats / mean"] == ["0.75 / 0.69", "0.62 / 0.56", "0.50 / 0.44"]
    assert by_label["Best layer, eval"] == ["L5 (62\\%)", "L6 (62\\%)", "L6 (62\\%)"]
    assert by_label["$\\Delta_{\\mathrm{test}}$, eval, L3--L5, both poolings"] == [
        "1.4 to 1.9", "2.4 to 2.9", "3.4 to 3.9"]
    assert by_label["Interaction, ITW"] == ["$-$4.50 [$-$5.25, $-$3.75]", "$-$5.50 [$-$6.25, $-$4.75]",
                                            "$-$6.50 [$-$7.25, $-$5.75]"]


def test_tables_are_written_and_a_missing_source_stops(folder, tmp_path):
    written = T.write_tables(folder, str(tmp_path / "tex"), log=lambda line: None)
    assert sorted(os.listdir(tmp_path / "tex")) == sorted(
        ["attack.tex", "bca.tex", "duration.tex", "effects-mean.tex", "effects-other.tex",
         "effects.tex", "main.tex", "probes.tex", "rank.tex", "rankdetail.tex", "silence.tex",
         "threshold.tex"])
    text = open(written["tab:silence"], encoding="utf-8").read().splitlines()
    assert text[0] == "% tab:silence" and text[1] == "% sources: measurements/silence_only_baseline.csv"
    assert text[3] == "sil & 30~dB & 33.75 & 21.88 & 50.00 \\\\" and len(text) == 7
    assert T.file_name("tab:effects-mean") == "effects-mean.tex"
    with pytest.raises(FileNotFoundError, match="rank_agreement.csv"):
        T.build("tab:bca", str(tmp_path))
    with pytest.raises(KeyError):
        T.build("tab:closest", folder)


# Manuscript --------------------------------------------------------------------------------------

def test_numbers_of_a_table_row():
    cells = ["\\textbf{$-$5.65 [$-$6.55, $-$4.78]}", "L5 (82\\%) / L6 (63\\%)", "1{,}147", "--",
             "$\\times$1.9--2.1", "0.44 $\\pm$ 0.02", "6 $\\rightarrow$ 12", "+0.04 to +0.12"]
    assert M.numbers_in(cells) == [-5.65, -6.55, -4.78, 82, 63, 1147, 1.9, 2.1, 0.44, 0.02, 6, 12,
                                   0.04, 0.12]
    assert M.numbers_in(["A10", "L3--L5", "40~dB"]) == [40]                # labels are not numbers
    assert M.split_cells(r"a \& b & 1.5 & $-$2") == [r"a \& b", "1.5", "$-$2"]


def test_rows_of_a_tabular_are_found_by_label(folder):
    tex = PT.fake_manuscript(folder)
    rows = M.tabular_rows(tex, "tab:silence")
    assert len(rows) == 4 and rows[0] == "sil & 30~dB & 33.75 & 21.88 & 50.00"   # header and rules left out
    assert len(M.tabular_rows(tex, "tab:effects")) == 24
    with pytest.raises(KeyError):
        M.tabular_rows(tex, "tab:none")
    with pytest.raises(ValueError):
        M.tabular_rows("\\label{tab:x} no table here", "tab:x")
    assert M.strip_comments("50\\% of a % comment\nline") == "50\\% of a \nline"


def test_the_comparison_finds_exactly_what_was_changed(folder, tmp_path):
    tex = PT.fake_manuscript(folder)
    path = tmp_path / "paper.tex"
    path.write_text(tex, encoding="utf-8")
    report = M.compare_tables(folder, str(path))
    assert list(report.columns) == M.REPORT_COLUMNS and report.empty
    assert M.summary(report)["identical"].all() and len(M.summary(report)) == 12

    changed = tex.replace("sil & 40~dB & 35.00 & 22.50 & 50.00", "sil & 40~dB & 35.00 & 22.51 & 50.00")
    changed = changed.replace("Most affected & A10 & +5.0 to +7.2 \\\\\n", "")           # a row removed
    changed = changed.replace("L5 (88\\%) / L5 (62\\%)", "L5 (88\\%) / L5")                # a number removed
    assert changed != tex
    path.write_text(changed, encoding="utf-8")
    report = M.compare_tables(folder, str(path))
    silence = report[report["table"] == "tab:silence"]
    assert len(silence) == 1
    row = silence.iloc[0]
    assert (row["row"], row["row_label"], row["number"], row["built"], row["manuscript"]) == (
        2, "sil & 40~dB", 2, 22.5, 22.51)
    attack = report[report["table"] == "tab:attack"]
    assert attack.iloc[0]["problem"] == "number of rows" and tuple(attack.iloc[0][["built", "manuscript"]]) == (5, 4)
    rank = report[report["table"] == "tab:rank"]
    # Both evaluation rows of the invented table hold this text.
    assert len(rank) == 2 and set(rank["problem"]) == {"number of numbers in the row"}
    summary = M.summary(report).set_index("table")
    assert not summary.loc["tab:silence", "identical"] and summary.loc["tab:effects", "identical"]
    assert summary["identical"].sum() == 9

    path.write_text(tex.replace("\\label{tab:bca}", "\\label{tab:other}"), encoding="utf-8")
    report = M.compare_tables(folder, str(path))
    assert list(report["table"]) == ["tab:bca"] and "no \\label{tab:bca}" in report.iloc[0]["problem"]
    one = M.compare_tables(folder, str(path), labels=["tab:silence"])
    assert one.empty


# The real manuscript and the original table script -------------------------------------------------

def test_tables_match_the_manuscript(manuscript, result_tables):
    report = M.compare_tables(result_tables, manuscript)
    assert report.empty, "\n" + report.to_string(index=False)


def test_every_cell_equals_the_original_table_script(original_src, original):
    """With SSL_SILENCE_PAPER_TABLES_ORIGINAL (the cited tables in the layout
    of the original code) and SSL_SILENCE_PAPER_TABLES (the same tables in
    the layout of this package): every cell of every table, label cells
    included, equals the cell written by the original table script."""
    old_dir = os.environ.get("SSL_SILENCE_PAPER_TABLES_ORIGINAL")
    new_dir = os.environ.get("SSL_SILENCE_PAPER_TABLES")
    if not (old_dir and new_dir):
        pytest.skip("SSL_SILENCE_PAPER_TABLES_ORIGINAL and SSL_SILENCE_PAPER_TABLES are not set")
    original("tidy")                                          # the script imports the original package
    script = original_src.parent / "scripts" / "make_paper_tables.py"
    spec = importlib.util.spec_from_file_location("original_table_script", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert {s["label"] for s in module.TABLE_SPECS.values()} == set(T.TABLES)
    for old in module.TABLE_SPECS.values():
        header, rows = T.build(old["label"], new_dir)
        old_header, old_rows = old["build"](old_dir)
        assert header == old_header and rows == old_rows, old["label"]
        assert T.TABLES[old["label"]]["label_cells"] == old["label_cells"]
