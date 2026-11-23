"""Registry of the in-text numbers: reading, comparing and the coverage report.

The machinery is tested on short invented texts and on the invented tables
of tests/paper_tables.py. ``pytest --manuscript <file.tex> --tables <folder>``
additionally compares every entry with the real manuscript.
"""

from __future__ import annotations

import re

import pandas as pd
import pytest

import paper_tables as PT
from ssl_silence.paper import intext as I
from ssl_silence.paper import tables as T

E = I.Entry


@pytest.fixture(scope="module")
def folder(tmp_path_factory) -> str:
    return PT.write_tables(str(tmp_path_factory.mktemp("results")))


# Reading -------------------------------------------------------------------------------------

def test_a_template_reads_the_numbers_of_its_phrase():
    text = I.normalise("The loss is $-5.65$ points,\n   from 19{,}387 files % a comment 7\nand $+0.11$; "
                       "50\\% remain.")
    assert "comment" not in text and "  " not in text
    found = I.template_regex(r"loss is {} points, from {} files and {}; {}\% remain").search(text)
    assert [I.read_number(g) for g in found.groups()] == [-5.65, 19387.0, 0.11, 50.0]
    # White space of the template matches any white space; other characters are literal.
    assert I.template_regex("a (b) {}.").search("a (b)   3.") and not I.template_regex("a.b {}").search("axb 3")
    assert I.printed(1.4849, 2) == 1.48 and I.printed(-0.004, 2) == 0.0 and I.printed(81.6, 0) == 82.0
    assert I.printed(0.15419, 2) == 0.15


def test_tables_are_read_once_and_rows_are_selected(folder):
    tables = I.Tables(folder)
    assert tables.val(T.EFFECTS, "estimate", condition="trim40", test_set="eval", pooling="stats",
                      layer=1, effect="delta_test") == 1.125
    assert tables.col(T.EFFECTS, "estimate", condition="trim40", test_set="eval", pooling="stats",
                      layer=[3, 4, 5], effect="delta_test").tolist() == [1.375, 1.5, 1.625]
    assert tables[T.EFFECTS] is tables[T.EFFECTS]
    with pytest.raises(I.SourceMissing, match="2 rows"):
        tables.val(T.EFFECTS, "estimate", condition="trim40", test_set="eval", pooling="stats",
                   layer=[1, 2], effect="delta_test")
    with pytest.raises(I.SourceMissing):
        tables.col(T.EFFECTS, "estimate", condition="trim99")
    with pytest.raises(I.SourceMissing):
        tables.col(T.EFFECTS, "no_such_column", condition="trim40")
    with pytest.raises(I.SourceMissing, match="column"):
        tables.rows(T.EFFECTS, attack="A10")
    with pytest.raises(I.SourceMissing, match="table"):
        tables["no_such_table.csv"]


def test_helpers_for_ranges():
    assert I.span([3.0, -1.0, 2.0], 1) == [(-1.0, 1), (3.0, 1)]
    assert I.magnitudes([-3.56, -1.85, -2.0], 2) == [(1.85, 2), (3.56, 2)]
    assert I.by_magnitude([-15.74, -5.65, -9.0], 2) == [(-5.65, 2), (-15.74, 2)]


def test_helpers_of_the_entries_added_for_the_introduction_and_the_statistics():
    assert I._common([5, 5.0, 5]) == 5.0 and I._common([5, 6]) != I._common([5, 6])       # NaN
    # A lower bound is cut off, not rounded; a value already at the digit is kept.
    assert I.lower_bound(99.883385, 1) == (99.8, 1) and I.printed(99.883385, 1) == 99.9
    assert I.lower_bound(99.8, 1) == (99.8, 1) and I.lower_bound(99.99999, 1) == (99.9, 1)
    assert I.lower_bound(0.29, 2) == (0.29, 2) and I.lower_bound(-0.11, 1) == (-0.2, 1)
    assert I.lower_bound(97.5, 0) == (97.0, 0)
    cells = pd.DataFrame({"condition": "trim40", "pooling": "stats",
                          "layer": [3, 3, 3, 4, 4, 4], "cell": ["cc", "tc", "tt"] * 2,
                          "eval_eer_mean": [2.0, 7.0, 9.0, 1.0, 3.55, 6.0]})
    tables = I.Tables.of({T.CELLS: cells})
    assert I._ratio_span(tables, layers=(3, 4)) == [(4.5, 1), (6.0, 1)]                   # E(T,T) / E(C,C)
    assert I._ratio_span(tables, layers=(3, 4), cell="tc") == [(3.5, 1), (3.55, 1)]       # E(T,C) / E(C,C)
    with pytest.raises(I.SourceMissing):
        I._ratio_span(tables, layers=(5,))
    rank = pd.DataFrame({"condition": ["trim40", "trim40", "trim20"], "test_set": "eval",
                         "pooling": ["stats", "mean", "mean"], "z0_tau": [0.0, 1.0, -0.5],
                         "a_tau": [0.0, 0.0, 0.0], "best_layer_clean": [5, 5, 5],
                         "best_layer_trimmed": [5, 5, 6]})
    tables = I.Tables.of({T.RANK: rank})
    # z0 = 0: half of the resamples are below; the upper BCa level is the 97.5th percentile.
    below = I._share_below(tables, condition="trim40")
    assert below[0] == 50.0 and round(float(below[1]), 2) == 84.13
    level = I._bca_upper_level(tables, condition="trim40")
    assert round(float(level[0]), 6) == 97.5 and round(float(level[1]), 3) == 99.996      # Phi(1 + 2.96)
    both = ("best_layer_clean", "best_layer_trimmed")
    assert I._best_layers(tables, "eval", [("trim40", "stats"), ("trim40", "mean")], both) == 5.0
    assert I._best_layers(tables, "eval", [("trim20", "mean")]) == 6.0
    entry = {e.key: e for e in I.ENTRIES}["statistics_bca_upper_level"]
    assert entry.values(tables) == [(97.5, 1)]            # the smaller of 97.5 and 99.996
    mixed = I._best_layers(tables, "eval", [("trim40", "mean"), ("trim20", "mean")])
    assert mixed != mixed                                                                 # they differ


def test_the_probe_entries_are_compared_with_the_numbers_the_manuscript_prints():
    entries = I.probe_entries()
    assert [entry.key for entry in entries] == ["probe_sample", "probe_lead", "probe_length"]
    assert I.STATED_PROBES == {"probe_sample": (768, 5000, 20), "probe_lead": (0.74,),
                               "probe_length": (0.90, 0.96, 0.88)}
    rows = []
    for condition, lengths in (("clean", [0.9003, 0.9617, 0.93]), ("trim40", [0.886956, 0.96, 0.9])):
        for layer, (lead, length) in enumerate(zip([0.7250, 0.7365, 0.7485], lengths), start=1):
            for target, value in (("win_lead_40", lead), ("length_in_window", length), ("pad", length)):
                rows.append({"encoder": "wavlm-base", "condition": condition, "cv": "speaker", "layer": layer,
                             "target": target, "control": "real", "n_files": 5000, "n_groups": 20,
                             "r2_mean": value})
    r2 = pd.DataFrame(rows)
    table = I.compare_stated(I.Tables.of({T.PROBE_R2: r2}), entries, I.STATED_PROBES)
    assert table["ok"].all() and table["stated"].tolist() == [768, 5000, 20, 0.74, 0.90, 0.96, 0.88]
    # "at least 0.88": the smallest R2 on trimmed audio is cut off, not rounded (0.886956 is not 0.89).
    assert table["printed"].tolist()[-1] == 0.88 and I.printed(0.886956, 2) == 0.89
    lower = r2.copy()
    lower.loc[(lower["condition"] == "trim40") & (lower["layer"] == 1) & (lower["target"] == "length_in_window"),
              "r2_mean"] = 0.8799
    other = I.compare_stated(I.Tables.of({T.PROBE_R2: lower}), entries, I.STATED_PROBES)
    assert other["ok"].tolist() == [True] * 6 + [False] and other["printed"].tolist()[-1] == 0.87


# Comparing --------------------------------------------------------------------------------------

def entries_for_a_small_text():
    effect = dict(condition="trim40", test_set="eval", pooling="stats", effect="delta_test")
    return (
        E("first", "1", r"the first layer loses {} points",
          lambda t: [(t.val(T.EFFECTS, "estimate", layer=1, **effect), 2)]),
        E("range", "1", r"between {} and {} points",
          lambda t: I.span(t.col(T.EFFECTS, "estimate", layer=[3, 4, 5], **effect), 1)),
        E("wrong", "1", r"layer~2 loses {} points",
          lambda t: [(t.val(T.EFFECTS, "estimate", layer=2, **effect), 2)]),
        E("absent", "1", r"this phrase is not in the text {}", lambda t: [(1.0, 0)]),
        E("twice", "1", r"repeated {}", lambda t: [(1.0, 0)]),
        E("missing", "1", r"At 10~dB it loses {} points",
          lambda t: [(t.val(T.EFFECTS, "estimate", layer=1, **{**effect, "condition": "trim10"}), 2)]),
        E("count", "1", r"two numbers {} and {}", lambda t: [(1.0, 0)]),
        E("prior", "1", r"Earlier work reported {}\%", kind="cited"),
        E("rounded", "1", r"about {} points wide", kind="approximate"),
        E("constant", "1", r"a window of {} samples", lambda t: [(64_000, 0)], kind="config"),
        E("to_correct", "1", r"and {} others", lambda t: [(78, 0)],
          pending="the manuscript is to be corrected"),
        E("corrected", "1", r"the first layer loses {} points, ",
          lambda t: [(t.val(T.EFFECTS, "estimate", layer=1, **effect), 2)],
          pending="the manuscript is to be corrected"),
        # Statements without a number: the phrase must be present and the tables must give what it states.
        E("stated", "1", r"Something is repeated",
          lambda t: [(int((t.col(T.EFFECTS, "estimate", layer=[3, 4, 5], **effect) > 1).sum()), 0), (3, 0)],
          kind="statement", expected=(3, 3)),
        E("stated_wrongly", "1", r"There are two numbers",
          lambda t: [(int((t.col(T.EFFECTS, "estimate", layer=[3, 4, 5], **effect) > 1.5).sum()), 0), (3, 0)],
          kind="statement", expected=(3, 3)),
        E("stated_nowhere", "1", r"the intervals include zero", lambda t: [(0, 0)],
          kind="statement", expected=(0,)),
        E("stated_without_source", "1", r"Earlier work reported",
          lambda t: [(t.val(T.EFFECTS, "estimate", layer=1, **{**effect, "condition": "trim10"}), 0)],
          kind="statement", expected=(0,)),
    )


SMALL_TEXT = r"""
In domain the first layer loses 1.12 points, % 1.125 in the table
and layers 3--5 lose between 1.4 and 1.6 points; layer~2 loses 1.30 points.
Something is repeated 1 and repeated 1. At 10~dB it loses 9.99 points.
There are two numbers 1 and 2. Earlier work reported 15.5\% and the intervals
are about 5 points wide, with a window of 64{,}000 samples and 77 others.
"""


def test_check_reports_one_status_per_entry(folder):
    report = I.check(folder, SMALL_TEXT, entries_for_a_small_text()).set_index("key")
    assert report["status"].to_dict() == {
        "first": "ok", "range": "ok", "wrong": "different", "absent": "not found",
        "twice": "ambiguous", "missing": "source missing", "count": "different", "prior": "cited",
        "rounded": "approximate", "constant": "ok", "to_correct": "pending correction",
        "corrected": "ok", "stated": "ok", "stated_wrongly": "different",
        "stated_nowhere": "not found", "stated_without_source": "source missing"}
    # A statement has no number in the manuscript: what the tables give and what it needs are reported.
    assert report.loc["stated", ["n_numbers", "manuscript", "computed", "detail"]].tolist() == [0, "", "3, 3", ""]
    wrong = report.loc["stated_wrongly"]
    assert wrong["manuscript"] == "" and wrong["computed"] != "3, 3"
    assert wrong["detail"] == "the statement needs 3, 3"
    assert report.loc["stated_nowhere", "detail"] == "0 occurrences of the phrase"
    assert report.loc["first", ["manuscript", "computed"]].tolist() == ["1.12", "1.12"]
    assert report.loc["range", "manuscript"] == "1.4, 1.6" and report.loc["range", "n_numbers"] == 2
    assert report.loc["wrong", ["manuscript", "computed"]].tolist() == ["1.3", "1.25"]
    assert report.loc["twice", "detail"] == "2 occurrences of the phrase"
    assert "trim10" in report.loc["missing", "detail"]
    assert report.loc["count", "detail"] == "2 numbers in the phrase, 1 computed"
    assert report.loc["constant", "manuscript"] == "64000"
    # A difference that the first author has decided to correct in the manuscript.
    assert report.loc["to_correct", ["manuscript", "computed", "detail"]].tolist() == [
        "77", "78", "the manuscript is to be corrected"]


def test_coverage_names_what_no_entry_covers(folder):
    text = ("\\section{Results}" + SMALL_TEXT
            + "\\begin{table} 1.23 & 4.56 \\end{table} See Table~\\ref{tab:x2} and layers 6, 8, and 9 "
              "at 20~dB with L4 and A10. \\bibliography{refs} 2026")
    report = I.coverage(text, entries_for_a_small_text())
    by_state = {state: sorted(part["number"]) for state, part in report.groupby("state")}
    assert set(by_state["checked"]) == {"1.12", "1.4", "1.6", "1.30", "9.99", "64{,}000", "1", "2",
                                        "77"}
    assert by_state["cited"] == ["15.5"] and by_state["approximate"] == ["5"]
    assert set(by_state["structural"]) == {"3", "5", "2", "6", "8", "9", "20", "10"}
    assert by_state["unchecked"] == ["1", "1"]                    # "repeated 1", found twice
    assert "1.23" not in set(report["number"]) and "2026" not in set(report["number"])
    row = report[report["number"] == "77"].iloc[0]
    assert "samples and 77 others" in row["context"] and row["entry"] == "to_correct"
    assert report[report["number"] == "1.6"].iloc[0]["entry"] == "range"    # second number of a range
    summary = I.coverage_summary(report).set_index("state")["n"]
    assert summary.to_dict() == {"checked": 9, "cited": 1, "approximate": 1, "structural": 8,
                                 "unchecked": 2}


# The registry -------------------------------------------------------------------------------------

def test_registry_is_well_formed():
    keys = [entry.key for entry in I.ENTRIES]
    assert len(keys) == len(set(keys)) and len(keys) > 80
    for entry in I.ENTRIES:
        assert entry.kind in I.KINDS, entry.key
        # A statement has no number in its phrase and says what the tables must give; every other
        # entry has at least one number and nothing expected.
        assert ("{}" in entry.template) == (entry.kind != "statement"), entry.key
        assert bool(entry.expected) == (entry.kind == "statement"), entry.key
        assert (entry.values is None) == (entry.kind in I.UNCOMPARED), entry.key
        I.template_regex(entry.template)
        assert re.fullmatch(r"abstract|\d(\.\d)?", entry.section), entry.key
    sections = {entry.section for entry in I.ENTRIES}
    assert {"abstract", "3.2", "4.1", "4.2", "4.3", "4.4", "4.5", "4.6", "4.7", "4.8"} <= sections
    # The numbers the first author asked for by name.
    assert {"fallback_per_class_itw", "fallback_per_class_asvspoof", "removed_eval", "removed_itw",
            "padded_against_unpadded"} <= set(keys)
    assert {"intro_tau", "intro_best_layer", "intro_best_layer_strict", "intro_delta_test",
            "intro_attack_a10", "data_speakers", "statistics_tau_below", "statistics_bca_z0",
            "statistics_tau_ties", "statistics_bca_upper_level", "discussion_factor",
            "discussion_factor_clean_test"} <= set(keys)
    # No entry of the registry is waiting for a correction of the manuscript.
    assert not [entry.key for entry in I.ENTRIES if entry.pending]


def test_every_entry_computes_or_names_its_missing_source(folder):
    """On the invented tables an entry either gives one value per number of its
    phrase or says which table or rows are missing; nothing else is raised."""
    tables = I.Tables(folder)
    computed = missing = 0
    for entry in I.ENTRIES:
        if entry.values is None:
            continue
        n_numbers = len(entry.expected) if entry.kind == "statement" else entry.template.count("{}")
        try:
            values = entry.values(tables)
        except I.SourceMissing:
            missing += 1
            continue
        assert len(values) == n_numbers, entry.key
        assert all(isinstance(digits, int) and float(value) == float(value) for value, digits in values), entry.key
        computed += 1
    assert computed > 45 and missing > 0


def test_in_text_numbers_match_the_manuscript(manuscript, result_tables):
    report = I.check(result_tables, open(manuscript, encoding="utf-8").read())
    # An entry marked as a pending correction of the manuscript does not fail the run.
    bad = report[report["status"].isin(["different", "not found", "ambiguous"])]
    assert bad.empty, "\n" + bad[["key", "section", "status", "manuscript", "computed", "detail"]].to_string(index=False)


def test_every_source_of_an_in_text_number_is_saved(manuscript, result_tables):
    report = I.check(result_tables, open(manuscript, encoding="utf-8").read())
    missing = report[report["status"] == "source missing"]
    assert missing.empty, "\n" + missing[["key", "section", "manuscript", "detail"]].to_string(index=False)


def test_no_number_of_the_manuscript_is_unchecked(manuscript):
    report = I.coverage(open(manuscript, encoding="utf-8").read())
    unchecked = report[report["state"] == "unchecked"]
    assert unchecked.empty, "\n" + unchecked[["number", "context"]].to_string(index=False)


# The padding numbers -----------------------------------------------------------------------------

def padding_frames(shift: float = 0.0) -> dict:
    rows = [{"condition": c, "sample": s, "comparison": m, "statistic": statistic, "layer": layer,
             "rel_mean_diff": 0.1 * layer + (0.004 if m == "mask" else 0.0) + (0.2 if c == "trim40" else 0.0)
             + (0.3 if s == "most_padded" else 0.0) + shift}
            for c in ("clean", "trim40") for s in ("random", "most_padded")
            for m in ("no_mask", "mask", "mask_vs_no_mask") for statistic in ("mean", "std")
            for layer in (3, 4)]
    amount = [{"condition": "trim40", "sample": "random", "comparison": "no_mask", "pad_bin": b,
               "mean_rel_L4": v + shift} for b, v in (("<1 s", 0.2017), (">3 s", 0.9418))]
    files = [{"condition": c, "sample": "random", "pad_s": v + shift}
             for c, values in (("trim40", (1.0, 1.76, 3.0)), ("clean", (0.5, 0.85, 2.0))) for v in values]
    return {I.PADDING_DEVIATION: pd.DataFrame(rows), I.PADDING_BY_AMOUNT: pd.DataFrame(amount),
            I.PADDING_PER_FILE: pd.DataFrame(files)}


def test_the_padding_entries_and_their_printed_values():
    entries = I.padding_entries()
    assert [e.key for e in entries] == ["padding_deviation_layer4", "padding_median", "padding_by_amount",
                                        "padding_most_padded_layer4"]
    assert entries[-1] not in I.ENTRIES and entries[-1].section == "analysis plan"
    values = I.printed_values(I.Tables.of(padding_frames()), entries)
    assert len(values) == 10 and set(values["digits"]) == {2}
    by_key = {key: part["printed"].tolist() for key, part in values.groupby("key", sort=False)}
    # layer 4, random: without a mask clean and trimmed, then with a mask
    assert by_key["padding_deviation_layer4"] == [0.4, 0.6, 0.4, 0.6]
    assert by_key["padding_median"] == [1.76, 0.85] and by_key["padding_by_amount"] == [0.2, 0.94]
    # most padded, trimmed, layer 4: with a mask, then without
    assert by_key["padding_most_padded_layer4"] == [0.9, 0.9]
    assert values.loc[2, "value"] == pytest.approx(0.404)
    with pytest.raises(I.SourceMissing):
        I.printed_values(I.Tables.of({}), entries)


def test_numbers_are_compared_at_the_printed_precision():
    entries = I.padding_entries()
    reference = I.Tables.of(padding_frames())
    same = I.compare_printed(I.Tables.of(padding_frames(0.0004)), reference, entries)
    assert same["ok"].all() and (same["value"] != same["reference_value"]).all() and len(same) == 10
    assert list(same.columns) == ["key", "position", "digits", "value", "reference_value", "printed",
                                  "reference_printed", "ok"]
    # 0.404 + 0.001 is printed 0.41 (rounded once), 0.2017 + 0.001 is still printed 0.20
    moved = I.compare_printed(I.Tables.of(padding_frames(0.001)), reference, entries).set_index(["key", "position"])
    assert not moved.loc[("padding_deviation_layer4", 2), "ok"]
    assert moved.loc[("padding_deviation_layer4", 2), ["printed", "reference_printed"]].tolist() == [0.41, 0.4]
    assert moved.loc[("padding_by_amount", 0), "ok"]
    # Without a mask the numbers are 0.4 and 0.6 exactly: 0.001 more does not change what is printed.
    assert moved.loc[("padding_deviation_layer4", 0), "ok"] and moved.loc[("padding_deviation_layer4", 1), "ok"]



# The silence that remains after trimming -----------------------------------------------------------

def remaining_share(itw_window: float = 2.149) -> pd.DataFrame:
    rows = [("train", "all", 4.15, 3.98, 0.20), ("dev", "all", 3.92, 3.72, 0.23),
            ("eval", "0", 6.70, 6.54, 0.26), ("eval", "1", 2.97, 2.71, 0.21), ("eval", "all", 3.36, 3.11, 0.22),
            ("itw", "0", 3.52, 2.49, 0.14), ("itw", "all", 3.10, itw_window, 0.16)]
    return pd.DataFrame(rows, columns=["table", "label", "pct_lead_over", "pct_win_lead_over", "lead_median_s"])


def test_the_remaining_silence_is_compared_with_the_numbers_the_manuscript_prints():
    entries = I.remaining_silence_entries()
    assert [entry.key for entry in entries] == ["remaining_silence_share", "remaining_silence_by_class"]
    assert all(entry in I.ENTRIES and entry.section == "4.7" for entry in entries)
    assert I.STATED == {"remaining_silence_share": (3.1, 4.2, 2.1, 4.0),
                        "remaining_silence_by_class": (6.7, 3.0, 0.16, 0.23)}
    table = I.compare_stated(I.Tables.of({I.RESIDUAL_SHARE: remaining_share()}), entries)
    assert list(table.columns) == ["key", "position", "digits", "value", "printed", "stated", "ok"]
    assert table["ok"].all() and table["digits"].tolist() == [1, 1, 1, 1, 1, 1, 2, 2]
    # Smallest and largest share of the sets (rows of all files), whole file and window; the
    # evaluation set by class; smallest and largest median.
    # A value is rounded once: 2.149 is printed 2.1, not 2.2 (2.15, then 2.2).
    assert table["printed"].tolist() == [3.1, 4.2, 2.1, 4.0, 6.7, 3.0, 0.16, 0.23]
    other = I.compare_stated(I.Tables.of({I.RESIDUAL_SHARE: remaining_share(2.151)}), entries)
    assert other["ok"].tolist() == [True, True, False] + [True] * 5 and other.loc[2, "printed"] == 2.2
    # The manuscript prints 2.1: neither entry is marked as a pending correction.
    assert not entries[0].pending and not entries[1].pending
    with pytest.raises(ValueError, match="stated"):
        I.compare_stated(I.Tables.of({I.RESIDUAL_SHARE: remaining_share()}), entries,
                         {**I.STATED, "remaining_silence_share": (3.1, 4.2)})
    with pytest.raises(I.SourceMissing):
        I.compare_stated(I.Tables.of({}), entries)
    # The phrases of the sentence are found in a source, each once.
    tex = (r"After trimming at 40~dB, more than 50~ms of leading silence remains in 3.1--4.2\% of the "
           r"files of each set (2.1--4.0\% within the 4-second window), more often in bonafide than in "
           r"spoofed files (6.7\% and 3.0\% on the evaluation set), with a median of 0.16--0.23~s.")
    for entry, stated in zip(entries, I.STATED.values()):
        found = I.template_regex(entry.template).findall(I.normalise(tex))
        assert [tuple(I.read_number(g) for g in groups) for groups in found] == [stated]


def test_the_quiet_files_are_compared_with_the_stated_number():
    assert I.STATED_QUIET == {"table": "itw", "top_db": 40, "n_files": 5, "label": 0}

    def quiet(labels, table="itw", top_db=40.0):
        return pd.DataFrame({"table": table, "audio_id": [f"{k}.wav" for k in range(len(labels))],
                             "label": labels, "top_db": top_db})

    row = I.compare_quiet(pd.concat([quiet([0] * 5), quiet([1, 1], table="eval")])).iloc[0]
    assert row.to_dict() == {"table": "itw", "top_db": 40.0, "n_files": 5, "n_bonafide": 5, "n_spoof": 0,
                             "stated_n_files": 5, "stated_label": 0, "ok": True}
    assert not I.compare_quiet(quiet([0] * 4)).iloc[0]["ok"]
    assert not I.compare_quiet(quiet([0] * 6)).iloc[0]["ok"]
    spoof = I.compare_quiet(quiet([0, 0, 0, 0, 1])).iloc[0]
    assert (spoof["n_files"], spoof["n_spoof"], spoof["ok"]) == (5, 1, False)
    assert not I.compare_quiet(quiet([0] * 5, table="eval")).iloc[0]["ok"]
    with pytest.raises(ValueError, match="40 dB"):
        I.compare_quiet(quiet([0] * 5, top_db=20.0))
    report = I.coverage(r"\section{Results} RMS below $10^{-3}$, so that the floor of $10^{-5}$ lies")
    assert set(report["state"]) == {"structural"} and len(report) >= 2
