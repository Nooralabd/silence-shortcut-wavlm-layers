"""Units of the replication rule on synthetic tables with known answers."""

from __future__ import annotations

import itertools
from pathlib import Path

import pandas as pd
import pytest

from ssl_silence import config as cfg
from ssl_silence import replication as rep


def effects_table(condition: str = "trim40", layers=range(1, 13)) -> pd.DataFrame:
    """delta_test is +2 [1, 3]; interaction is -1 [-2, -0.5] on eval and -1 [-2, 0.5] on
    itw; every other effect is 0.5 [-0.1, 1]."""
    rows = []
    for effect, test_set, pooling, layer in itertools.product(
            cfg.EFFECTS, cfg.TEST_SETS, cfg.POOLINGS, layers):
        if effect == "delta_test":
            value, lo, hi = 2.0, 1.0, 3.0
        elif effect == "interaction":
            value, lo, hi = -1.0, -2.0, (-0.5 if test_set == "eval" else 0.5)
        else:
            value, lo, hi = 0.5, -0.1, 1.0
        rows.append({"condition": condition, "test_set": test_set, "pooling": pooling,
                     "layer": layer, "effect": effect, "estimate": value,
                     "spk_ci_low": lo, "spk_ci_high": hi})
    return pd.DataFrame(rows)


def threshold_effects(condition: str) -> pd.DataFrame:
    """delta_total at every layer, the other effects at layers 3-5 only."""
    table = effects_table(condition)
    return table[(table["effect"] == "delta_total") | table["layer"].isin((3, 4, 5))]


def all_effects() -> pd.DataFrame:
    return pd.concat([effects_table(), threshold_effects("trim20"),
                      threshold_effects("trim15")], ignore_index=True)


def rank_table() -> pd.DataFrame:
    rows = [{"condition": c, "test_set": t, "pooling": p, "kendall_tau": 0.5,
             "tau_ci_low": 0.1 if t == "eval" else -0.1, "tau_ci_high": 0.8}
            for c in ("trim40", "trim20", "trim15") for t in cfg.TEST_SETS for p in cfg.POOLINGS]
    return pd.DataFrame(rows)


def difference_table() -> pd.DataFrame:
    rows = [{"condition": c, "test_set": t, "pooling": p, "estimate": 0.2,
             "spk_ci_low": 0.05 if c == "trim15" else -0.05, "spk_ci_high": 0.4}
            for c in ("trim20", "trim15") for t in cfg.TEST_SETS for p in cfg.POOLINGS]
    return pd.DataFrame(rows)


def attack_table() -> pd.DataFrame:
    rows = []
    for condition, attack, layer, pooling in itertools.product(
            ("trim40", "trim20", "trim15"), ["A07", "A17"], (3, 4, 5), cfg.POOLINGS):
        value = 3.0 if attack == "A07" else -0.2
        rows.append({"condition": condition, "attack": attack, "pooling": pooling, "layer": layer,
                     "estimate": value, "spk_ci_low": value - 0.1, "spk_ci_high": value + 0.1})
    return pd.DataFrame(rows)


def test_effect_units_count_sign_and_flag():
    units = rep.effect_units(effects_table())
    assert len(units) == 288 and list(units.columns) == rep.UNIT_COLUMNS
    assert list(units["quantity"].unique()) == list(cfg.EFFECTS)
    dtest = units[units["quantity"] == "delta_test"]
    assert dtest["in_rule"].all() and (dtest["sign"] == "positive").all()
    inter = units[units["quantity"] == "interaction"]
    assert (inter["sign"] == "negative").all()
    assert inter[inter["test_set"] == "eval"]["in_rule"].all()
    assert not inter[inter["test_set"] == "itw"]["in_rule"].any()
    assert int(units["in_rule"].sum()) == 48 + 24


def test_interval_touching_zero_is_not_in_the_rule():
    table = effects_table()
    table.loc[table["effect"] == "delta_test", "spk_ci_low"] = 0.0
    units = rep.effect_units(table)
    assert not units[units["quantity"] == "delta_test"]["in_rule"].any()


def test_incomplete_or_duplicated_effects_raise():
    table = effects_table()
    with pytest.raises(ValueError):
        rep.effect_units(table.iloc[:-1])
    with pytest.raises(ValueError):
        rep.effect_units(pd.concat([table, table.iloc[:1]]))
    with pytest.raises(KeyError):
        rep.effect_units(table.drop(columns="spk_ci_low"))


def test_threshold_conditions_use_their_own_layers():
    units = rep.effect_units(all_effects(), "trim20", layers=(3, 4, 5))
    assert len(units) == 6 * 2 * 3 * 2
    assert set(units["condition"]) == {"trim20"} and set(units["layer"]) == {3, 4, 5}
    with pytest.raises(ValueError):
        rep.effect_units(all_effects(), "trim20")
    total = rep.effect_units(all_effects(), "trim20", names=("delta_total",))
    assert len(total) == 2 * 12 * 2 and set(total["quantity"]) == {"delta_total"}


def test_all_units_and_summary():
    units = rep.replication_units(all_effects(), rank_table(), attack_table(),
                                  tau_differences=difference_table())
    assert len(units) == (288 + 4 + 12) + 2 * (48 + 60 + 4 + 12 + 4)
    assert not units.duplicated(rep.KEYS).any()
    summary = rep.summarise_units(units).set_index(["finding", "quantity", "condition", "test_set"])
    row = summary.loc[("effect", "delta_test", "trim40", "eval")]
    assert row["n_units"] == 24 and row["n_in_rule"] == 24 and row["n_positive_in_rule"] == 24
    assert summary.loc[("effect", "interaction", "trim40", "eval"), "n_negative_in_rule"] == 24
    assert summary.loc[("effect", "interaction", "trim40", "itw"), "n_in_rule"] == 0
    assert summary.loc[("effect", "delta_test", "trim15", "eval"), "n_units"] == 6
    assert summary.loc[("effect", "delta_total", "trim15", "eval"), "n_units"] == 24
    assert summary.loc[("per_attack", "delta_total", "trim20", "eval"), "n_units"] == 12
    assert summary.loc[("rank_agreement", "kendall_tau", "trim40", "eval"), "n_in_rule"] == 2
    assert summary.loc[("rank_agreement", "kendall_tau", "trim20", "itw"), "n_in_rule"] == 0
    assert summary.loc[("rank_agreement", "tau_difference", "trim20", "eval"), "n_in_rule"] == 0
    assert summary.loc[("rank_agreement", "tau_difference", "trim15", "eval"), "n_in_rule"] == 2
    row = summary.loc[("per_attack", "delta_total", "trim40", "eval")]
    assert row["n_units"] == 12 and row["n_positive_in_rule"] == 6 and row["n_negative_in_rule"] == 6


def test_duration_units():
    from ssl_silence import config as cfg

    rows = [{"condition": "trim40", "subset": s, "pooling": p, "layer": layer, "effect": e,
             "estimate": -3.0 if s == "<=4s" else 0.1,
             "spk_ci_low": -4.0 if s == "<=4s" else -0.5, "spk_ci_high": -2.0 if s == "<=4s" else 0.6}
            for s in ("all", "<=4s", ">4s") for p in cfg.POOLINGS for layer in (3, 4, 5)
            for e in cfg.EFFECTS]
    table = pd.DataFrame(rows)
    units = rep.duration_units(table)
    # 6 effects x 2 subsets x layers 3-5 x 2 poolings; the whole-set rows are no units.
    assert len(units) == 72 and len(table) == 108
    assert set(units["subset"]) == {"<=4s", ">4s"} and "all" not in set(units["subset"])
    assert set(units["test_set"]) == {"itw"} and set(units["finding"]) == {"duration"}
    assert units.groupby("subset")["in_rule"].all().to_dict() == {"<=4s": True, ">4s": False}
    summary = rep.summarise_units(units)
    assert len(summary) == 12 and set(summary["n_units"]) == {6}     # "k of 6 each"
    with pytest.raises(ValueError):                                  # a subset is missing
        rep.duration_units(table[table["subset"] != ">4s"])
    with pytest.raises(ValueError):                                  # a layer is missing
        rep.duration_units(table[table["layer"] != 5])
    assert len(rep.duration_units(table, layers=(3, 4))) == 48


def test_comparison_and_outcome_table():
    reference = rep.effect_units(effects_table())
    table = effects_table()
    eval_dtest = (table["effect"] == "delta_test") & (table["test_set"] == "eval")
    # Replica: delta_test on eval keeps its sign in layers 1-6, flips in 7-9 and
    # includes zero in 10-12; delta_train on itw now excludes zero everywhere.
    table.loc[eval_dtest & table["layer"].between(7, 9), ["estimate", "spk_ci_low", "spk_ci_high"]] = \
        [-2.0, -3.0, -1.0]
    table.loc[eval_dtest & table["layer"].between(10, 12), "spk_ci_low"] = -1.0
    itw_dtrain = (table["effect"] == "delta_train") & (table["test_set"] == "itw")
    table.loc[itw_dtrain, "spk_ci_low"] = 0.2
    comparison = rep.compare_units(reference, rep.effect_units(table))
    assert len(comparison) == 288
    assert int(comparison["replicated"].sum()) == 12 + 24 + 24
    outcome = rep.outcome_table(comparison).set_index(["quantity", "test_set"])
    row = outcome.loc[("delta_test", "eval")]
    assert (row["both_exclude_same_sign"], row["both_exclude_opposite_sign"],
            row["reference_only"], row["replica_only"], row["neither"]) == (12, 6, 6, 0, 0)
    row = outcome.loc[("delta_train", "itw")]
    assert (row["replica_only"], row["neither"], row["both_exclude_same_sign"]) == (24, 0, 0)
    count_columns = ["both_exclude_same_sign", "both_exclude_opposite_sign", "reference_only",
                     "replica_only", "neither"]
    assert (outcome[count_columns].sum(axis=1) == outcome["n_units"]).all()
    with pytest.raises(KeyError):
        rep.compare_units(reference, rep.effect_units(table).iloc[:-1])


def test_matched_totals_fill_the_layers_the_effects_table_lacks():
    full = all_effects()
    matched = full[(full["effect"] == "delta_total") & full["condition"].isin(("trim20", "trim15"))]
    effects = full[(full["condition"] == "trim40") | full["layer"].isin((3, 4, 5))]
    combined = rep.add_matched_totals(effects, matched.assign(extra=1))
    assert len(combined) == len(full) and not combined.duplicated(
        ["condition", "test_set", "pooling", "layer", "effect"]).any()
    assert "extra" not in combined.columns
    units = rep.replication_units(combined, rank_table(), attack_table(),
                                  tau_differences=difference_table())
    assert len(units) == (288 + 4 + 12) + 2 * (48 + 60 + 4 + 12 + 4)
    with pytest.raises(ValueError, match="expected 48"):
        rep.replication_units(effects, rank_table(), attack_table())
    # A row that both tables have must be identical.
    changed = matched.copy()
    row = changed.index[(changed["layer"] == 4)][0]
    changed.loc[row, "spk_ci_high"] += 1e-12
    with pytest.raises(ValueError, match="1 rows of the matched totals differ .* spk_ci_high"):
        rep.add_matched_totals(effects, changed)
    with pytest.raises(KeyError):
        rep.add_matched_totals(effects, matched.drop(columns="estimate"))


def test_the_layers_of_the_units_can_be_given():
    full = all_effects()
    effects = full[full["layer"].isin((3, 4, 5))]
    units = rep.replication_units(effects, rank_table(), attack_table(), layers=(3, 4, 5))
    assert len(units) == (6 * 2 * 3 * 2 + 4 + 12) + 2 * (12 + 60 + 4 + 12)


# The rule, the verdict and the findings of the plan ------------------------------------------------

REPO = Path(__file__).resolve().parents[1]


def committed_units() -> pd.DataFrame:
    return rep.read_units(str(REPO / rep.PREREGISTERED_UNITS))


def test_a_units_file_is_read_in_the_layout_of_computed_units(tmp_path):
    units = rep.replication_units(all_effects(), rank_table(), attack_table())
    path = tmp_path / "units.csv"
    units.to_csv(path, index=False, lineterminator="\n")
    read = rep.read_units(str(path))
    assert list(read.columns) == rep.UNIT_COLUMNS and len(read) == len(units)
    assert read["in_rule"].dtype == bool and read["in_rule"].tolist() == units["in_rule"].tolist()
    assert str(read["layer"].dtype) == "Int64" and read["layer"].isna().sum() == units["layer"].isna().sum()
    assert set(read["subset"]) == {""} and "" in set(read["attack"])
    assert read["estimate"].tolist() == units["estimate"].tolist()
    # Read and computed units are the same units: every one is compared with itself.
    comparison = rep.compare_units(read, units)
    assert len(comparison) == len(units) and comparison["same_sign"].all()
    assert comparison["replicated"].tolist() == units["in_rule"].tolist()
    bad = units.astype({"in_rule": object})
    bad.loc[0, "in_rule"] = "yes"
    with pytest.raises(ValueError, match="in_rule"):
        rep.normalise_units(bad)


def test_the_verdict_is_given_to_the_units_in_the_rule_only():
    reference = rep.effect_units(effects_table())
    table = effects_table()
    eval_dtest = (table["effect"] == "delta_test") & (table["test_set"] == "eval")
    table.loc[eval_dtest & table["layer"].between(7, 9), ["estimate", "spk_ci_low", "spk_ci_high"]] = \
        [-2.0, -3.0, -1.0]                                          # opposite sign, excludes zero
    table.loc[eval_dtest & table["layer"].between(10, 12), "spk_ci_low"] = -1.0   # same sign, includes zero
    itw_dtrain = (table["effect"] == "delta_train") & (table["test_set"] == "itw")
    table.loc[itw_dtrain, "spk_ci_low"] = 0.2                       # excludes zero where the reference did not
    comparison = rep.compare_units(reference, rep.effect_units(table))
    assert set(comparison["verdict"]) == set(rep.VERDICTS)
    in_rule = comparison[comparison["in_rule"]]
    assert set(in_rule["verdict"]) == {"replicated", "not replicated"}
    assert set(comparison[~comparison["in_rule"]]["verdict"]) == {"descriptive"}
    assert (comparison["verdict"] == "replicated").tolist() == comparison["replicated"].tolist()
    part = comparison[(comparison["quantity"] == "delta_test") & (comparison["test_set"] == "eval")]
    by_layer = part.groupby("layer")["verdict"].agg(lambda v: set(v))
    assert all(by_layer[k] == {"replicated"} for k in range(1, 7))
    assert all(by_layer[k] == {"not replicated"} for k in range(7, 13))
    # A unit outside the rule gets no verdict, whatever the second encoder shows.
    reverse = comparison[(comparison["quantity"] == "delta_train") & (comparison["test_set"] == "itw")]
    assert reverse["replica_excludes_zero"].all() and set(reverse["verdict"]) == {"descriptive"}
    assert not reverse["replicated"].any()
    outcome = rep.outcome_table(comparison).set_index(["quantity", "test_set"])
    assert list(outcome.columns)[-len(rep.OUTCOME_COUNTS):] == rep.OUTCOME_COUNTS
    row = outcome.loc[("delta_test", "eval")]
    assert (row["n_units"], row["n_in_rule"], row["n_replicated"]) == (24, 24, 12)
    row = outcome.loc[("delta_train", "itw")]
    assert (row["n_in_rule"], row["n_replicated"], row["replica_only"]) == (0, 0, 24)
    assert outcome["n_replicated"].sum() == comparison["replicated"].sum()
    assert (outcome["n_replicated"] <= outcome["n_in_rule"]).all()


def test_tau_follows_the_general_rule_and_the_comparison_shows_both_signs():
    reference = rep.rank_units(rank_table())                       # eval in the rule, itw not
    replica = rank_table()
    replica.loc[(replica["test_set"] == "eval") & (replica["pooling"] == "mean"),
                ["kendall_tau", "tau_ci_low", "tau_ci_high"]] = [-0.5, -0.8, -0.1]
    comparison = rep.compare_units(reference, rep.rank_units(replica)).set_index(["test_set", "pooling"])
    assert not hasattr(rep, "INTERVAL_ONLY_QUANTITIES")
    # Opposite sign with an interval that excludes zero: not replicated, and counted as
    # "opposite sign" in the 2x2 classification. Both signs are in the table.
    row = comparison.loc[("eval", "mean")]
    assert (row["sign"], row["replica_sign"], row["same_sign"]) == ("positive", "negative", False)
    assert row["replica_excludes_zero"] and row["verdict"] == "not replicated"
    assert comparison.loc[("eval", "stats"), "verdict"] == "replicated"
    assert set(comparison["sign"]) == {"positive"}
    assert set(comparison.loc["itw", "verdict"]) == {"descriptive"}
    outcome = rep.outcome_table(comparison.reset_index()).set_index("test_set")
    assert outcome.loc["eval", ["n_in_rule", "n_replicated", "both_exclude_same_sign",
                                "both_exclude_opposite_sign", "reference_only"]].tolist() == [2, 1, 1, 1, 0]
    # An interval that includes zero: not replicated.
    replica.loc[(replica["test_set"] == "eval") & (replica["pooling"] == "stats"), "tau_ci_low"] = -0.2
    comparison = rep.compare_units(reference, rep.rank_units(replica)).set_index(["test_set", "pooling"])
    assert comparison.loc[("eval", "stats"), "verdict"] == "not replicated"
    # The difference of tau between two thresholds: the same rule.
    reference = rep.tau_difference_units(difference_table(), "trim15")
    flipped = difference_table()
    flipped[["estimate", "spk_ci_low", "spk_ci_high"]] = [-0.2, -0.4, -0.05]
    comparison = rep.compare_units(reference, rep.tau_difference_units(flipped, "trim15"))
    assert comparison["in_rule"].all() and set(comparison["verdict"]) == {"not replicated"}


def test_every_committed_unit_belongs_to_one_finding_of_the_plan():
    units = committed_units()
    assert len(units) == 830 and int(units["in_rule"].sum()) == 608
    ids = rep.finding_ids(units)
    counts = ids.value_counts().to_dict()
    assert counts == {**{f"F{k}": 24 for k in range(1, 13)}, "F13": 4, "F15": 36, "F16": 6, "F17": 30,
                      "F18": 78, "F19": 12, "F20": 48, "F21": 12, "F22": 12, "F23": 12, "F24": 16,
                      "F26": 156, rep.REMAINING: 120}
    assert [name for name, *_ in rep.FINDINGS][-1] == rep.REMAINING
    assert [name for name, kind, *_ in rep.FINDINGS if kind == rep.ABSENCE] == ["F7", "F16"]
    # The units of single findings, as the plan lists them.
    f16 = units[ids == "F16"]
    assert set(f16["quantity"]) == {"delta_test"} and set(f16["subset"]) == {rep.LONG_SUBSET}
    f17 = units[ids == "F17"]
    assert "delta_test" not in set(f17["quantity"]) and set(f17["subset"]) == {rep.LONG_SUBSET}
    assert set(units[ids == "F15"]["subset"]) == {rep.SHORT_SUBSET}
    assert set(units[ids == "F24"]["quantity"]) == {"kendall_tau", "tau_difference"}
    assert set(units[ids == "F13"]["condition"]) == {"trim40"}
    rest = units[ids == rep.REMAINING]
    assert set(rest["condition"]) == {"trim20", "trim15"} and set(rest["finding"]) == {"effect"}
    assert set(rest[rest["test_set"] == "eval"]["quantity"]) == {"delta_train", "delta_train_c", "delta_test_t"}
    assert set(rest[rest["test_set"] == "itw"]["quantity"]) == {"delta_train", "delta_train_c",
                                                                "delta_test_t", "delta_total"}
    # The subsets are those of the duration split.
    import numpy as np

    from ssl_silence import effects as E

    assert list(E.duration_subsets(np.array([1.0]))) == [rep.WHOLE_SET, rep.SHORT_SUBSET, rep.LONG_SUBSET]
    other = units.copy()
    other.loc[0, "finding"] = "something else"
    with pytest.raises(ValueError, match="1 units fit no finding"):
        rep.finding_ids(other)


def test_the_findings_table_counts_per_finding_and_leaves_the_absence_findings_uncounted():
    units = committed_units()
    replica = units.copy()
    # The second encoder: F1 loses its interval in half of the units, F7 gains one everywhere,
    # F2 changes its sign.
    ids = rep.finding_ids(units)
    f1 = (ids == "F1") & (units["pooling"] == "mean")
    replica.loc[f1, "spk_ci_low"], replica.loc[f1, "in_rule"] = -1.0, False
    f7 = ids == "F7"
    replica.loc[f7, ["estimate", "spk_ci_low", "spk_ci_high"]] = [0.5, 0.1, 0.9]
    replica.loc[f7, "sign"], replica.loc[f7, "in_rule"] = "positive", True
    f2 = ids == "F2"
    replica.loc[f2, ["estimate", "spk_ci_low", "spk_ci_high"]] = [-1.0, -2.0, -0.5]
    replica.loc[f2, "sign"] = "negative"
    comparison = rep.compare_units(units, replica)
    table = rep.findings_table(comparison).set_index("finding_id")
    assert list(table.index) == [name for name, *_ in rep.FINDINGS] + [rep.ALL_UNITS]
    assert list(table.columns) == rep.FINDING_COLUMNS[1:]
    assert table.loc["F1", ["n_units", "n_in_rule", "n_replicated", "reference_only"]].tolist() == [24, 24, 12, 12]
    assert table.loc["F2", ["n_replicated", "both_exclude_opposite_sign"]].tolist() == [0, 24]
    # F7: 3 of the 24 units are in the rule (2 positive, 1 negative); the finding is not counted.
    row = table.loc["F7"]
    assert row["type"] == rep.ABSENCE and pd.isna(row["n_replicated"])
    assert (row["n_in_rule"], row["both_exclude_same_sign"], row["both_exclude_opposite_sign"],
            row["replica_only"], row["neither"]) == (3, 2, 1, 21, 0)
    assert pd.isna(table.loc["F16", "n_replicated"]) and table.loc["F16", "neither"] == 6
    assert table.loc["F3", ["n_in_rule", "n_replicated", "neither"]].tolist() == [12, 12, 12]
    total = table.loc[rep.ALL_UNITS]
    assert (total["n_units"], total["n_in_rule"]) == (830, 608)
    assert total["n_replicated"] == int(comparison["replicated"].sum()) == 608 - 12 - 24 - 1
    assert (table[rep.OUTCOME_COUNTS].sum(axis=1) == table["n_units"]).all()
    assert table.iloc[:-1]["n_units"].sum() == 830
    absent = rep.absence_units(comparison)
    assert len(absent) == 24 + 6 and set(absent["finding_id"]) == {"F7", "F16"}
    assert list(absent.columns)[0] == "finding_id" and "replica_spk_ci_low" in absent.columns
    # The verdict of a unit of an absence finding stays in the unit-level table.
    assert set(absent[absent["in_rule"]]["verdict"]) == {"replicated", "not replicated"}


# The pre-registered files --------------------------------------------------------------------------

def test_the_pre_registered_files_have_their_recorded_sha256(tmp_path):
    recorded = rep.preregistered_files(str(REPO))
    assert list(recorded) == ["docs/analysis_plan_hubert.md", rep.PREREGISTERED_UNITS, rep.PREREGISTERED_SUMMARY]
    check = rep.preregistration_check(str(REPO))
    assert check["ok"].all() and len(check) == 3
    assert check["file"].tolist() == list(recorded) == check["registered_as"].tolist()
    # Another copy of the units file: equal, with other line endings, changed, missing.
    data = (REPO / rep.PREREGISTERED_UNITS).read_bytes().replace(b"\r\n", b"\n")
    same, crlf, changed = tmp_path / "same.csv", tmp_path / "crlf.csv", tmp_path / "changed.csv"
    same.write_bytes(data)
    crlf.write_bytes(data.replace(b"\n", b"\r\n"))
    changed.write_bytes(data.replace(b"positive", b"negative", 1))
    copies = {str(p): rep.PREREGISTERED_UNITS for p in (same, crlf, changed, tmp_path / "missing.csv")}
    check = rep.preregistration_check(str(REPO), copies)
    assert check["ok"].tolist() == [True, True, True, True, True, False, False]
    assert check["sha256"].iloc[-1] == "missing" and check["file"].iloc[3] == str(same)
    assert set(check["registered_as"].iloc[3:]) == {rep.PREREGISTERED_UNITS}
    with pytest.raises(KeyError, match="not a pre-registered file"):
        rep.preregistration_check(str(REPO), {str(same): "docs/conventions.md"})


def test_a_changed_plan_fails_the_check(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    for name in ("preregistration.md", "analysis_plan_hubert.md", "replication_units.csv",
                 "replication_units_summary.csv"):
        (docs / name).write_bytes((REPO / "docs" / name).read_bytes())
    assert rep.preregistration_check(str(tmp_path))["ok"].all()
    plan = docs / "analysis_plan_hubert.md"
    plan.write_bytes(plan.read_bytes().replace(b"factor 10", b"factor 20"))
    check = rep.preregistration_check(str(tmp_path))
    assert check["ok"].tolist() == [False, True, True]
    (docs / "preregistration.md").write_text("no table here\n", encoding="utf-8")
    with pytest.raises(ValueError, match="names no file"):
        rep.preregistration_check(str(tmp_path))
