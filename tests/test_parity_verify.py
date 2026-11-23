"""verify against the two comparison functions of the original code, and a
rehearsal of the score and table checks on the synthetic runs written by the
original code. Skipped without the original code.

The rehearsal builds a mapping of the kind that is kept on Drive: the
original run ids, score-set names, column names and labels are stated there
and nowhere in the functions that compare.
"""

from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd
import pytest

import world
from original_names import ORIGINAL_VALUES
from ssl_silence import analysis as AN
from ssl_silence import bootstrap as B
from ssl_silence import effects as E
from ssl_silence import ranking as R
from ssl_silence import training as T
from ssl_silence import verify as V

N_BOOT = 25
INTERVALS = {"estimate": "observed", "file_ci_low": "ci_low", "file_ci_high": "ci_high",
             "file_boot_se": "boot_se", "file_excludes_zero": "excludes_zero",
             "spk_ci_low": "spk_ci_low", "spk_ci_high": "spk_ci_high", "spk_boot_se": "spk_boot_se",
             "spk_excludes_zero": "spk_excludes_zero"}


def record(parity_record, name, differences):
    parity_record("verify", name, len(differences), max(differences))
    assert max(differences) == 0.0, f"{name}: max abs diff {max(differences)}"


# The two original comparison functions ---------------------------------------------------------

def test_score_files_side_by_side(original, parity_record, tmp_path):
    A, O = original("analysis"), original("train")
    rng = np.random.default_rng(0)
    labels, ids = (rng.random(50) < 0.5).astype(np.int64), np.array([f"f{k}" for k in range(50)])
    scores = rng.random(50).astype(np.float32)
    sets_a = {"dev": T.ScoreSet(scores, labels, ids), "eval": T.ScoreSet(scores[::-1].copy(), labels, ids),
              "itw": T.ScoreSet(scores, labels, ids), "other": T.ScoreSet(scores, labels, ids)}
    moved = scores.copy()
    moved[3] += np.float32(2e-5)
    sets_b = {"dev": T.ScoreSet(scores, labels, ids), "eval": T.ScoreSet(moved[::-1].copy(), labels, ids),
              "itw": T.ScoreSet(scores, labels, ids[::-1].copy()),
              "other": T.ScoreSet(scores, 1 - labels, ids)}
    for name, sets in (("a", sets_a), ("b", sets_b)):
        O.save_scores(str(tmp_path / f"{name}.npz"), {k: O.ScoreSet(s.scores, s.labels, s.ids)
                                                      for k, s in sets.items()})
    old = A.compare_score_files(str(tmp_path / "a.npz"), str(tmp_path / "b.npz")).set_index("set")
    differences = []
    for name in sets_a:
        new = V.compare_score_sets(sets_a[name], sets_b[name])
        reference = old.loc[name]
        a, b = new["max_abs_diff"], float(reference["max_abs_diff"])
        differences += [0.0 if (np.isnan(a) and np.isnan(b)) else abs(a - b),
                        0.0 if new["ids_equal"] == bool(reference["ids_equal"]) else np.inf,
                        0.0 if new["labels_equal"] == bool(reference["labels_equal"]) else np.inf,
                        float(abs(new["n"] - reference["n"]))]
    record(parity_record, "compare_score_sets", differences)


def test_tables_side_by_side(original, parity_record):
    A = original("analysis")
    rng = np.random.default_rng(1)
    keys = ["test_set", "layer"]
    old = pd.DataFrame({"test_set": np.repeat(["eval", "itw"], 12), "layer": list(range(1, 13)) * 2,
                        "value": rng.normal(size=24), "low": rng.normal(size=24),
                        "flag": rng.random(24) < 0.5})
    new = old.sample(frac=1.0, random_state=2).reset_index(drop=True)
    new["value"] = new["value"] + rng.normal(0, 1e-7, 24)
    new.loc[4, "low"] += 0.5
    mine = V.compare_tables(new, old, keys).set_index("column")
    reference = A.table_max_abs_diff(new, old, keys, ["value", "low"]).set_index("column")
    differences = [abs(mine.loc[c, "max_abs_diff"] - reference.loc[c, "max_abs_diff"])
                   for c in ("value", "low")]
    differences += [float(abs(mine.loc[c, "n_matched"] - reference.loc[c, "n_rows"]))
                    for c in ("value", "low")]
    assert mine.loc["value", "max_abs_diff"] > 0 and not mine.loc["low", "ok"]
    record(parity_record, "compare_tables", differences)


# Rehearsal ------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rehearsal(tmp_path_factory, original, original_function):
    """The synthetic runs written by both codes, the tables of the original
    analysis written as files, and the mapping that describes them."""
    base = tmp_path_factory.mktemp("rehearsal")
    runs_dir = world.write_runs(str(base / "new" / "runs"))
    old_root = base / "old"
    main_dir, threshold_dir = str(old_root / "main"), str(old_root / "threshold")
    feature_set = ORIGINAL_VALUES["unpadded_feature_set"]
    world.write_original_runs(original("train"), feature_set, main_dir, threshold_dir)
    old_names = world.OLD_CONDITIONS
    A = original("analysis")
    speakers = {"eval": world.speakers_of("eval"), "itw": world.speakers_of("itw")}
    layer_configs = original_function("analysis", "layer_configs")
    old_dirs = {"trim40": main_dir, "trim20": {"clean": main_dir, old_names["trim20"]: threshold_dir}}
    tables = old_root / "tables"
    tables.mkdir()
    cross = {"trim40": world.LAYERS, "trim20": world.CROSS_LAYERS}
    specs = {}
    for condition in ("trim40", "trim20"):
        b = old_names[condition]
        A.cross_effects_ci(layer_configs(cross[condition], world.POOLINGS), old_dirs[condition],
                           seeds=world.SEEDS, n_boot=N_BOOT, speaker_maps=speakers, progress=False,
                           conditions=("clean", b), include_total=True, aggregation="seed-mean"
                           ).to_csv(tables / f"effects_{b}.csv", index=False)
        specs[f"effects_2x2_{condition}"] = {
            "file": f"tables/effects_{b}.csv",
            "combine": {"effect": {"columns": ["effect", "fixed"], "values": {
                "test effect|train=clean": "delta_test", f"train effect|test={b}": "delta_train",
                f"{b} - clean|": "delta_total", "train effect|test=clean": "delta_train_c",
                f"test effect|train={b}": "delta_test_t", "interaction|": "interaction"}}},
            "columns": {"test_set": "test_set", "pooling": "pooling", "layer": "layer", **INTERVALS,
                        **{f"eer_{c}": f"eer_{c}" for c in E.CELLS}, "n_files": "n_files"},
            "constants": {"condition": condition},
            "keys": ["condition", "test_set", "pooling", "layer", "effect"]}
        summary, detail = A.layer_rank_agreement_ci(
            old_dirs[condition], conditions=("clean", b), layers=world.LAYERS, seeds=world.SEEDS,
            n_boot=N_BOOT, speaker_maps=speakers, progress=False, ci="both")
        summary.to_csv(tables / f"rank_{b}.csv", index=False)
        detail.to_csv(tables / f"rank_layers_{b}.csv", index=False)
        same = [c for c in summary.columns if not c.startswith("best_layer_")]
        specs[f"rank_agreement_{condition}"] = {
            "file": f"tables/rank_{b}.csv",
            "columns": {**{c: c for c in same}, "best_layer_clean": "best_layer_clean",
                        "best_layer_trimmed": f"best_layer_{b}"},
            "constants": {"condition": condition}, "keys": ["condition", "test_set", "pooling"]}
        specs[f"rank_agreement_layers_{condition}"] = {
            "file": f"tables/rank_layers_{b}.csv",
            "columns": {"test_set": "test_set", "pooling": "pooling", "layer": "layer",
                        "eer_clean": "eer_clean", "eer_trimmed": f"eer_{b}",
                        "p_best_clean": "p_best_clean", "p_best_trimmed": f"p_best_{b}"},
            "constants": {"condition": condition},
            "keys": ["condition", "test_set", "pooling", "layer"]}
    b = old_names["trim40"]
    A.per_attack_trimmed_minus_clean_ci(
        layer_configs(world.CROSS_LAYERS, world.POOLINGS), world.attacks_of_eval(), main_dir,
        seeds=world.SEEDS, n_boot=N_BOOT, speaker_map=speakers["eval"], progress=False,
        conditions=("clean", b), aggregation="seed-mean").to_csv(tables / "per_attack.csv", index=False)
    specs["per_attack_effect"] = {
        "file": "tables/per_attack.csv",
        "columns": {"pooling": "pooling", "layer": "layer", "attack": "attack", "estimate": "diff",
                    "eer_clean": "eer_clean", "eer_trimmed": f"eer_{b}",
                    **{k: v for k, v in INTERVALS.items() if k != "estimate"},
                    "n_bonafide": "n_bonafide", "n_spoof": "n_spoof",
                    "n_speakers_spoof": "n_speakers_spoof"},
        "constants": {"condition": "trim40"}, "keys": ["condition", "pooling", "layer", "attack"]}
    duration = A.itw_2x2_by_duration(main_dir, world.durations_of_itw(), conditions=("clean", b),
                                     poolings=world.POOLINGS, layers=world.CROSS_LAYERS,
                                     seeds=world.SEEDS)
    duration.to_csv(tables / "duration.csv", index=False)
    labels = {t: t.split("=")[0].strip() for t in duration.loc[duration["train"] == "delta", "test"]}
    specs["itw_by_duration"] = {
        "file": "tables/duration.csv",
        "combine": {"quantity": {"columns": ["train", "test"], "values": {
            "clean|clean": "eer_cc", f"clean|{b}": "eer_ct", f"{b}|clean": "eer_tc",
            f"{b}|{b}": "eer_tt", **{f"delta|{old}": new for old, new in labels.items()}}}},
        "columns": {"pooling": "pooling", "layer": "layer", "subset": "subset", "mean": "eer_mean",
                    "sd": "eer_sd", "n_files": "n_files", "n_bonafide": "n_bonafide",
                    "n_spoof": "n_spoof"},
        "constants": {"condition": "trim40"},
        "keys": ["condition", "pooling", "layer", "subset", "quantity"]}
    legacy = {
        "root": str(old_root),
        "forbidden": {"values": [feature_set, main_dir, threshold_dir],
                      "patterns": [rf"{world.ENCODER}_{re.escape(feature_set)}_L\d\d_.*"]},
        "conditions": dict(old_names),
        "runs": {world.ENCODER: {"unpadded": {
            "run_id": "{encoder}_" + feature_set + "_L{layer:02d}_{pooling}_none_{condition}{cross}_s{seed}",
            "cross": "_test-{test_condition}", "folders": ["main", "threshold"],
            "matched_sets": {"dev": "dev", "eval": "eval", "itw": "itw"},
            "cross_sets": {"dev": "dev_tc", "eval": "eval", "itw": "itw"}}}},
        "tables": specs}
    V.check_map(legacy)
    resamples = B.Resamples(B.ScoreStore(runs_dir, seeds=world.SEEDS), speakers, n_boot=N_BOOT)
    return {"runs_dir": runs_dir, "legacy": legacy, "resamples": resamples, "cross": cross,
            "checks": str(base / "checks")}


def test_rehearsal_of_the_score_check(rehearsal, parity_record):
    table = V.compare_runs(rehearsal["runs_dir"], rehearsal["legacy"])
    assert len(table) == 3 * sum(len(world.test_conditions(c, layer)) for c in world.CONDITIONS
                                 for layer in world.LAYERS) * len(world.POOLINGS) * len(world.SEEDS)
    V.write_check(table, "scores_wavlm-base_unpadded", rehearsal["checks"],
                  rehearsal["legacy"]["forbidden"])
    V.require_ok(table, "scores")
    assert table["exactly_zero"].all() and table["best_epoch_equal"].all()
    record(parity_record, "compare_runs (every score set of the synthetic runs)",
           table["max_abs_diff"].tolist())


def test_rehearsal_of_the_table_checks(rehearsal, parity_record):
    resamples, legacy = rehearsal["resamples"], rehearsal["legacy"]
    new = {"per_attack_effect": E.per_attack_effect_ci(resamples, "trim40", world.attacks_of_eval(),
                                                       layers=world.CROSS_LAYERS),
           "itw_by_duration": E.itw_by_duration(resamples, "trim40", world.durations_of_itw(),
                                                layers=world.CROSS_LAYERS)}
    for condition in ("trim40", "trim20"):
        new[f"effects_2x2_{condition}"] = E.effects_ci(resamples, condition,
                                                       rehearsal["cross"][condition])
        summary, detail = R.rank_agreement_ci(resamples, condition, layers=world.LAYERS)
        new[f"rank_agreement_{condition}"] = summary
        new[f"rank_agreement_layers_{condition}"] = detail
    # The original duration table has three of the six effects.
    duration = new["itw_by_duration"]
    new["itw_by_duration"] = duration[duration["quantity"].isin(
        ["eer_cc", "eer_ct", "eer_tc", "eer_tt", "delta_test", "delta_train", "delta_total"])]
    assert set(new) == set(legacy["tables"])
    checks = {name: V.compare_with_legacy(table, name, legacy) for name, table in new.items()}
    for name, check in checks.items():
        V.write_check(check, f"table_{name}", rehearsal["checks"], legacy["forbidden"])
        V.require_ok(check, name)
        assert (check["n_only_new"] == 0).all() and (check["n_only_old"] == 0).all(), name
    summary = V.summarise_checks(checks)
    assert summary["ok"].all() and len(summary) == 8
    assert sorted(os.listdir(rehearsal["checks"]))[-1].startswith("table_")
    numeric = pd.concat(checks.values())["max_abs_diff"].dropna()
    assert len(numeric) > 60
    record(parity_record, "compare_with_legacy (effects, rank agreement, per attack, duration)",
           numeric.tolist())


def test_the_rehearsal_notices_a_changed_original(rehearsal):
    legacy = rehearsal["legacy"]
    path = V.legacy_path(legacy, legacy["tables"]["per_attack_effect"]["file"])
    original_table = pd.read_csv(path)
    with open(path, "rb") as fh:
        content = fh.read()            # restored byte for byte: the default parser can change the last digit
    try:
        changed = original_table.copy()
        changed.loc[0, "spk_ci_low"] += 1e-6
        changed.iloc[1:].to_csv(path, index=False)          # one row removed
        new = E.per_attack_effect_ci(rehearsal["resamples"], "trim40", world.attacks_of_eval(),
                                     layers=world.CROSS_LAYERS)
        check = V.compare_with_legacy(new, "per_attack_effect", legacy)
        assert not check["ok"].any() and set(check["n_only_new"]) == {1}
        changed.to_csv(path, index=False)
        check = V.compare_with_legacy(new, "per_attack_effect", legacy).set_index("column")
        assert check["ok"].sum() == len(check) - 1 and not check.loc["spk_ci_low", "ok"]
        with pytest.raises(V.CheckFailed):
            V.require_ok(check.reset_index(), "per_attack_effect")
    finally:
        with open(path, "wb") as fh:
            fh.write(content)


def test_rehearsal_through_the_analysis_steps(rehearsal, tmp_path, parity_record):
    """The tables as the analysis notebook gets them (computed in steps, written and
    read back) against the tables of the original analysis."""
    legacy = rehearsal["legacy"]
    conditions = ("trim40", "trim20")
    tables = AN.ScoreTables(
        world.ENCODER, rehearsal["runs_dir"], str(tmp_path / "tables"),
        speaker_maps=rehearsal["resamples"].speaker_maps, attack_map=world.attacks_of_eval(),
        duration_map=world.durations_of_itw(), conditions=conditions, layers=world.LAYERS,
        cross_layers=rehearsal["cross"], detail_layers=world.CROSS_LAYERS, seeds=world.SEEDS,
        n_boot=N_BOOT, log=lambda line: None)
    tables.metrics()
    rank, rank_layers = tables.rank_agreement()
    checks = {}
    for name, table in (("effects_2x2", tables.effects()), ("rank_agreement", rank),
                        ("rank_agreement_layers", rank_layers)):
        checks.update(AN.compare_with_original(name, table, legacy, conditions=conditions))
    # One original table each, of 40 dB, under a name without the condition.
    for name, table in (("per_attack_effect", tables.per_attack_effect()),
                        ("itw_by_duration", tables.itw_by_duration())):
        assert set(table["condition"]) == set(conditions)
        checks.update(AN.compare_with_original(name, table[table["condition"] == "trim40"], legacy,
                                               conditions=(None,)))
    assert set(checks) == {f"table_{name}" for name in legacy["tables"]}
    for name, check in checks.items():
        V.write_check(check, name, str(tmp_path / "checks"), legacy["forbidden"])
        V.require_ok(check, name)
        assert (check["n_only_new"] == 0).all() and (check["n_only_old"] == 0).all(), name
    numeric = pd.concat(checks.values())["max_abs_diff"].dropna()
    assert len(numeric) > 60
    record(parity_record, "ScoreTables and compare_with_original (tables written in steps)",
           numeric.tolist())
