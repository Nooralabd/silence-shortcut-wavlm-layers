"""The analysis tables written in steps, and their comparison with original tables."""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

import synthetic
import world
from ssl_silence import analysis as A
from ssl_silence import config as cfg
from ssl_silence import effects as E
from ssl_silence import provenance
from ssl_silence import ranking as R
from ssl_silence import replication as rep
from ssl_silence import selection as S
from ssl_silence import shards
from ssl_silence import training as T
from ssl_silence import verify as V

N_BOOT = 40
UNIT_TABLES = ("effects_matched_cells", "rank_agreement_threshold_difference", "itw_by_duration_ci",
               "itw_by_duration_difference")
CONDITIONS = ("trim40", "trim20")
CROSS = {"trim40": world.LAYERS, "trim20": world.CROSS_LAYERS}
COMMIT = "0123456789abcdef0123456789abcdef01234567"


def write_padded_runs(runs_dir: str) -> str:
    """The runs of the padded comparison: the two configurations, every seed."""
    for condition, frames, pooling, layer in S.PADDED_CONFIGS:
        for seed in world.SEEDS:
            spec = T.RunSpec(world.ENCODER, "padded", condition, layer, pooling, seed, frames)
            sets = {T.set_name(split, condition): T.ScoreSet(
                world.scores_of(condition, condition, pooling, layer - 1, seed, split),
                world.labels_of(split), world.ids_of(split)) for split in world.SPLITS}
            T.record_run(spec, sets, runs_dir=runs_dir, overwrite=True,
                         fit_info={"n_train": 0, "in_dim": 0, "best_dev_eer_roc_corners": "nan",
                                   "best_epoch": 1, "epochs_run": 50, "device": "cpu",
                                   "gpu_name": "cpu"})
    return runs_dir


def score_tables(base, log=lambda line: None, **options) -> A.ScoreTables:
    return A.ScoreTables(
        world.ENCODER, str(base / "runs"), str(base / "tables"),
        speaker_maps={t: world.speakers_of(t) for t in cfg.TEST_SETS},
        attack_map=world.attacks_of_eval(), duration_map=world.durations_of_itw(),
        asv=S.load_asv(str(base / "asv")), padded_runs_dir=str(base / "padded"),
        conditions=CONDITIONS, layers=world.LAYERS, cross_layers=CROSS,
        detail_layers=world.CROSS_LAYERS, seeds=world.SEEDS, n_boot=N_BOOT, log=log, **options)


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    """Every table of the synthetic runs, computed once at a fixed commit."""
    base = tmp_path_factory.mktemp("analysis")
    world.write_runs(str(base / "runs"))
    write_padded_runs(str(base / "padded"))
    world.write_asv_scores(str(base / "asv"))
    features = {c: str(base / "features" / c) for c in CONDITIONS}
    for k, folder in enumerate(features.values()):
        synthetic.write_feature_condition(folder, seed=k)
    patch = pytest.MonkeyPatch()
    patch.setattr(provenance, "git_commit", lambda repo=None: COMMIT)
    lines: list[str] = []
    tables = score_tables(base, lines.append)
    out = {"seed_summary": tables.metrics(), "cells_2x2": tables.cells(),
           "padded_vs_unpadded": tables.padded_vs_unpadded(),
           "fallback_counts": tables.fallback_counts(features),
           "selected_configs_eer_ci": tables.selected_configs_eer_ci(),
           "itw_by_duration": tables.itw_by_duration(), "effects_2x2": tables.effects()}
    out["rank_agreement"], out["rank_agreement_layers"] = tables.rank_agreement()
    out["threshold_summary"] = tables.threshold_summary()
    out["per_attack_effect"] = tables.per_attack_effect()
    out["effects_matched_cells"] = tables.matched_total_effects()
    out["rank_agreement_threshold_difference"] = tables.tau_differences()
    out["itw_by_duration_ci"], out["itw_by_duration_difference"] = tables.itw_by_duration_ci()
    out["units"], out["units_summary"] = tables.replication_units()
    yield {"base": base, "tables": tables, "out": out, "lines": lines, "features": features}
    patch.undo()


def same(new: pd.DataFrame, direct: pd.DataFrame, keys: list[str]) -> None:
    """Every column of the directly computed table equals the written table, exactly."""
    check = V.compare_tables(new, direct, keys, tolerance=0.0)
    assert check["ok"].all(), check[~check["ok"]]
    assert list(new.columns) == list(direct.columns) and len(new) == len(direct)


# Layers of the configuration --------------------------------------------------------------------

def test_conditions_and_cross_layers_come_from_the_experiment_configuration():
    assert A.trained_conditions("wavlm-base") == ("trim40", "trim20", "trim15")
    assert A.cross_layers_of("wavlm-base", "trim40") == tuple(range(1, 13))
    assert A.cross_layers_of("wavlm-base", "trim20") == (3, 4, 5) == A.cross_layers_of("wavlm-base", "trim15")
    assert A.cross_layers_of("hubert-base", "trim15") == tuple(range(1, 13))
    assert A.trained_conditions("wavlm-base", "padded") == ("trim40",)


def test_the_default_layers_are_those_of_the_configuration(setup):
    base = setup["base"]
    tables = A.ScoreTables(world.ENCODER, str(base / "runs"), str(base / "other"), speaker_maps={})
    assert tables.conditions == ("trim40", "trim20", "trim15") and tables.layers == tuple(range(1, 13))
    assert tables.cross_layers == {"trim40": tuple(range(1, 13)), "trim20": (3, 4, 5), "trim15": (3, 4, 5)}
    assert tables.detail_layers == (3, 4, 5) and tables.n_boot == cfg.N_BOOT
    assert not os.path.exists(base / "other")                 # nothing is written before a table is asked for


# The tables are those of the functions ------------------------------------------------------------

def test_the_written_tables_equal_the_tables_of_the_functions(setup):
    out, resamples = setup["out"], setup["tables"].resamples
    runs, padded = str(setup["base"] / "runs"), str(setup["base"] / "padded")
    asv = S.load_asv(str(setup["base"] / "asv"))
    per_run = pd.concat([S.metrics_per_run(runs, asv), S.metrics_per_run(padded, asv)], ignore_index=True)
    summary = S.seed_summary(per_run)
    same(out["seed_summary"], summary, S.CELL_KEYS)
    assert set(out["seed_summary"]["extraction"]) == {"unpadded", "padded"}
    assert set(out["seed_summary"]["frames"]) == {"", "all-frames", "real-frames"}
    unpadded = summary[summary["extraction"] == "unpadded"]
    cells = pd.concat([E.cells_table(unpadded, c) for c in CONDITIONS], ignore_index=True)
    same(out["cells_2x2"], cells, ["condition", "pooling", "layer", "cell"])
    assert out["cells_2x2"].groupby("condition")["layer"].nunique().to_dict() == {"trim20": 2, "trim40": 4}
    effects = pd.concat([E.effects_ci(resamples, c, CROSS[c]) for c in CONDITIONS], ignore_index=True)
    same(out["effects_2x2"], effects, ["condition", "test_set", "pooling", "layer", "effect"])
    ranks = [R.rank_agreement_ci(resamples, c, layers=world.LAYERS) for c in CONDITIONS]
    same(out["rank_agreement"], pd.concat([r[0] for r in ranks], ignore_index=True),
         ["condition", "test_set", "pooling"])
    same(out["rank_agreement_layers"], pd.concat([r[1] for r in ranks], ignore_index=True),
         ["condition", "test_set", "pooling", "layer"])
    attack = pd.concat([E.per_attack_effect_ci(resamples, c, world.attacks_of_eval(),
                                               layers=world.CROSS_LAYERS) for c in CONDITIONS],
                       ignore_index=True)
    same(out["per_attack_effect"], attack, ["condition", "pooling", "layer", "attack"])
    duration = pd.concat([E.itw_by_duration(resamples, c, world.durations_of_itw(),
                                            layers=world.CROSS_LAYERS) for c in CONDITIONS],
                         ignore_index=True)
    same(out["itw_by_duration"], duration, ["condition", "pooling", "layer", "subset", "quantity"])
    configs = S.selected_configs(unpadded, world.ENCODER)
    same(out["selected_configs_eer_ci"], E.selected_configs_eer_ci(resamples, configs),
         ["condition", "pooling", "layer", "test_set"])
    assert out["effects_2x2"]["n_boot"].eq(N_BOOT).all()


def test_the_padded_comparison_and_the_threshold_summary(setup):
    out, features = setup["out"], setup["features"]
    padded = out["padded_vs_unpadded"].set_index(["config", "extraction"])
    assert len(padded) == 6 and padded["eval_tdcf_mean"].notna().all()
    row = padded.loc[("clean/stats/L4", "unpadded - padded")]
    assert row["itw_eer_mean"] == pytest.approx(
        padded.loc[("clean/stats/L4", "unpadded"), "itw_eer_mean"]
        - padded.loc[("clean/stats/L4", "padded"), "itw_eer_mean"])
    counts = out["fallback_counts"]
    assert list(counts.columns)[:3] == ["condition", "split", "label"] and len(counts) == 2 * 4 * 3
    same(counts[counts["condition"] == "trim20"].drop(columns="condition"),
         shards.fallback_counts(features["trim20"]).astype({"label": str}), ["split", "label"])
    thresholds = out["threshold_summary"].set_index(["condition", "pooling"])
    assert len(thresholds) == 4 and thresholds["top_db"].to_dict()[("trim20", "mean")] == 20
    effects = out["effects_2x2"].set_index(["condition", "test_set", "pooling", "layer", "effect"])
    assert thresholds.loc[("trim20", "stats"), "interaction_itw_L4_spk_ci_low"] == \
        effects.loc[("trim20", "itw", "stats", 4, "interaction"), "spk_ci_low"]
    # The dev-selected layer is that of the unpadded runs, whatever the padded runs score.
    dev = S.select_on_dev(setup["tables"].summary()).set_index(["condition", "pooling"])["layer"]
    assert thresholds["dev_selected_layer"].to_dict() == {k: dev[k] for k in thresholds.index}
    assert thresholds.loc[("trim40", "mean"), "n_fallback_eval"] == \
        counts[(counts["condition"] == "trim40") & (counts["split"] == "eval")
               & (counts["label"] == "all")]["n_fallback"].iloc[0]


def test_every_table_has_a_meta_file_with_its_inputs(setup):
    folder = setup["base"] / "tables"
    names = {"metrics_per_run", "seed_summary", *A.ORIGINAL_TABLES, *UNIT_TABLES,
             "replication_units", "replication_units_summary"}
    assert {p.stem for p in folder.glob("*.csv")} == names
    meta = json.loads((folder / "effects_2x2.meta.json").read_text(encoding="utf-8"))
    assert meta["git_commit"] == COMMIT and meta["inputs"]["n_boot"] == N_BOOT
    assert meta["inputs"]["cross_layers"] == {"trim40": [1, 2, 3, 4], "trim20": [3, 4]}
    assert set(meta["inputs"]["runs_csv_sha256"]) == {"unpadded", "padded"}
    assert meta["sha256"] == provenance.sha256_file(folder / "effects_2x2.csv")
    parts = {p.stem for p in (folder / A.PARTS).glob("*.csv")}
    assert parts == {"metrics_per_run", "selected_configs_eer_ci"} | {
        f"{name}_{c}" for c in CONDITIONS for name in (
            "fallback_counts", "effects_2x2", "rank_agreement", "rank_agreement_layers",
            "per_attack_effect")} | {
        "effects_matched_cells_trim20", "rank_agreement_threshold_difference_trim20",
        "itw_by_duration_ci_trim40", "itw_by_duration_difference_trim40"}


# Parts after a restart ----------------------------------------------------------------------------

def test_a_part_is_reused_at_the_same_commit_and_computed_again_otherwise(setup, monkeypatch):
    base, first = setup["base"], setup["tables"]
    steps = first.steps_table().set_index("step")
    assert set(steps["status"]) == {"computed"} and "effects_2x2_trim40: computed in" in "\n".join(setup["lines"])
    calls = []
    real = E.effects_ci
    monkeypatch.setattr(E, "effects_ci", lambda *a, **k: calls.append(a[1]) or real(*a, **k))
    lines: list[str] = []
    again = score_tables(base, lines.append)
    table = again.effects()
    assert calls == [] and [s["status"] for s in again.steps] == ["reused", "reused"]
    assert "effects_2x2_trim20: reused" in lines[1]
    same(table, setup["out"]["effects_2x2"], ["condition", "test_set", "pooling", "layer", "effect"])
    # A changed part file is not reused.
    path = again.path("effects_2x2_trim20", part=True)
    original = open(path, encoding="utf-8").read()
    try:
        open(path, "w", encoding="utf-8", newline="\n").write(original.replace("delta_total", "delta_other", 1))
        score_tables(base).effects()
        assert calls == ["trim20"]
        assert open(path, encoding="utf-8").read() == original
        # Another commit, a working tree with changes, or no repository: computed again.
        for commit in ("f" * 40, COMMIT + "-dirty", "unknown"):
            monkeypatch.setattr(provenance, "git_commit", lambda repo=None, commit=commit: commit)
            score_tables(base).effects()
        assert calls == ["trim20"] + ["trim40", "trim20"] * 3
    finally:
        monkeypatch.setattr(provenance, "git_commit", lambda repo=None: COMMIT)
        score_tables(base).effects()                               # the parts of the fixture's commit again


def test_a_part_is_computed_again_when_the_runs_or_the_resamples_change(setup, monkeypatch, tmp_path):
    base = setup["base"]
    calls = []
    real = E.per_attack_effect_ci
    monkeypatch.setattr(E, "per_attack_effect_ci", lambda *a, **k: calls.append(a[1]) or real(*a, **k))
    other = A.ScoreTables(world.ENCODER, str(base / "runs"), str(base / "tables"), speaker_maps={},
                          conditions=CONDITIONS, layers=world.LAYERS, cross_layers=CROSS,
                          detail_layers=world.CROSS_LAYERS, seeds=world.SEEDS, n_boot=N_BOOT + 1,
                          padded_runs_dir=str(base / "padded"), log=lambda line: None)
    assert {k for k in other.inputs if other.inputs[k] != setup["tables"].inputs[k]} == {"n_boot"}
    runs = base / "runs" / "runs.csv"
    text = runs.read_text(encoding="utf-8")
    try:
        runs.write_text(text + text.splitlines()[-1] + "\n", encoding="utf-8")     # a run recorded again
        changed = score_tables(base)
        assert changed.inputs["runs_csv_sha256"]["unpadded"] != setup["tables"].inputs["runs_csv_sha256"]["unpadded"]
        changed.per_attack_effect()
        assert calls == list(CONDITIONS)
    finally:
        runs.write_text(text, encoding="utf-8")
        score_tables(base).per_attack_effect()
    assert calls == list(CONDITIONS) * 2


def test_a_table_without_its_input_raises(setup):
    base = setup["base"]
    tables = A.ScoreTables(world.ENCODER, str(base / "runs"), str(base / "tables"), speaker_maps={},
                           conditions=CONDITIONS, layers=world.LAYERS, cross_layers=CROSS,
                           seeds=world.SEEDS, n_boot=N_BOOT, log=lambda line: None)
    with pytest.raises(ValueError, match="attack"):
        tables.per_attack_effect()
    with pytest.raises(ValueError, match="durations"):
        tables.itw_by_duration()
    with pytest.raises(ValueError, match="padded runs"):
        tables.padded_vs_unpadded()


# Comparison with original tables -----------------------------------------------------------------

OLD_CONDITION = {"trim40": "cut-a", "trim20": "cut-b"}


def original_world(setup, folder) -> dict:
    """Original tables written from the new ones, one file per condition with
    other column names, and the mapping that describes them."""
    out = setup["out"]
    os.makedirs(folder / "old", exist_ok=True)
    specs = {}

    def add(name, table, keys, condition=None, drop=()):
        original = name if condition is None else f"{name}_{condition}"
        part = table if condition is None else table[table["condition"] == condition]
        part = part.drop(columns=["encoder", "n_boot", *drop], errors="ignore")
        columns = [c for c in part.columns if c != "condition" or condition is None]
        part[columns].rename(columns={c: f"old_{c}" for c in columns}).to_csv(
            folder / "old" / f"{original}.csv", index=False)
        specs[original] = {"file": f"old/{original}.csv", "columns": {c: f"old_{c}" for c in columns},
                           "keys": keys, **({} if condition is None else {"constants": {"condition": condition}})}

    for c in CONDITIONS:
        add("effects_2x2", out["effects_2x2"], ["condition", "test_set", "pooling", "layer", "effect"], c)
        add("rank_agreement", out["rank_agreement"], ["condition", "test_set", "pooling"], c)
        add("per_attack_effect", out["per_attack_effect"], ["condition", "pooling", "layer", "attack"], c)
        add("fallback_counts", out["fallback_counts"], ["split", "label"], c)
        add("threshold_summary", out["threshold_summary"], ["condition", "pooling"], c)
    add("cells_2x2", out["cells_2x2"], ["condition", "pooling", "layer", "cell"], "trim40")
    add("rank_agreement_layers", out["rank_agreement_layers"],
        ["condition", "test_set", "pooling", "layer"], "trim40")
    duration = out["itw_by_duration"]
    add("itw_by_duration", duration[duration["quantity"].isin(A.ORIGINAL_DURATION_QUANTITIES)],
        ["condition", "pooling", "layer", "subset", "quantity"], "trim40")
    add("selected_configs_eer_ci", out["selected_configs_eer_ci"],
        ["condition", "pooling", "layer", "test_set"], drop=("config", "selection"))
    add("padded_vs_unpadded", out["padded_vs_unpadded"], ["config", "extraction"], drop=("frames", "n_seeds"))
    return {"root": str(folder), "forbidden": {"values": [], "patterns": []},
            "conditions": {"clean": "clean", **OLD_CONDITION}, "tables": specs}


def test_original_table_names():
    names = A.original_table_names()
    assert sum(len(v) for v in names.values()) == 20
    assert names["effects_2x2"] == ["effects_2x2_trim40", "effects_2x2_trim20", "effects_2x2_trim15"]
    assert names["padded_vs_unpadded"] == ["padded_vs_unpadded"]
    assert names["rank_agreement_layers"] == ["rank_agreement_layers_trim40"]
    # The cheap tables come first: a difference shows before the long steps.
    assert list(names)[:5] == ["cells_2x2", "padded_vs_unpadded", "fallback_counts",
                               "selected_configs_eer_ci", "itw_by_duration"]


def test_every_table_equals_its_original(setup, tmp_path):
    legacy = original_world(setup, tmp_path)
    conditions = {name: [c for c in found if c in (None, *CONDITIONS)]
                  for name, found in A.ORIGINAL_TABLES.items()}
    checks = {}
    for name, table in setup["out"].items():
        if name in A.ORIGINAL_TABLES:
            checks.update(A.compare_with_original(name, table, legacy, conditions=conditions[name]))
    assert len(checks) == 2 * 5 + 5 and all(name.startswith("table_") for name in checks)
    summary = V.summarise_checks(checks)
    assert summary["ok"].all() and (summary["max_abs_diff"] == 0).all(), summary
    # The duration table of this package has six effects; the original has three of them.
    assert setup["out"]["itw_by_duration"]["quantity"].nunique() == 10
    assert pd.read_csv(tmp_path / "old/itw_by_duration_trim40.csv")["old_quantity"].nunique() == 7
    assert set(checks["table_itw_by_duration_trim40"]["n_only_new"]) == {0}
    for check in checks.values():
        V.write_check(check, "x", str(tmp_path / "checks"), {"values": list(OLD_CONDITION.values()), "patterns": []})


def test_a_changed_or_missing_original_row_fails(setup, tmp_path):
    legacy = original_world(setup, tmp_path)
    path = tmp_path / "old/effects_2x2_trim20.csv"
    table = pd.read_csv(path, float_precision="round_trip")
    table.loc[3, "old_spk_ci_high"] += 1e-8
    table.to_csv(path, index=False)
    checks = A.compare_with_original("effects_2x2", setup["out"]["effects_2x2"], legacy, conditions=CONDITIONS)
    assert checks["table_effects_2x2_trim40"]["ok"].all()
    failed = checks["table_effects_2x2_trim20"].set_index("column")
    assert not failed.loc["spk_ci_high", "ok"] and failed["ok"].sum() == len(failed) - 1
    assert failed.loc["spk_ci_high", "max_abs_diff"] == pytest.approx(1e-8, rel=1e-3)
    table.iloc[1:].to_csv(path, index=False)
    checks = A.compare_with_original("effects_2x2", setup["out"]["effects_2x2"], legacy, conditions=("trim20",))
    assert list(checks) == ["table_effects_2x2_trim20"] and not checks["table_effects_2x2_trim20"]["ok"].any()
    # A quantity that the original duration table has must be in the new table.
    duration = setup["out"]["itw_by_duration"]
    check = A.compare_with_original("itw_by_duration", duration[duration["quantity"] != "delta_train"],
                                    legacy)["table_itw_by_duration_trim40"]
    assert not check["ok"].any() and set(check["n_only_old"]) == {2 * 2 * 3}
    with pytest.raises(KeyError):
        A.compare_with_original("effects_2x2", setup["out"]["effects_2x2"], legacy)     # no 15 dB table in this mapping


def test_the_migrated_copy_is_read_when_its_folder_is_given(setup, tmp_path):
    legacy = original_world(setup, tmp_path)
    copies = tmp_path / "copies" / "tables"
    copies.mkdir(parents=True)
    source = tmp_path / "old/cells_2x2_trim40.csv"
    (copies / "cells_2x2_trim40.csv").write_bytes(source.read_bytes())
    source.unlink()
    cells = setup["out"]["cells_2x2"]
    with pytest.raises(FileNotFoundError):
        A.compare_with_original("cells_2x2", cells, legacy)
    check = A.compare_with_original("cells_2x2", cells, legacy, str(tmp_path / "copies"))["table_cells_2x2_trim40"]
    assert check["ok"].all() and set(check["table"]) == {"cells_2x2_trim40"}
    assert set(check["n_matched"]) == {4 * 2 * 4}                 # the rows of the other condition are left out
    assert np.isfinite(check["max_abs_diff"].dropna()).all()


# Parts of another commit ---------------------------------------------------------------------------

def test_parts_of_a_named_commit_are_reused_at_another_commit(setup, monkeypatch):
    base = setup["base"]
    calls = []
    real = E.effects_ci
    monkeypatch.setattr(E, "effects_ci", lambda *a, **k: calls.append(a[1]) or real(*a, **k))
    later = "a" * 40
    monkeypatch.setattr(provenance, "git_commit", lambda repo=None: later)
    try:
        lines: list[str] = []
        tables = score_tables(base, lines.append, reuse_commits=(COMMIT,))
        table = tables.effects()
        assert calls == [] and "reused (computed in" in lines[0] and f"at commit {COMMIT[:7]})" in lines[0]
        steps = tables.steps_table()
        assert set(steps["status"]) == {"reused"} and set(steps["git_commit"]) == {COMMIT}
        same(table, setup["out"]["effects_2x2"], ["condition", "test_set", "pooling", "layer", "effect"])
        # The stacked table is written again, at the later commit; the parts keep theirs.
        meta = json.loads((base / "tables" / "effects_2x2.meta.json").read_text(encoding="utf-8"))
        part = json.loads((base / "tables" / A.PARTS / "effects_2x2_trim40.meta.json").read_text(encoding="utf-8"))
        assert meta["git_commit"] == later and part["git_commit"] == COMMIT
        # A commit that is not named is not reused, and a changed file is not reused either.
        score_tables(base, reuse_commits=("b" * 40,)).per_attack_effect()
        path = tables.path("effects_2x2_trim20", part=True)
        text = open(path, encoding="utf-8").read()
        open(path, "w", encoding="utf-8", newline="\n").write(text.replace("delta_total", "delta_other", 1))
        score_tables(base, reuse_commits=(COMMIT,)).effects()
        assert calls == ["trim20"]
        with pytest.raises(ValueError, match="full commit hashes"):
            score_tables(base, reuse_commits=(COMMIT[:7],))
    finally:
        monkeypatch.setattr(provenance, "git_commit", lambda repo=None: COMMIT)
        score_tables(base).effects()
        score_tables(base).per_attack_effect()


# Units of the replication rule ----------------------------------------------------------------------

def test_the_tables_of_the_units_equal_the_tables_of_the_functions(setup):
    out, tables = setup["out"], setup["tables"]
    resamples = tables.resamples
    assert tables.threshold_conditions == ("trim20",)
    same(out["effects_matched_cells"], E.total_effect_ci(resamples, "trim20", world.LAYERS),
         ["condition", "test_set", "pooling", "layer", "effect"])
    same(out["rank_agreement_threshold_difference"],
         R.tau_difference_ci(resamples, "trim20", layers=world.LAYERS),
         ["condition", "test_set", "pooling"])
    by_subset, difference = E.itw_by_duration_ci(resamples, "trim40", world.durations_of_itw(),
                                                 layers=world.CROSS_LAYERS)
    same(out["itw_by_duration_ci"], by_subset, ["condition", "pooling", "layer", "subset", "effect"])
    same(out["itw_by_duration_difference"], difference, ["condition", "pooling", "layer", "effect"])
    assert set(out["itw_by_duration_ci"]["condition"]) == {"trim40"}


def test_replication_units_of_the_synthetic_runs(setup):
    out = setup["out"]
    units, summary = out["units"], out["units_summary"]
    n_layers, n_cross = len(world.LAYERS), len(world.CROSS_LAYERS)
    main = 6 * 2 * n_layers * 2 + 4 + len(world.ATTACKS) * n_cross * 2 + 6 * 2 * n_cross * 2
    threshold = 2 * n_layers * 2 + 5 * 2 * n_cross * 2 + 4 + 4 + len(world.ATTACKS) * n_cross * 2
    assert len(units) == main + threshold == 236 and not units.duplicated(rep.KEYS).any()
    assert units.groupby("condition").size().to_dict() == {"trim20": threshold, "trim40": main}
    assert summary["n_units"].sum() == len(units) and summary["n_in_rule"].sum() == units["in_rule"].sum()
    # A total effect at 20 dB of a layer without cross-condition cells comes from the matched cells.
    matched = out["effects_matched_cells"].set_index(["test_set", "pooling", "layer"])
    row = units[(units["finding"] == "effect") & (units["condition"] == "trim20") & (units["layer"] == 1)
                & (units["test_set"] == "itw") & (units["pooling"] == "stats")]
    assert row["quantity"].tolist() == ["delta_total"]
    assert row["spk_ci_low"].iloc[0] == matched.loc[("itw", "stats", 1), "spk_ci_low"]
    assert bool(row["in_rule"].iloc[0]) == bool(matched.loc[("itw", "stats", 1), "spk_excludes_zero"])
    # The files hold what was returned.
    folder = setup["base"] / "tables"
    written = A.read_table(str(folder / "replication_units.csv"))
    assert list(written.columns) == rep.UNIT_COLUMNS and len(written) == len(units)
    assert written["estimate"].tolist() == units["estimate"].tolist()
    assert written["in_rule"].tolist() == units["in_rule"].tolist()
    assert len(A.read_table(str(folder / "replication_units_summary.csv"))) == len(summary)


# A second encoder: the same tables, the paired differences and the replication rule -------------

OTHER = "hubert-base"


def second_tables(base, folder="tables_other", **options) -> A.ScoreTables:
    settings = dict(speaker_maps={t: world.speakers_of(t) for t in cfg.TEST_SETS},
                    attack_map=world.attacks_of_eval(), duration_map=world.durations_of_itw(),
                    asv=S.load_asv(str(base / "asv")), conditions=CONDITIONS, layers=world.LAYERS,
                    cross_layers=CROSS, detail_layers=world.CROSS_LAYERS, seeds=world.SEEDS, n_boot=N_BOOT,
                    log=lambda line: None)
    return A.ScoreTables(OTHER, str(base / "runs_other"), str(base / folder), **{**settings, **options})


@pytest.fixture(scope="module")
def second(setup, tmp_path_factory):
    """The tables of a second encoder next to those of the first, as the analysis
    notebook computes them (at the commit of ``setup``)."""
    base = setup["base"]
    world.write_runs(str(base / "runs_other"), encoder=OTHER)
    tables, reference = second_tables(base), score_tables(base)
    out = {"seed_summary": tables.metrics(), "cells_2x2": tables.cells(),
           "fallback_counts": tables.fallback_counts(setup["features"]),
           "selected_configs_eer_ci": tables.selected_configs_eer_ci(all_conditions=True),
           "dev_selected": tables.dev_selected(),
           "itw_by_duration": tables.itw_by_duration((rep.MAIN_CONDITION,)),
           "exploratory": tables.itw_by_duration_thresholds(), "effects_2x2": tables.effects()}
    out["rank_agreement"], out["rank_agreement_layers"] = tables.rank_agreement()
    out["threshold_summary"] = tables.threshold_summary()
    out["per_attack_effect"] = tables.per_attack_effect()
    out["effects_matched_cells"] = tables.matched_total_effects()
    out["rank_agreement_threshold_difference"] = tables.tau_differences()
    out["itw_by_duration_ci"], out["itw_by_duration_difference"] = tables.itw_by_duration_ci()
    out["units"], out["units_summary"] = tables.replication_units()
    out["selected_layers"] = tables.selected_layer_tables()
    out["encoder_effects"] = tables.encoder_differences(reference)
    out["encoder_tau"] = tables.encoder_tau_differences(reference)
    reference_units = str(base / "tables" / "replication_units.csv")
    out.update(tables.replication_outcome(reference_units))
    return {"base": base, "tables": tables, "reference": reference, "out": out,
            "reference_units": reference_units}


def test_the_tables_of_a_second_encoder_are_those_of_the_same_functions(second):
    out, resamples = second["out"], second["tables"].resamples
    assert set(out["seed_summary"]["encoder"]) == {OTHER} and set(out["seed_summary"]["extraction"]) == {"unpadded"}
    effects = pd.concat([E.effects_ci(resamples, c, CROSS[c]) for c in CONDITIONS], ignore_index=True)
    same(out["effects_2x2"], effects, ["condition", "test_set", "pooling", "layer", "effect"])
    ranks = [R.rank_agreement_ci(resamples, c, layers=world.LAYERS) for c in CONDITIONS]
    same(out["rank_agreement"], pd.concat([r[0] for r in ranks], ignore_index=True),
         ["condition", "test_set", "pooling"])
    same(out["rank_agreement_threshold_difference"],
         R.tau_difference_ci(resamples, "trim20", layers=world.LAYERS), ["condition", "test_set", "pooling"])
    # No reference configuration: the dev-selected layer of every condition and pooling.
    selected = out["selected_configs_eer_ci"]
    assert set(selected["selection"]) == {"dev-selected"}
    assert set(selected["condition"]) == {"clean", *CONDITIONS}
    dev = out["dev_selected"]
    same(dev, S.select_on_dev(second["tables"].summary()), ["condition", "pooling"])
    assert len(dev) == 3 * 2 and {"dev_eer_mean", "eval_tdcf_mean", "eval_eer_mean", "itw_eer_mean"} <= set(dev.columns)
    assert sorted(set(zip(selected["condition"], selected["pooling"], selected["layer"]))) == \
        sorted(set(zip(dev["condition"], dev["pooling"], dev["layer"])))
    # The other encoder has other scores: its tables are not those of the first.
    first = second["reference"].resamples
    assert not np.array_equal(E.effects_ci(first, "trim20", (3,))["estimate"].to_numpy(),
                              effects[(effects["condition"] == "trim20") & (effects["layer"] == 3)]["estimate"].to_numpy())


def test_the_duration_table_of_the_plan_and_the_exploratory_one(second):
    out, tables = second["out"], second["tables"]
    assert set(out["itw_by_duration"]["condition"]) == {"trim40"}
    exploratory = out["exploratory"]
    assert set(exploratory["condition"]) == {"trim20"} and list(exploratory.columns) == list(out["itw_by_duration"].columns)
    same(exploratory, E.itw_by_duration(tables.resamples, "trim20", world.durations_of_itw(), layers=world.CROSS_LAYERS),
         ["condition", "pooling", "layer", "subset", "quantity"])
    path = second["base"] / "tables_other" / A.EXPLORATORY / "exploratory_itw_by_duration_thresholds.csv"
    assert path.exists() and tables.exploratory_path("exploratory_itw_by_duration_thresholds") == str(path)
    meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta["exploratory"] is True and meta["git_commit"] == COMMIT
    assert not (second["base"] / "tables_other" / "exploratory_itw_by_duration_thresholds.csv").exists()
    # A table is exploratory only under a listed name with the prefix.
    for name in ("itw_by_duration_thresholds", "exploratory_something", "effects_2x2"):
        with pytest.raises(ValueError, match="EXPLORATORY_TABLES"):
            tables.write_exploratory(name, exploratory)
    assert all(name.startswith(A.EXPLORATORY_PREFIX) for name in A.EXPLORATORY_TABLES)
    assert not set(A.EXPLORATORY_TABLES) & (set(A.PLAN_TABLES) | set(A.SELECTED_LAYER_TABLES))


def test_the_additional_tables_exist_for_a_dev_selected_layer_outside_the_detail_layers(second):
    # The main tables of the fixture: whatever was selected, the check of the written tables agrees.
    out = second["out"]
    outside = tuple(sorted(set(out["dev_selected"]["layer"]) - set(world.CROSS_LAYERS)))
    assert second["tables"].selected_layers_outside_detail() == outside
    assert set(out["selected_layers"]) == (set(A.SELECTED_LAYER_TABLES) if outside else set())
    # Tables whose detail layer is layer 1 only: every other dev-selected layer is outside.
    tables = second_tables(second["base"], folder="tables_layers", detail_layers=(1,))
    tables.metrics()
    outside = tables.selected_layers_outside_detail()
    assert outside and outside == tuple(sorted(set(tables.dev_selected()["layer"]) - {1}))
    extra = tables.selected_layer_tables()
    assert set(extra) == set(A.SELECTED_LAYER_TABLES)
    attack = extra["per_attack_effect_dev_selected_layers"]
    assert set(attack["layer"]) == set(outside) and set(attack["condition"]) == set(CONDITIONS)
    direct = pd.concat([E.per_attack_effect_ci(tables.resamples, c, world.attacks_of_eval(), layers=outside)
                        for c in CONDITIONS], ignore_index=True)
    same(attack, direct, ["condition", "pooling", "layer", "attack"])
    by_subset, difference = E.itw_by_duration_ci(tables.resamples, "trim40", world.durations_of_itw(), layers=outside)
    same(extra["itw_by_duration_ci_dev_selected_layers"], by_subset, ["condition", "pooling", "layer", "subset", "effect"])
    same(extra["itw_by_duration_difference_dev_selected_layers"], difference, ["condition", "pooling", "layer", "effect"])
    duration = extra["itw_by_duration_dev_selected_layers"]
    assert set(duration["layer"]) == set(outside) and set(duration["condition"]) == {"trim40"}
    # The tables of the detail layers have other names, and the meta names the layers.
    folder = second["base"] / "tables_layers"
    assert not (folder / "per_attack_effect.csv").exists()
    meta = json.loads((folder / "per_attack_effect_dev_selected_layers.meta.json").read_text(encoding="utf-8"))
    assert meta["inputs"]["detail_layers"] == list(outside)
    part = json.loads((folder / A.PARTS / "itw_by_duration_ci_dev_selected_layers_trim40.meta.json")
                      .read_text(encoding="utf-8"))
    assert part["inputs"]["detail_layers"] == list(outside)
    selected = tables.written_tables_check()
    assert selected[selected["kind"].str.contains("dev-selected")]["ok"].all()
    # With every layer among the detail layers nothing is added, and the tables must not exist.
    every = second_tables(second["base"], folder="tables_layers", detail_layers=world.LAYERS)
    assert every.selected_layers_outside_detail() == () and every.selected_layer_tables() == {}
    check = every.written_tables_check()
    assert not check[check["kind"].str.contains("dev-selected")]["ok"].any()


def test_the_resamples_of_the_two_encoders_are_checked(second, tmp_path):
    check = A.paired_resamples_check(second["reference"], second["tables"])
    assert check["ok"].all() and len(check) == 4
    assert check[["test_set", "level"]].to_numpy().tolist() == [["eval", "file"], ["eval", "spk"],
                                                                ["itw", "file"], ["itw", "spk"]]
    assert (check[f"sha256_{world.ENCODER}"] == check[f"sha256_{OTHER}"]).all()
    assert check[f"sha256_{OTHER}"].nunique() == 4 and set(check["n_boot"]) == {N_BOOT}
    assert check["n_files"].tolist() == [world.SPLITS["eval"]] * 2 + [world.SPLITS["itw"]] * 2
    base = second["base"]
    # Another number of resamples, other speakers, another file order.
    fewer = A.paired_resamples_check(second["reference"], second_tables(base, n_boot=N_BOOT - 1))
    assert not fewer["ok"].any() and fewer["same_files_in_order"].all()
    other = second_tables(base)
    other.resamples.speaker_maps = {"eval": world.speakers_of("eval"),
                                    "itw": {k: "one" for k in world.speakers_of("itw")}}
    table = A.paired_resamples_check(second["reference"], other).set_index(["test_set", "level"])
    assert table["ok"].to_dict() == {("eval", "file"): True, ("eval", "spk"): True,
                                     ("itw", "file"): False, ("itw", "spk"): False}
    assert not table.loc[("itw", "spk"), "same_speakers"]
    world.write_runs(str(tmp_path / "reversed"), encoder=OTHER, reverse_files=True)
    reversed_tables = A.ScoreTables(
        OTHER, str(tmp_path / "reversed"), str(tmp_path / "t"),
        speaker_maps={t: world.speakers_of(t) for t in cfg.TEST_SETS}, conditions=CONDITIONS,
        layers=world.LAYERS, cross_layers=CROSS, seeds=world.SEEDS, n_boot=N_BOOT, log=lambda line: None)
    table = A.paired_resamples_check(second["reference"], reversed_tables)
    assert not table["ok"].any() and not table["same_files_in_order"].any()


def test_the_differences_between_the_encoders_are_the_reference_minus_the_second(second):
    out, tables, reference = second["out"], second["tables"], second["reference"]
    table = out["encoder_effects"]
    assert set(table["encoder_a"]) == {world.ENCODER} and set(table["encoder_b"]) == {OTHER}
    assert (table["estimate"] == table["estimate_a"] - table["estimate_b"]).all()
    keys = ["condition", "test_set", "pooling", "layer", "quantity"]
    all_cells = [f"eer_{c}" for c in E.CELLS] + list(cfg.EFFECTS)
    matched = ["eer_cc", "eer_tt", "delta_total"]
    direct = pd.concat([
        E.encoder_differences(reference.resamples, tables.resamples, "trim40", world.LAYERS),
        E.encoder_differences(reference.resamples, tables.resamples, "trim20", world.CROSS_LAYERS),
        E.encoder_differences(reference.resamples, tables.resamples, "trim20",
                              [x for x in world.LAYERS if x not in world.CROSS_LAYERS], cells=E.MATCHED_CELLS)],
        ignore_index=True)
    same(table, direct, keys)
    quantities = table.groupby(["condition", "layer"])["quantity"].agg(lambda q: sorted(set(q))).to_dict()
    for layer in world.LAYERS:
        assert quantities[("trim40", layer)] == sorted(all_cells)
        assert quantities[("trim20", layer)] == sorted(all_cells if layer in world.CROSS_LAYERS else matched)
    # The estimates are those of the effects tables of the two encoders.
    index = ["condition", "test_set", "pooling", "layer", "effect"]
    first = A.read_table(str(second["base"] / "tables" / "effects_2x2.csv")).set_index(index)
    other = out["effects_2x2"].set_index(index)
    for row in table[table["quantity"].isin(cfg.EFFECTS) & (table["layer"] == 3)].itertuples():
        key = (row.condition, row.test_set, row.pooling, row.layer, row.quantity)
        assert row.estimate_a == first.loc[key, "estimate"] and row.estimate_b == other.loc[key, "estimate"]
    # A layer without the cross-condition cells: the total effect of the matched-cells tables.
    totals = out["effects_matched_cells"].set_index(index)
    row = table[(table["condition"] == "trim20") & (table["layer"] == 1) & (table["quantity"] == "delta_total")
                & (table["test_set"] == "itw") & (table["pooling"] == "stats")].iloc[0]
    assert row["estimate_b"] == totals.loc[("trim20", "itw", "stats", 1, "delta_total"), "estimate"]
    tau = out["encoder_tau"]
    direct = pd.concat([R.encoder_tau_differences(reference.resamples, tables.resamples, c, layers=world.LAYERS)
                        for c in CONDITIONS], ignore_index=True)
    same(tau, direct, ["condition", "test_set", "pooling"])
    rank = out["rank_agreement"].set_index(["condition", "test_set", "pooling"])["kendall_tau"]
    for row in tau.itertuples():
        assert row.tau_b == rank[(row.condition, row.test_set, row.pooling)]
    # The meta names the runs of both encoders.
    meta = json.loads((second["base"] / "tables_other" / "encoder_difference_cells_effects.meta.json")
                      .read_text(encoding="utf-8"))
    assert meta["inputs"]["encoder"] == OTHER and meta["inputs"]["reference"]["encoder"] == world.ENCODER
    assert meta["inputs"]["reference"]["runs_csv_sha256"]["unpadded"] == \
        provenance.sha256_file(second["base"] / "runs" / "runs.csv")


def test_the_cells_are_computed_once_for_all_tables(second, monkeypatch):
    from ssl_silence import bootstrap as B

    base = second["base"]
    calls = []
    real = B.seed_mean_resampled_eer
    monkeypatch.setattr(B, "seed_mean_resampled_eer", lambda *a: calls.append(1) or real(*a))
    tables = second_tables(base, folder="tables_again")
    resamples = tables.resamples
    first = E.effects_ci(resamples, "trim40", (3,), poolings=("stats",), test_sets=("itw",))
    n = len(calls)
    assert n == 4 * 2                                              # four cells, file and speaker level
    again = E.effects_ci(resamples, "trim40", (3,), poolings=("stats",), test_sets=("itw",))
    E.total_effect_ci(resamples, "trim40", (3,), poolings=("stats",), test_sets=("itw",))
    assert len(calls) == n and again.equals(first)
    E.effects_ci(resamples, "trim20", (3,), poolings=("stats",), test_sets=("itw",))
    assert len(calls) == n + 3 * 2                                 # the untrimmed cell is the same cell
    fresh = E.effects_ci(second_tables(base, folder="tables_again").resamples, "trim40", (3,),
                         poolings=("stats",), test_sets=("itw",))
    assert fresh.equals(first)


def test_the_replication_rule_on_the_units_of_the_two_encoders(second):
    out, base = second["out"], second["base"]
    comparison = out["replication_comparison"]
    reference = rep.read_units(second["reference_units"])
    units = rep.read_units(str(base / "tables_other" / "replication_units.csv"))
    assert len(comparison) == len(reference) == len(units) == 236
    direct = rep.compare_units(reference, units)
    assert comparison["verdict"].tolist() == direct["verdict"].tolist()
    assert comparison["estimate"].tolist() == reference["estimate"].tolist()
    assert comparison["replica_estimate"].tolist() == direct["replica_estimate"].tolist()
    assert comparison["in_rule"].tolist() == reference["in_rule"].tolist()
    # The verdict by hand, for every unit.
    for row in comparison.itertuples():
        excludes = row.replica_spk_ci_low > 0 or row.replica_spk_ci_high < 0
        same_sign = (row.estimate > 0) == (row.replica_estimate > 0) and row.replica_estimate != 0
        if not row.in_rule:
            assert row.verdict == "descriptive"
        else:
            assert row.verdict == ("replicated" if excludes and same_sign else "not replicated")
    assert {"replicated", "descriptive"} <= set(comparison["verdict"])
    outcome = out["replication_outcome"]
    assert len(outcome) == len(out["units_summary"]) and outcome["n_units"].sum() == 236
    assert outcome["n_in_rule"].tolist() == rep.summarise_units(reference)["n_in_rule"].tolist()
    assert outcome["n_replicated"].sum() == (comparison["verdict"] == "replicated").sum()
    assert (outcome[rep.OUTCOME_COUNTS].sum(axis=1) == outcome["n_units"]).all()
    # The reverse count is the number of units in which only the second encoder excludes zero.
    assert outcome["replica_only"].sum() == (~comparison["in_rule"] & comparison["replica_excludes_zero"]).sum()
    findings = out["replication_findings"].set_index("finding_id")
    assert findings.loc[rep.ALL_UNITS, "n_units"] == 236
    assert findings.loc[rep.ALL_UNITS, "n_in_rule"] == reference["in_rule"].sum()
    assert pd.isna(findings.loc["F7", "n_replicated"]) and findings.loc["F7", "n_units"] == 2 * len(world.LAYERS)
    assert findings.iloc[:-1]["n_units"].sum() == 236
    absent = out["replication_absence_findings"]
    assert set(absent["finding_id"]) == {"F7", "F16"} and len(absent) == 2 * len(world.LAYERS) + 2 * len(world.CROSS_LAYERS)
    meta = json.loads((base / "tables_other" / "replication_comparison.meta.json").read_text(encoding="utf-8"))
    assert meta["inputs"]["reference_units_sha256"] == rep.sha256_lf(second["reference_units"])
    # Units of other layers are not the same units.
    fewer = reference[reference["layer"] != 1]
    path = base / "fewer_units.csv"
    fewer.to_csv(path, index=False, lineterminator="\n")
    with pytest.raises(ValueError, match="the units of the two encoders differ: 0 only in the reference"):
        second["tables"].replication_outcome(str(path))


def test_every_table_of_the_plan_is_written_and_nothing_else(second):
    tables, folder = second["tables"], second["base"] / "tables_other"
    planned = A.planned_tables()
    assert list(planned.columns) == ["table", "kind", "plan"]
    assert planned["kind"].value_counts().to_dict() == {
        "analysis plan": len(A.PLAN_TABLES), "analysis plan, dev-selected layers": 4, "exploratory": 1}
    assert planned[planned["kind"] == "exploratory"]["table"].tolist() == \
        ["exploratory/exploratory_itw_by_duration_thresholds"]
    assert (planned["plan"].str.len() > 0).all() and "padded_vs_unpadded" not in set(planned["table"])
    check = tables.written_tables_check()
    assert check["ok"].all(), check[~check["ok"]]
    assert check["table"].tolist() == planned["table"].tolist()
    additional = set(A.SELECTED_LAYER_TABLES) if tables.selected_layers_outside_detail() else set()
    assert set(check[check["written"]]["table"]) == set(planned["table"]) - (set(A.SELECTED_LAYER_TABLES) - additional)
    assert {p.stem for p in folder.glob("*.csv")} == set(A.PLAN_TABLES) | additional
    assert {p.stem for p in (folder / A.EXPLORATORY).glob("*.csv")} == set(A.EXPLORATORY_TABLES)
    # A table that is not listed, in either folder, and a missing table of the plan fail.
    stray = [folder / "something_else.csv", folder / A.EXPLORATORY / "exploratory_other.csv"]
    moved = folder / "cells_2x2.csv"
    try:
        for path in stray:
            path.write_text("a\n1\n", encoding="utf-8")
        os.replace(moved, str(moved) + ".away")
        check = tables.written_tables_check().set_index("table")
        assert check[~check["ok"]].index.tolist() == ["cells_2x2", "something_else", "exploratory/exploratory_other"]
        assert check.loc["something_else", "kind"] == "not listed"
    finally:
        os.replace(str(moved) + ".away", moved)
        for path in stray:
            os.replace(path, str(path) + ".away")
    assert tables.written_tables_check()["ok"].all()


# Noise ceiling of the rank agreement ------------------------------------------------------------

def test_the_noise_ceiling_is_written_in_parts_and_is_not_a_table_of_the_plan(setup, tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "git_commit", lambda repo=None: COMMIT)
    base, lines = setup["base"], []
    tables = A.ScoreTables(world.ENCODER, str(base / "runs"), str(tmp_path / "tables"),
                           speaker_maps={t: world.speakers_of(t) for t in cfg.TEST_SETS},
                           conditions=CONDITIONS, layers=world.LAYERS, cross_layers=CROSS,
                           seeds=world.SEEDS, n_boot=N_BOOT, log=lines.append)
    ceiling, splits = tables.rank_noise_ceiling()
    direct, direct_splits = R.noise_ceiling(tables.resamples, "trim40", layers=world.LAYERS)
    same(ceiling, direct, ["condition", "test_set", "pooling"])
    same(splits, direct_splits, ["condition", "test_set", "pooling", "split"])
    assert set(ceiling["condition"]) == {rep.MAIN_CONDITION} and len(splits) == 4 * N_BOOT
    folder = tmp_path / "tables"
    assert {p.stem for p in folder.glob("*.csv")} == {"rank_noise_ceiling", "rank_noise_ceiling_splits"}
    assert {p.stem for p in (folder / A.PARTS).glob("*.csv")} == {
        "rank_noise_ceiling_trim40", "rank_noise_ceiling_splits_trim40"}
    meta = json.loads((folder / "rank_noise_ceiling.meta.json").read_text(encoding="utf-8"))
    assert meta["git_commit"] == COMMIT and meta["inputs"]["n_boot"] == N_BOOT
    assert meta["sha256"] == provenance.sha256_file(folder / "rank_noise_ceiling.csv")
    # A second call at the same commit reads both parts back.
    assert sum("computed" in line for line in lines) == 2
    again, _ = tables.rank_noise_ceiling()
    assert sum("reused" in line for line in lines) == 2 and again.equals(ceiling)
    assert [step["status"] for step in tables.steps] == ["computed", "computed", "reused", "reused"]
    # The plan does not name it: it is not among the tables of a second encoder.
    listed = set(A.PLAN_TABLES) | set(A.SELECTED_LAYER_TABLES) | set(A.EXPLORATORY_TABLES)
    assert not {"rank_noise_ceiling", "rank_noise_ceiling_splits"} & listed
