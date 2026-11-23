"""selection against the original metric table, seed summary and dev selection.
Skipped without the original code."""

from __future__ import annotations

import numpy as np
import pytest

import world
from original_names import ORIGINAL_VALUES
from ssl_silence import config as cfg
from ssl_silence import selection as S

SELECTION_LABELS = {"pre-specified": "reference", "dev-selected": "dev-selected",
                    "pre-specified + dev-selected": "reference + dev-selected"}
SUMMARY_COLUMNS = ["n_seeds"] + [f"{name}_{stat}" for name in S.TARGETS.values()
                                 for stat in ("mean", "sd")]


@pytest.fixture(scope="module")
def both(tmp_path_factory, original):
    """The world in both layouts, with the tables of both codes."""
    base = tmp_path_factory.mktemp("world")
    feature_set = ORIGINAL_VALUES["unpadded_feature_set"]
    asv_dir = world.write_asv_scores(str(base / "asv"))
    runs_dir = world.write_runs(str(base / "runs"))
    main_dir, threshold_dir = str(base / "old_main"), str(base / "old_threshold")
    world.write_original_runs(original("train"), feature_set, main_dir, threshold_dir)
    tidy = original("tidy")
    asv_files = {split: f"{asv_dir}/{name}" for split, name in cfg.ASV_SCORE_FILENAMES.items()}
    old = {name: tidy.build_tidy(folder, asv_score_files=asv_files, write=False, progress=False)
           for name, folder in (("main", main_dir), ("threshold", threshold_dir))}
    new = S.metrics_per_run(runs_dir, S.load_asv(asv_dir))
    return {"new": new, "old": old, "feature_set": feature_set}


def record(parity_record, name, differences):
    worst = max(differences)
    parity_record("selection", name, len(differences), worst)
    assert worst == 0.0, f"{name}: max abs diff {worst}"


def difference(new, old) -> float:
    new, old = float(new), float(old)
    if np.isnan(new) and np.isnan(old):
        return 0.0
    return abs(new - old)


def test_metrics_per_run(both, original, parity_record):
    O = original("train")
    old = {name: table.set_index(["run_id", "test_set", "metric"])["value"]
           for name, table in both["old"].items()}
    differences, counts = [], [0, 0]
    for row in both["new"].itertuples():
        c, t = row.condition, row.test_condition
        spec = O.RunSpec(encoder=row.encoder, layer=int(row.layer), pooling=row.pooling,
                         mask="none", condition=world.OLD_CONDITIONS[c], seed=int(row.seed),
                         feature_set=both["feature_set"],
                         test_condition=None if t == c else world.OLD_CONDITIONS[t])
        test_set = row.split if (t == c or row.split != "dev") else "dev_tc"
        table = old["threshold" if "trim20" in (c, t) else "main"]
        key = (spec.run_id, test_set, row.metric)
        if key not in table.index:
            # The original code computed the t-DCF for the sets named dev and eval only.
            assert row.metric == "tandem_min_tdcf" and test_set == "dev_tc"
            continue
        counts[row.metric == "eer"] += 1
        differences.append(difference(row.value, table[key]))
    assert counts[1] == (both["new"]["metric"] == "eer").sum() and counts[0] > 0
    record(parity_record, "metrics_per_run (EER and tandem t-DCF)", differences)


def test_seed_summary_and_dev_selection(both, original, original_function, parity_record):
    tidy = original("tidy")
    summary = S.seed_summary(both["new"])
    matched = S.matched_cells(summary)
    old_summary = {name: tidy.grid_summary(table, feature_set=both["feature_set"],
                                           encoder=world.ENCODER)
                   for name, table in both["old"].items()}
    differences = []
    for condition in world.CONDITIONS:
        old = old_summary["threshold" if condition == "trim20" else "main"]
        old = old[old["condition"] == world.OLD_CONDITIONS[condition]].sort_values(["pooling", "layer"])
        new = matched[matched["condition"] == condition].sort_values(["pooling", "layer"])
        assert new[["pooling", "layer"]].values.tolist() == old[["pooling", "layer"]].values.tolist()
        differences += [difference(a, b) for column in SUMMARY_COLUMNS
                        for a, b in zip(new[column], old[column])]
    record(parity_record, "seed_summary (matched cells)", differences)

    differences = []
    new_best = S.select_on_dev(summary).set_index(["condition", "pooling"])
    for condition in world.CONDITIONS:
        old = tidy.best_layer_on_dev(
            old_summary["threshold" if condition == "trim20" else "main"])
        old = old[old["condition"] == world.OLD_CONDITIONS[condition]]
        for row in old.itertuples():
            new = new_best.loc[(condition, row.pooling)]
            differences += [float(abs(new["layer"] - row.layer)),
                            0.0 if bool(new["best_in_condition"]) == bool(row.best_in_condition)
                            else np.inf]
    record(parity_record, "select_on_dev", differences)

    old_configs = original_function("analysis", "selected_configs")(old_summary["main"])
    new_configs = S.selected_configs(summary, world.ENCODER)
    reverse = {v: k for k, v in world.OLD_CONDITIONS.items()}
    old_rows = sorted((reverse[r.condition], r.pooling, int(r.layer), SELECTION_LABELS[r.selection])
                      for r in old_configs.itertuples())
    new_rows = sorted((r.condition, r.pooling, int(r.layer), r.selection)
                      for r in new_configs.itertuples())
    record(parity_record, "selected_configs", [0.0 if new_rows == old_rows else np.inf])
