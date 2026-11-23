"""ranking against the original rank agreement, BCa interval and threshold table.
Skipped without the original code."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import world
from original_names import ORIGINAL_VALUES
from ssl_silence import bootstrap as B
from ssl_silence import effects as E
from ssl_silence import ranking as R
from ssl_silence import selection as S
from test_parity_effects import difference, record

N_BOOT = 40
SUMMARY_COLUMNS = ("n_layers", "n_speakers", "n_jackknife", "n_boot", "kendall_tau", "tau_ci_low",
                   "tau_ci_high", "tau_bca_low", "tau_bca_high", "z0_tau", "a_tau", "tie_share_tau",
                   "kendall_p", "kendall_p_exact", "kendall_exact_applies", "spearman_rho",
                   "rho_ci_low", "rho_ci_high", "rho_bca_low", "rho_bca_high", "z0_rho", "a_rho",
                   "spearman_p")


@pytest.fixture(scope="module")
def both(tmp_path_factory, original):
    base = tmp_path_factory.mktemp("world")
    runs_dir = world.write_runs(str(base / "runs"))
    main_dir, threshold_dir = str(base / "old_main"), str(base / "old_threshold")
    world.write_original_runs(original("train"), ORIGINAL_VALUES["unpadded_feature_set"],
                              main_dir, threshold_dir)
    speakers = {"eval": world.speakers_of("eval"), "itw": world.speakers_of("itw")}
    store = B.ScoreStore(runs_dir, seeds=world.SEEDS)
    A = original("analysis")
    old_dirs = {"trim40": main_dir, "trim20": {"clean": main_dir, "trimmed20": threshold_dir}}
    new, old = {}, {}
    resamples = B.Resamples(store, speakers, n_boot=N_BOOT)
    for condition in ("trim40", "trim20"):
        new[condition] = R.rank_agreement_ci(resamples, condition, layers=world.LAYERS)
        old[condition] = A.layer_rank_agreement_ci(
            old_dirs[condition], conditions=("clean", world.OLD_CONDITIONS[condition]),
            layers=world.LAYERS, seeds=world.SEEDS, n_boot=N_BOOT, speaker_maps=speakers,
            progress=False, ci="both")
    return {"A": A, "runs_dir": runs_dir, "speakers": speakers, "resamples": resamples,
            "old_dirs": old_dirs, "new": new, "old": old}


def test_bca_interval_and_jackknife(original, parity_record):
    A = original("analysis")
    rng = np.random.default_rng(8)
    differences = []
    for n in (12, 80, 400):
        labels = (rng.random(n) < 0.6).astype(int)
        labels[:2] = [0, 1]
        groups = np.array([f"s{int(k)}" for k in rng.integers(0, 7, n)])
        new_counts, new_units = R.jackknife_speaker_counts(labels, groups)
        old_counts, old_units = A.jackknife_speaker_counts(labels, groups)
        differences.append(0.0 if new_units == old_units and new_counts.dtype == old_counts.dtype
                           else np.inf)
        differences.append(float(np.max(np.abs(new_counts.astype(int) - old_counts.astype(int)))))
        boots = np.round(rng.normal(0.3, 0.2, 500), 2)       # rounded: ties with the observed value
        jack = rng.normal(0.3, 0.05, n)
        cases = [(0.3, boots, jack), (0.31, boots, jack), (0.3, boots, np.full(n, 0.3)),
                 (-9.0, boots, jack), (9.0, boots, jack), (0.3, boots, jack, 0.1)]
        for case in cases:
            new, old = R.bca_interval(*case), A.bca_interval(*case)
            assert set(new) == set(old)
            differences += [difference(new[key], old[key]) for key in old]
    record(parity_record, "ranking", "bca_interval and jackknife_speaker_counts", differences)


def test_rank_agreement(both, parity_record):
    differences = []
    for condition in ("trim40", "trim20"):
        old_condition = world.OLD_CONDITIONS[condition]
        (summary, detail), (old_summary, old_detail) = both["new"][condition], both["old"][condition]
        old_summary = old_summary.set_index(["test_set", "pooling"])
        old_detail = old_detail.set_index(["test_set", "pooling", "layer"])
        assert len(summary) == len(old_summary) and len(detail) == len(old_detail)
        for row in summary.itertuples():
            reference = old_summary.loc[(row.test_set, row.pooling)]
            differences += [difference(getattr(row, column), reference[column])
                            for column in SUMMARY_COLUMNS]
            differences += [difference(row.best_layer_clean, reference["best_layer_clean"]),
                            difference(row.best_layer_trimmed,
                                       reference[f"best_layer_{old_condition}"])]
        for row in detail.itertuples():
            reference = old_detail.loc[(row.test_set, row.pooling, row.layer)]
            differences += [difference(row.eer_clean, reference["eer_clean"]),
                            difference(row.eer_trimmed, reference[f"eer_{old_condition}"]),
                            difference(row.p_best_clean, reference["p_best_clean"]),
                            difference(row.p_best_trimmed, reference[f"p_best_{old_condition}"])]
    record(parity_record, "ranking",
           "rank_agreement_ci (percentile and BCa intervals, exact p, best layer; 40 and 20 dB)",
           differences)


def test_threshold_summary(both, original, original_function, parity_record):
    A = both["A"]
    layer_configs = original_function("analysis", "layer_configs")
    # The dev selection of the original code, from its own metric tables: the selected
    # configurations at 40 dB, the configurations of the extra condition at 20 dB.
    tidy = original("tidy")
    old_summary = {
        condition: tidy.grid_summary(
            tidy.build_tidy(folder, asv_score_files={}, write=False, progress=False),
            feature_set=ORIGINAL_VALUES["unpadded_feature_set"], encoder=world.ENCODER)
        for condition, folder in (("trim40", both["old_dirs"]["trim40"]),
                                  ("trim20", both["old_dirs"]["trim20"]["trimmed20"]))}
    old_selection = {
        "trim40": original_function("analysis", "selected_configs")(old_summary["trim40"]),
        "trim20": original_function("analysis", "threshold_configs")(
            old_summary["trim20"], world.OLD_CONDITIONS["trim20"])}
    conditions = ("trim40", "trim20")
    cross = {"trim40": world.LAYERS, "trim20": world.CROSS_LAYERS}
    fallback = {"trim40": pd.DataFrame({"split": ["train", "dev", "eval", "itw"], "label": "all",
                                        "n_fallback": [8, 3, 21, 0],
                                        "pct_fallback": [0.03, 0.01, 0.029, 0.0]}),
                "trim20": pd.DataFrame({"split": ["train", "train", "itw"], "label": ["0", "all", "all"],
                                        "n_fallback": [5, 80, 7], "pct_fallback": [0.2, 0.3, 0.02]})}
    dev = S.select_on_dev(S.seed_summary(S.metrics_per_run(both["runs_dir"])))
    new = R.threshold_summary(
        pd.concat([both["new"][c][0] for c in conditions]),
        pd.concat([both["new"][c][1] for c in conditions]),
        pd.concat([E.effects_ci(both["resamples"], c, cross[c]) for c in conditions]),
        fallback, dev, conditions=conditions, layers=world.CROSS_LAYERS)
    per_condition = {}
    for condition in conditions:
        old_condition = world.OLD_CONDITIONS[condition]
        effects = A.cross_effects_ci(layer_configs(cross[condition], world.POOLINGS),
                                     both["old_dirs"][condition], seeds=world.SEEDS, n_boot=N_BOOT,
                                     speaker_maps=both["speakers"], progress=False,
                                     conditions=("clean", old_condition), include_total=True,
                                     aggregation="seed-mean")
        chosen = old_selection[condition]
        assert chosen[chosen["condition"] == old_condition]["selection"].str.contains(
            "dev-selected").sum() == len(world.POOLINGS)
        per_condition[old_condition] = {
            "top_db": int(new[new["condition"] == condition]["top_db"].iloc[0]),
            "rank": both["old"][condition][0], "rank_detail": both["old"][condition][1],
            "effects": effects, "fallback": fallback[condition], "dev_selected": chosen}
    old = A.threshold_comparison(per_condition, layers=world.CROSS_LAYERS)
    assert len(new) == len(old) and new["dev_selected_layer"].notna().all()
    old = old.set_index(["condition", "pooling"])
    differences = []
    for row in new.to_dict("records"):
        old_condition = world.OLD_CONDITIONS[row["condition"]]
        reference = old.loc[(old_condition, row["pooling"])]
        renamed = {column.replace("_trimmed", f"_{old_condition}"): value
                   for column, value in row.items() if column not in ("condition", "pooling")}
        # The original table names the best-layer columns after each condition, so a
        # row has empty columns for the other conditions.
        others = set(reference.index) - set(renamed)
        assert set(renamed) <= set(reference.index) and reference[sorted(others)].isna().all()
        differences += [difference(value, reference[column]) for column, value in renamed.items()]
    record(parity_record, "ranking", "threshold_summary", differences)
