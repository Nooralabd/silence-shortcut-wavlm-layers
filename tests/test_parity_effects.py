"""bootstrap and effects against the original seed-mean analysis.
Skipped without the original code."""

from __future__ import annotations

import numpy as np
import pytest

import world
from original_names import ORIGINAL_VALUES
from ssl_silence import bootstrap as B
from ssl_silence import config as cfg
from ssl_silence import effects as E
from ssl_silence import selection as S

N_BOOT = 40
# Effects of this repository -> (effect, fixed) labels of the original code,
# for the trimmed condition named b there.
OLD_EFFECTS = {"delta_test": ("test effect", "train=clean"),
               "delta_train": ("train effect", "test={b}"),
               "delta_total": ("{b} - clean", ""),
               "delta_train_c": ("train effect", "test=clean"),
               "delta_test_t": ("test effect", "train={b}"),
               "interaction": ("interaction", "")}
INTERVALS = {"estimate": "observed", "file_ci_low": "ci_low", "file_ci_high": "ci_high",
             "file_boot_se": "boot_se", "spk_ci_low": "spk_ci_low", "spk_ci_high": "spk_ci_high",
             "spk_boot_se": "spk_boot_se"}


@pytest.fixture(scope="module")
def both(tmp_path_factory, original, original_function):
    base = tmp_path_factory.mktemp("world")
    runs_dir = world.write_runs(str(base / "runs"))
    main_dir, threshold_dir = str(base / "old_main"), str(base / "old_threshold")
    world.write_original_runs(original("train"), ORIGINAL_VALUES["unpadded_feature_set"],
                              main_dir, threshold_dir)
    speakers = {"eval": world.speakers_of("eval"), "itw": world.speakers_of("itw")}
    store = B.ScoreStore(runs_dir, seeds=world.SEEDS)
    return {"runs_dir": runs_dir, "speakers": speakers, "store": store,
            "resamples": B.Resamples(store, speakers, n_boot=N_BOOT),
            "old_dirs": {"trim40": main_dir,
                         "trim20": {"clean": main_dir, "trimmed20": threshold_dir}},
            "main_dir": main_dir, "threshold_dir": threshold_dir,
            "layer_configs": original_function("analysis", "layer_configs"),
            "A": original("analysis")}


def record(parity_record, module, name, differences):
    worst = max(differences)
    parity_record(module, name, len(differences), worst)
    assert worst == 0.0, f"{name}: max abs diff {worst}"


def difference(new, old) -> float:
    if isinstance(new, (bool, np.bool_)) or isinstance(old, (bool, np.bool_)):
        return 0.0 if bool(new) == bool(old) else np.inf
    new, old = float(new), float(old)
    return 0.0 if (np.isnan(new) and np.isnan(old)) else abs(new - old)


def test_resampled_eer_and_draws(original, original_function, parity_record):
    A = original("analysis")
    rng = np.random.default_rng(4)
    eer, draws = [], []
    for n in (60, 500, 3_000):
        labels = (rng.random(n) < 0.7).astype(int)
        labels[:2] = [0, 1]
        scores = np.round(rng.random(n) + 0.3 * labels, 2).astype(np.float32)
        groups = np.array([f"s{int(k)}" for k in rng.integers(0, 9, n)])
        counts = rng.integers(0, 4, size=(15, n)).astype(np.int16)
        eer.append(float(np.max(np.abs(B.resampled_eer(labels, scores, counts)
                                       - A.bootstrap_eer(labels, scores, counts)))))
        matrix = np.stack([scores.astype(np.float64), scores[::-1].astype(np.float64)])
        eer.append(abs(B.seed_mean_eer(matrix, labels) - A.seed_mean_eer(matrix, labels)))
        eer.append(float(np.max(np.abs(B.seed_mean_resampled_eer(matrix, labels, counts)
                                       - A.seed_mean_bootstrap(matrix, labels, counts)))))
        cls_groups = B.class_groups(labels, groups)
        for seed in range(5):
            new = B.draw_file_counts(np.random.default_rng(seed), labels)
            old = A.draw_file_counts(np.random.default_rng(seed), labels)
            draws.append(float(np.max(np.abs(new.astype(int) - old.astype(int)))))
            new = B.draw_speaker_counts(np.random.default_rng(seed), labels, cls_groups)
            old = A.draw_speaker_counts(np.random.default_rng(seed), labels, cls_groups)
            draws.append(float(np.max(np.abs(new.astype(int) - old.astype(int)))))
        draws.append(0.0 if B.speakers_per_label(labels, groups) == A.speakers_per_label(labels, groups)
                     else np.inf)
    old_seeds = original_function("analysis", "replicate_seeds")
    for n_boot in (7, 1_000):
        new, old = B.replicate_seeds(("eval", "itw"), n_boot), old_seeds(("eval", "itw"), n_boot)
        for test_set in ("eval", "itw"):
            for level in (0, 1):
                draws.append(0.0 if np.array_equal(new[test_set][level], old[test_set][level]) else np.inf)
    record(parity_record, "bootstrap", "resampled_eer and seed-mean aggregation", eer)
    record(parity_record, "bootstrap", "file and speaker draws, replicate seeds", draws)


def compare_effects(new, old, old_condition) -> list[float]:
    old = old.fillna({"fixed": ""}).set_index(["test_set", "pooling", "layer", "effect", "fixed"])
    differences = []
    for row in new.itertuples():
        effect, fixed = (s.format(b=old_condition) for s in OLD_EFFECTS[row.effect])
        reference = old.loc[(row.test_set, row.pooling, row.layer, effect, fixed)]
        differences += [difference(getattr(row, column), reference[other])
                        for column, other in INTERVALS.items()]
        differences += [difference(getattr(row, f"eer_{cell}"), reference[f"eer_{cell}"])
                        for cell in E.CELLS]
        differences += [difference(row.file_excludes_zero, reference["excludes_zero"]),
                        difference(row.spk_excludes_zero, reference["spk_excludes_zero"]),
                        difference(row.n_files, reference["n_files"])]
    assert len(new) == len(old)
    return differences


def test_effects_of_the_2x2(both, parity_record):
    A = both["A"]
    differences = []
    for condition, layers in (("trim40", world.LAYERS), ("trim20", world.CROSS_LAYERS)):
        old_condition = world.OLD_CONDITIONS[condition]
        new = E.effects_ci(both["resamples"], condition, layers)
        old = A.cross_effects_ci(both["layer_configs"](layers, world.POOLINGS),
                                 both["old_dirs"][condition], seeds=world.SEEDS, n_boot=N_BOOT,
                                 speaker_maps=both["speakers"], progress=False,
                                 conditions=("clean", old_condition), include_total=True,
                                 aggregation="seed-mean")
        differences += compare_effects(new, old, old_condition)
    record(parity_record, "effects", "effects_ci (six effects, file and speaker level, 40 and 20 dB)",
           differences)


def test_effects_at_the_full_number_of_resamples(both, parity_record):
    """The replicate seeds depend on the number of resamples: one configuration at 1,000."""
    A = both["A"]
    resamples = B.Resamples(both["store"], both["speakers"], n_boot=cfg.N_BOOT)
    new = E.effects_ci(resamples, "trim40", layers=(3,), poolings=("stats",))
    old = A.cross_effects_ci(both["layer_configs"]((3,), ("stats",)), both["main_dir"],
                             seeds=world.SEEDS, n_boot=cfg.N_BOOT, speaker_maps=both["speakers"],
                             progress=False, conditions=("clean", "trimmed"), include_total=True,
                             aggregation="seed-mean")
    record(parity_record, "effects", "effects_ci (1,000 resamples)",
           compare_effects(new, old, "trimmed"))


def test_selected_configs_intervals(both, original_function, parity_record):
    A = both["A"]
    summary = S.seed_summary(S.metrics_per_run(both["runs_dir"]))
    configs = S.selected_configs(summary, world.ENCODER)
    new = E.selected_configs_eer_ci(both["resamples"], configs)
    old_configs = configs.assign(condition=configs["condition"].map(world.OLD_CONDITIONS))
    old = A.bootstrap_eer_ci(old_configs, both["main_dir"], seeds=world.SEEDS, n_boot=N_BOOT,
                             speaker_maps=both["speakers"], progress=False,
                             aggregation="seed-mean")
    old = old.set_index(["config", "test_set"])
    differences = []
    for row in new.itertuples():
        reference = old.loc[(row.config, row.test_set)]
        differences += [difference(row.eer, reference["eer"]),
                        difference(row.file_ci_low, reference["ci_low"]),
                        difference(row.file_ci_high, reference["ci_high"]),
                        difference(row.file_boot_se, reference["boot_se"]),
                        difference(row.spk_ci_low, reference["spk_ci_low"]),
                        difference(row.spk_ci_high, reference["spk_ci_high"]),
                        difference(row.spk_boot_se, reference["spk_boot_se"]),
                        difference(row.n_speakers_bonafide, reference["n_speakers_bonafide"]),
                        difference(row.n_speakers_spoof, reference["n_speakers_spoof"]),
                        difference(row.n_files, reference["n_files"])]
    assert len(new) == len(old) == 2 * len(configs)
    record(parity_record, "effects", "selected_configs_eer_ci", differences)


def test_per_attack(both, parity_record):
    A = both["A"]
    attack_map = world.attacks_of_eval()
    differences = []
    for condition in ("trim40", "trim20"):
        old_condition = world.OLD_CONDITIONS[condition]
        new = E.per_attack_effect_ci(both["resamples"], condition, attack_map,
                                     layers=world.CROSS_LAYERS)
        old = A.per_attack_trimmed_minus_clean_ci(
            both["layer_configs"](world.CROSS_LAYERS, world.POOLINGS), attack_map,
            both["old_dirs"][condition], seeds=world.SEEDS, n_boot=N_BOOT,
            speaker_map=both["speakers"]["eval"], progress=False,
            conditions=("clean", old_condition), aggregation="seed-mean")
        old = old.set_index(["pooling", "layer", "attack"])
        for row in new.itertuples():
            reference = old.loc[(row.pooling, row.layer, row.attack)]
            differences += [difference(row.estimate, reference["diff"]),
                            difference(row.eer_clean, reference["eer_clean"]),
                            difference(row.eer_trimmed, reference[f"eer_{old_condition}"]),
                            difference(row.file_ci_low, reference["ci_low"]),
                            difference(row.file_ci_high, reference["ci_high"]),
                            difference(row.file_boot_se, reference["boot_se"]),
                            difference(row.file_excludes_zero, reference["excludes_zero"]),
                            difference(row.spk_ci_low, reference["spk_ci_low"]),
                            difference(row.spk_ci_high, reference["spk_ci_high"]),
                            difference(row.spk_boot_se, reference["spk_boot_se"]),
                            difference(row.spk_excludes_zero, reference["spk_excludes_zero"]),
                            difference(row.n_bonafide, reference["n_bonafide"]),
                            difference(row.n_spoof, reference["n_spoof"]),
                            difference(row.n_speakers_spoof, reference["n_speakers_spoof"])]
        assert len(new) == len(old)
    record(parity_record, "effects", "per_attack_effect_ci (40 and 20 dB)", differences)


def test_itw_by_duration_and_cells(both, original, parity_record):
    A = both["A"]
    durations = world.durations_of_itw()
    new = E.itw_by_duration(both["resamples"], "trim40", durations, layers=world.CROSS_LAYERS)
    old = A.itw_2x2_by_duration(both["main_dir"], durations, conditions=("clean", "trimmed"),
                                poolings=world.POOLINGS, layers=world.CROSS_LAYERS,
                                seeds=world.SEEDS)
    old = old.set_index(["pooling", "layer", "subset", "train", "test"])
    cells = {"eer_cc": ("clean", "clean"), "eer_ct": ("clean", "trimmed"),
             "eer_tc": ("trimmed", "clean"), "eer_tt": ("trimmed", "trimmed")}
    deltas = {name: label for label in sorted({t for (_, _, _, tr, t) in old.index if tr == "delta"})
              for name in ("delta_test", "delta_train", "delta_total") if label.startswith(name)}
    assert len(deltas) == 3
    differences = []
    for row in new.itertuples():
        if row.quantity in cells:
            key = (row.pooling, row.layer, row.subset, *cells[row.quantity])
        elif row.quantity in deltas:
            key = (row.pooling, row.layer, row.subset, "delta", deltas[row.quantity])
        else:
            continue   # the original table has three of the six effects
        reference = old.loc[key]
        differences += [difference(row.mean, reference["eer_mean"]),
                        difference(row.sd, reference["eer_sd"]),
                        difference(row.n_files, reference["n_files"]),
                        difference(row.n_bonafide, reference["n_bonafide"]),
                        difference(row.n_spoof, reference["n_spoof"])]
    assert len(differences) == 5 * len(old)
    record(parity_record, "effects", "itw_by_duration (cells and three effects)", differences)

    tidy = original("tidy")
    summary = S.seed_summary(S.metrics_per_run(both["runs_dir"]))
    new_cells = E.cells_table(summary, "trim40").set_index(["pooling", "layer", "cell"])
    old_cells = A.cross_cells_all_layers(
        [tidy.build_tidy(both["main_dir"], asv_score_files={}, write=False, progress=False)],
        layers=world.LAYERS, poolings=world.POOLINGS, conditions=("clean", "trimmed"),
        n_seeds=len(world.SEEDS)).set_index(["pooling", "layer", "cell"])
    assert len(new_cells) == len(old_cells)
    differences = [difference(new_cells.loc[key, column], old_cells.loc[key, column])
                   for key in old_cells.index
                   for column in ("n_seeds", "eval_eer_mean", "eval_eer_sd", "itw_eer_mean",
                                  "itw_eer_sd")]
    record(parity_record, "effects", "cells_table", differences)
