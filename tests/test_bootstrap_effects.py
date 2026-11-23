"""The stratified bootstrap and the effects of the 2x2 design, with known answers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import world
from ssl_silence import bootstrap as B
from ssl_silence import effects as E
from ssl_silence import metrics as M
from ssl_silence import selection as S
from ssl_silence import training as T

N_BOOT = 25


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    runs_dir = world.write_runs(str(tmp_path_factory.mktemp("world") / "runs"))
    store = B.ScoreStore(runs_dir, seeds=world.SEEDS)
    speakers = {"eval": world.speakers_of("eval"), "itw": world.speakers_of("itw")}
    return runs_dir, store, B.Resamples(store, speakers, n_boot=N_BOOT)


# EER of resamples --------------------------------------------------------------------

def test_resampled_eer_equals_the_eer_of_the_explicit_resample():
    rng = np.random.default_rng(0)
    for rounding in (None, 1):
        labels = (rng.random(300) < 0.7).astype(int)
        scores = rng.random(300) + 0.3 * labels
        if rounding is not None:
            scores = np.round(scores, rounding)  # many ties
        counts = rng.integers(0, 4, size=(20, 300)).astype(np.int16)
        got = B.resampled_eer(labels, scores.astype(np.float32), counts)
        for b in range(20):
            index = np.repeat(np.arange(300), counts[b])
            assert got[b] == M.eer(labels[index], scores.astype(np.float32)[index])
    ones = np.ones((1, 300), dtype=np.int16)
    assert B.resampled_eer(labels, scores, ones)[0] == M.eer(labels, scores)


def test_seed_mean_is_the_mean_of_the_per_seed_eer():
    rng = np.random.default_rng(1)
    labels = (rng.random(200) < 0.6).astype(int)
    matrix = rng.random((3, 200)) + 0.4 * labels
    assert B.seed_mean_eer(matrix, labels) == np.mean([M.eer(labels, row) for row in matrix])
    counts = rng.integers(0, 3, size=(7, 200)).astype(np.int16)
    expected = np.mean([B.resampled_eer(labels, row, counts) for row in matrix], axis=0)
    assert np.array_equal(B.seed_mean_resampled_eer(matrix, labels, counts), expected)


# Draws -------------------------------------------------------------------------------

def test_file_draw_is_stratified_by_class():
    labels = np.array([0] * 30 + [1] * 70)
    rng = np.random.default_rng(5)
    for _ in range(20):
        counts = B.draw_file_counts(rng, labels)
        assert counts.dtype == np.int16 and counts.shape == (100,)
        assert counts[:30].sum() == 30 and counts[30:].sum() == 70
    # The draw is integers(0, n, n) for bonafide, then for spoof.
    reference = np.random.default_rng(9)
    expected = np.bincount(np.r_[reference.integers(0, 30, 30), 30 + reference.integers(0, 70, 70)],
                           minlength=100)
    assert np.array_equal(B.draw_file_counts(np.random.default_rng(9), labels), expected)


def test_speaker_draw_takes_all_files_of_a_speaker_within_a_class():
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    groups = np.array(["b", "a", "a", "c", "a", "a", "b", "b", "b", "c"])
    cls_groups = B.class_groups(labels, groups)
    assert [g.tolist() for g in cls_groups[0]] == [[1, 2], [0], [3]]      # speakers a, b, c
    assert [g.tolist() for g in cls_groups[1]] == [[4, 5], [6, 7, 8], [9]]
    rng = np.random.default_rng(3)
    for _ in range(30):
        counts = B.draw_speaker_counts(rng, labels, cls_groups)
        assert counts[1] == counts[2] and counts[4] == counts[5]          # files of one unit move together
        assert counts[6] == counts[7] == counts[8]
        assert counts[0] + counts[1] + counts[3] == 3                     # three speakers drawn per class
        assert counts[4] + counts[6] + counts[9] == 3
    # Speaker a has files of both labels: the two classes are drawn independently.
    drawn = np.array([B.draw_speaker_counts(rng, labels, cls_groups) for _ in range(200)])
    assert (drawn[:, 1] != drawn[:, 4]).any()
    assert B.speakers_per_label(labels, groups) == {"n_speakers_bonafide": 3, "n_speakers_spoof": 3}


def test_replicate_seeds_are_drawn_per_test_set_in_order():
    seeds = B.replicate_seeds(("eval", "itw"), n_boot=10, seed=2026)
    rng = np.random.default_rng(2026)
    for test_set in ("eval", "itw"):
        assert np.array_equal(seeds[test_set][0], rng.integers(0, 2 ** 31, 10))   # files
        assert np.array_equal(seeds[test_set][1], rng.integers(0, 2 ** 31, 10))   # speakers
    assert not np.array_equal(seeds["eval"][0], seeds["eval"][1])
    assert np.array_equal(B.replicate_seeds(n_boot=10)["itw"][0], seeds["itw"][0])


def test_percentile_interval_and_columns():
    values = np.arange(1, 102, dtype=float) - 51.0
    low, high, se = B.percentile_interval(values)
    assert (low, high) == (-47.5, 47.5) and se == pytest.approx(np.std(values, ddof=1))
    columns = B.interval_columns("spk", values + 60.0)
    assert columns["spk_excludes_zero"] is True and columns["spk_ci_low"] == 12.5
    assert B.interval_columns("spk", values)["spk_excludes_zero"] is False
    touching = np.r_[np.zeros(50), np.ones(50)]   # lower bound exactly 0: zero is not excluded
    assert B.interval_columns("spk", touching)["spk_ci_low"] == 0.0
    assert B.interval_columns("spk", touching)["spk_excludes_zero"] is False
    assert B.interval_columns("spk", None)["spk_excludes_zero"] is None


# Score store and resamples -------------------------------------------------------------

def test_cells_are_read_from_the_run_of_the_training_condition(setup):
    runs_dir, store, _ = setup
    matrix, labels, ids = store.cell("clean", "trim40", "stats", 3, "eval")
    assert matrix.shape == (3, world.SPLITS["eval"]) and matrix.dtype == np.float64
    for k, seed in enumerate(world.SEEDS):
        sets, _ = T.load_scores(f"{runs_dir}/scores/wavlm-base_unpadded_clean_L03_stats_s{seed}.npz")
        assert np.array_equal(matrix[k], sets["eval@trim40"].scores.astype(np.float64))
    assert np.array_equal(ids, world.ids_of("eval"))
    with pytest.raises(KeyError):
        store.cell("clean", "trim20", "stats", 1, "eval")   # not in the design for layer 1


def test_resamples_use_the_reference_order_and_the_fixed_replicate_seeds(setup):
    _, store, resamples = setup
    entry = resamples.get("itw")
    assert entry["file"].shape == entry["spk"].shape == (N_BOOT, world.SPLITS["itw"])
    labels = world.labels_of("itw")
    assert np.array_equal(entry["labels"], labels) and np.array_equal(entry["ids"], world.ids_of("itw"))
    seeds = B.replicate_seeds(("eval", "itw"), N_BOOT)["itw"]
    assert np.array_equal(entry["file"][4], B.draw_file_counts(np.random.default_rng(seeds[0][4]), labels))
    assert (entry["file"][:, labels == 0].sum(axis=1) == (labels == 0).sum()).all()
    only_itw = B.Resamples(store, {"itw": world.speakers_of("itw")}, n_boot=N_BOOT)
    assert np.array_equal(only_itw.get("itw")["spk"], entry["spk"])
    assert B.Resamples(store, None, n_boot=N_BOOT).get("eval")["spk"] is None
    incomplete = B.Resamples(store, {"eval": {"eval_0000": "x"}}, n_boot=N_BOOT)
    with pytest.raises(KeyError):
        incomplete.get("eval")


def test_scores_in_another_file_order_are_aligned_to_the_reference(tmp_path):
    runs_dir = world.write_runs(str(tmp_path / "runs"))
    spec = T.RunSpec("wavlm-base", "unpadded", "trim40", 2, "mean", 43)
    path = f"{runs_dir}/scores/{spec.run_id}.npz"
    sets, meta = T.load_scores(path)
    order = np.random.default_rng(0).permutation(world.SPLITS["eval"])
    for name in sets:
        if name.startswith("eval@"):
            s = sets[name]
            sets[name] = T.ScoreSet(s.scores[order], s.labels[order], s.ids[order])
    store = B.ScoreStore(runs_dir, seeds=(43,))
    T.save_scores(path, sets, meta)
    resamples = B.Resamples(B.ScoreStore(runs_dir, seeds=world.SEEDS), n_boot=3)
    aligned = B.Resamples(store, n_boot=3, reference={"condition": "clean", "pooling": "stats",
                                                      "layer": 4}).matrix("trim40", "trim40", "mean", 2, "eval")
    expected = world.scores_of("trim40", "trim40", "mean", 2, 43, "eval").astype(np.float64)
    assert np.array_equal(aligned[0], expected)
    with pytest.raises(ValueError):
        resamples.matrix("trim40", "trim40", "mean", 2, "eval")   # seeds hold different orders


# Effects -------------------------------------------------------------------------------

def test_effect_definitions():
    cells = {"cc": 1.5, "ct": 7.75, "tc": 5.25, "tt": 6.0}
    values = {effect: E.weighted(cells, weights) for effect, weights in E.EFFECT_WEIGHTS.items()}
    assert values == {"delta_test": 6.25, "delta_train": -1.75, "delta_total": 4.5,
                      "delta_train_c": 3.75, "delta_test_t": 0.75, "interaction": -5.5}
    assert values["interaction"] == values["delta_test_t"] - values["delta_test"]
    assert values["interaction"] == values["delta_train"] - values["delta_train_c"]
    assert values["delta_total"] == values["delta_test"] + values["delta_train"]
    assert E.cell_conditions("trim20") == {"cc": ("clean", "clean"), "ct": ("clean", "trim20"),
                                           "tc": ("trim20", "clean"), "tt": ("trim20", "trim20")}
    assert list(E.EFFECT_WEIGHTS) == ["delta_test", "delta_train", "delta_total", "delta_train_c",
                                      "delta_test_t", "interaction"]
    with pytest.raises(ValueError):
        E.cell_conditions("clean")


def test_effects_table(setup):
    runs_dir, store, resamples = setup
    table = E.effects_ci(resamples, "trim40", layers=(2, 3))
    assert len(table) == 6 * 2 * 2 * 2 and set(table["n_boot"]) == {N_BOOT}
    row = table[(table["test_set"] == "eval") & (table["pooling"] == "stats")
                & (table["layer"] == 3)].set_index("effect")
    labels = world.labels_of("eval")
    eer = lambda c, t: np.mean([M.eer(labels, world.scores_of(c, t, "stats", 3, s, "eval"))
                                for s in world.SEEDS])
    cc, ct, tc, tt = eer("clean", "clean"), eer("clean", "trim40"), eer("trim40", "clean"), eer("trim40", "trim40")
    assert row.loc["delta_test", "eer_cc"] == cc and row.loc["delta_test", "eer_tt"] == tt
    assert row.loc["delta_test", "estimate"] == ct - cc
    assert row.loc["delta_total", "estimate"] == tt - cc
    assert row.loc["interaction", "estimate"] == pytest.approx(tt - tc - ct + cc, abs=1e-12)
    assert (table["file_ci_low"] <= table["file_ci_high"]).all()
    assert (table["spk_ci_low"] <= table["spk_ci_high"]).all()
    excludes = (table["spk_ci_low"] > 0) | (table["spk_ci_high"] < 0)
    assert (table["spk_excludes_zero"] == excludes).all()
    # A layer computed alone has the same rows as within a larger call.
    alone = E.effects_ci(resamples, "trim40", layers=(3,))
    both = table[table["layer"] == 3].reset_index(drop=True)
    pd.testing.assert_frame_equal(alone.sort_values(["test_set", "pooling", "effect"]).reset_index(drop=True),
                                  both.sort_values(["test_set", "pooling", "effect"]).reset_index(drop=True))


def test_threshold_condition_uses_its_cross_layers(setup):
    _, _, resamples = setup
    table = E.effects_ci(resamples, "trim20", layers=world.CROSS_LAYERS, test_sets=("itw",))
    assert set(table["condition"]) == {"trim20"} and len(table) == 6 * 2 * 2
    with pytest.raises(KeyError):
        E.effects_ci(resamples, "trim20", layers=(1,))


def test_cells_table_agrees_with_the_effects_table(setup):
    runs_dir, _, resamples = setup
    summary = S.seed_summary(S.metrics_per_run(runs_dir))
    cells = E.cells_table(summary, "trim40")
    assert len(cells) == 4 * len(world.LAYERS) * 2 and set(cells["n_seeds"]) == {3}
    assert set(E.cells_table(summary, "trim20")["layer"]) == set(world.CROSS_LAYERS)
    effects = E.effects_ci(resamples, "trim40", layers=world.LAYERS)
    for row in effects.drop_duplicates(["test_set", "pooling", "layer"]).itertuples():
        part = cells[(cells["pooling"] == row.pooling) & (cells["layer"] == row.layer)].set_index("cell")
        for cell in E.CELLS:
            assert part.loc[cell, f"{row.test_set}_eer_mean"] == pytest.approx(
                getattr(row, f"eer_{cell}"), abs=1e-9)


def test_selected_configs_intervals(setup):
    runs_dir, _, resamples = setup
    configs = S.selected_configs(S.seed_summary(S.metrics_per_run(runs_dir)), "wavlm-base")
    table = E.selected_configs_eer_ci(resamples, configs)
    assert len(table) == 2 * len(configs)
    row = table[(table["test_set"] == "itw")].iloc[0]
    labels = world.labels_of("itw")
    expected = np.mean([M.eer(labels, world.scores_of(row["condition"], row["condition"],
                                                      row["pooling"], row["layer"], s, "itw"))
                        for s in world.SEEDS])
    assert row["eer"] == expected and row["file_ci_low"] <= row["eer"] <= row["file_ci_high"]
    assert row["n_speakers_bonafide"] > 0 and "file_excludes_zero" not in table.columns


def test_per_attack_effect(setup):
    _, _, resamples = setup
    attack_map = world.attacks_of_eval()
    table = E.per_attack_effect_ci(resamples, "trim40", attack_map, layers=(3,), poolings=("stats",))
    assert table["attack"].tolist() == list(world.ATTACKS)
    labels, ids = world.labels_of("eval"), world.ids_of("eval")
    attack = np.array([attack_map[a] for a in ids])
    for row in table.itertuples():
        keep = (labels == 0) | (attack == row.attack)
        assert row.n_bonafide == (labels == 0).sum() and row.n_spoof == (attack == row.attack).sum()
        eer = lambda c: np.mean([M.eer(labels[keep], world.scores_of(c, c, "stats", 3, s, "eval")[keep])
                                 for s in world.SEEDS])
        assert row.eer_clean == eer("clean") and row.eer_trimmed == eer("trim40")
        assert row.estimate == row.eer_trimmed - row.eer_clean
        assert row.file_ci_low <= row.file_ci_high and row.spk_ci_low <= row.spk_ci_high
    wrong = dict(attack_map)
    wrong[ids[0]] = "A07" if labels[0] == 0 else "-"
    with pytest.raises(ValueError):
        E.per_attack_effect_ci(resamples, "trim40", wrong, layers=(3,), poolings=("stats",))
    assert E.attack_number("A19") == 19
    with pytest.raises(ValueError):
        E.attack_number("itw")


def test_itw_by_duration(setup):
    _, _, resamples = setup
    durations = world.durations_of_itw()
    table = E.itw_by_duration(resamples, "trim40", durations, layers=(4,), poolings=("stats",))
    assert set(table["subset"]) == {"all", "<=4s", ">4s"}
    sizes = table.drop_duplicates("subset").set_index("subset")["n_files"]
    assert sizes["<=4s"] + sizes[">4s"] == sizes["all"] == world.SPLITS["itw"]
    d = np.array([durations[a] for a in world.ids_of("itw")])
    assert sizes["<=4s"] == (d <= 4.0).sum()
    labels = world.labels_of("itw")
    short = d <= 4.0
    per_seed = lambda c, t: np.array([M.eer(labels[short], world.scores_of(c, t, "stats", 4, s, "itw")[short])
                                      for s in world.SEEDS])
    rows = table[table["subset"] == "<=4s"].set_index("quantity")
    assert rows.loc["eer_ct", "mean"] == per_seed("clean", "trim40").mean()
    delta = per_seed("clean", "trim40") - per_seed("clean", "clean")
    assert rows.loc["delta_test", "mean"] == delta.mean()
    assert rows.loc["delta_test", "sd"] == delta.std(ddof=1)
    assert set(rows.index) == {"eer_cc", "eer_ct", "eer_tc", "eer_tt", *E.EFFECT_WEIGHTS}
    assert (table["n_bonafide"] + table["n_spoof"] == table["n_files"]).all()


def test_the_total_effect_of_the_matched_cells_equals_the_effects_table(setup):
    resamples = setup[-1]
    total = E.total_effect_ci(resamples, "trim20", world.LAYERS)
    assert len(total) == 2 * 2 * len(world.LAYERS) and set(total["effect"]) == {"delta_total"}
    assert "eer_ct" not in total.columns and {"eer_cc", "eer_tt", "spk_ci_low"} <= set(total.columns)
    keys = ["condition", "test_set", "pooling", "layer", "effect"]
    effects = E.effects_ci(resamples, "trim20", world.CROSS_LAYERS)
    both = total.merge(effects[effects["effect"] == "delta_total"], on=keys, suffixes=("", "_all"))
    assert len(both) == 2 * 2 * len(world.CROSS_LAYERS)
    for column in ("estimate", "file_ci_low", "file_ci_high", "file_boot_se", "spk_ci_low",
                   "spk_ci_high", "spk_boot_se", "eer_cc", "eer_tt"):
        assert (both[column] == both[f"{column}_all"]).all(), column
    assert (both["spk_excludes_zero"] == both["spk_excludes_zero_all"]).all()
    # The other layers have no cross-condition cells: the effects table cannot be computed there.
    other = [layer for layer in world.LAYERS if layer not in world.CROSS_LAYERS]
    with pytest.raises(KeyError, match="has no score set"):
        E.effects_ci(resamples, "trim20", other)
    part = total[total["layer"].isin(other)]
    assert len(part) == 2 * 2 * len(other) and np.isfinite(part["spk_ci_low"]).all()
    assert (part["estimate"] == part["eer_tt"] - part["eer_cc"]).all()
