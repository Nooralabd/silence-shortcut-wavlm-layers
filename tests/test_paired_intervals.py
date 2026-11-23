"""Intervals of the duration split, the threshold difference of the rank
agreement and the paired encoder differences, with known answers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import world
from ssl_silence import bootstrap as B
from ssl_silence import config as cfg
from ssl_silence import effects as E
from ssl_silence import ranking as R
from ssl_silence import replication

N_BOOT = 30
OTHER = "hubert-base"
SPEAKERS = {"eval": world.speakers_of("eval"), "itw": world.speakers_of("itw")}
INTERVALS = ["estimate", "file_ci_low", "file_ci_high", "file_boot_se", "file_excludes_zero",
             "spk_ci_low", "spk_ci_high", "spk_boot_se", "spk_excludes_zero"]


def resamples_of(runs_dir, encoder=world.ENCODER, speakers=SPEAKERS, n_boot=N_BOOT, **kwargs):
    return B.Resamples(B.ScoreStore(runs_dir, encoder=encoder, seeds=world.SEEDS), speakers,
                       n_boot=n_boot, **kwargs)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    base = tmp_path_factory.mktemp("world")
    return {"a": world.write_runs(str(base / "a")),
            "b": world.write_runs(str(base / "b"), encoder=OTHER),
            "reversed": world.write_runs(str(base / "reversed"), encoder=OTHER, reverse_files=True)}


@pytest.fixture(scope="module")
def duration(runs):
    resamples = resamples_of(runs["a"])
    by_subset, difference = E.itw_by_duration_ci(resamples, "trim40", world.durations_of_itw(),
                                                 layers=world.CROSS_LAYERS)
    return resamples, by_subset, difference


# Duration split ---------------------------------------------------------------------------

def test_the_subset_all_equals_the_effects_of_the_whole_test_set(duration):
    resamples, by_subset, _ = duration
    whole = E.effects_ci(resamples, "trim40", world.CROSS_LAYERS, test_sets=("itw",))
    keys = ["pooling", "layer", "effect"]
    part = by_subset[by_subset["subset"] == "all"].set_index(keys)
    whole = whole.set_index(keys).loc[part.index]
    assert len(part) == len(cfg.EFFECTS) * len(world.CROSS_LAYERS) * len(world.POOLINGS)
    for column in INTERVALS + ["n_files", "n_boot"]:
        assert part[column].tolist() == whole[column].tolist(), column


def test_a_subset_uses_the_resampled_files_of_that_subset(duration):
    resamples, by_subset, _ = duration
    entry = resamples.get("itw")
    labels = entry["labels"]
    durations = np.array([world.durations_of_itw()[i] for i in entry["ids"]])
    mask = durations > 4.0
    assert 0 < mask.sum() < mask.size
    cells = {"cc": ("clean", "clean"), "ct": ("clean", "trim40")}
    boots = {level: {} for level in ("file", "spk")}
    observed = {}
    for cell, (train, test) in cells.items():
        matrix = resamples.matrix(train, test, "stats", 3, "itw")[:, mask]
        observed[cell] = np.mean([E.M.eer(labels[mask], row) for row in matrix])
        for level in boots:
            counts = entry[level][:, mask]
            boots[level][cell] = np.mean([B.resampled_eer(labels[mask], row, counts)
                                          for row in matrix], axis=0)
    row = by_subset[(by_subset["subset"] == ">4s") & (by_subset["pooling"] == "stats")
                    & (by_subset["layer"] == 3) & (by_subset["effect"] == "delta_test")].iloc[0]
    assert row["estimate"] == pytest.approx(observed["ct"] - observed["cc"], abs=1e-12)
    for level in boots:
        values = boots[level]["ct"] - boots[level]["cc"]
        assert row[f"{level}_ci_low"] == pytest.approx(np.percentile(values, 2.5), abs=1e-12)
        assert row[f"{level}_ci_high"] == pytest.approx(np.percentile(values, 97.5), abs=1e-12)
    assert row["n_files"] == mask.sum() and row["n_spoof"] == (labels[mask] == 1).sum()
    # The number of files of the subset varies between resamples.
    assert len(set(entry["file"][:, mask].sum(axis=1).tolist())) > 1


def test_duration_estimates_equal_the_table_without_intervals(duration):
    resamples, by_subset, _ = duration
    plain = E.itw_by_duration(resamples, "trim40", world.durations_of_itw(),
                              layers=world.CROSS_LAYERS)
    plain = plain.set_index(["pooling", "layer", "subset", "quantity"])
    assert set(by_subset["subset"]) == {"all", "<=4s", ">4s"}
    for row in by_subset.itertuples():
        reference = plain.loc[(row.pooling, row.layer, row.subset, row.effect)]
        assert row.estimate == pytest.approx(reference["mean"], abs=1e-10)
        assert (row.n_files, row.n_bonafide, row.n_spoof) == tuple(
            reference[["n_files", "n_bonafide", "n_spoof"]])


def test_difference_between_the_subsets_is_paired(duration):
    _, by_subset, difference = duration
    assert len(difference) == len(cfg.EFFECTS) * len(world.CROSS_LAYERS) * len(world.POOLINGS)
    assert set(difference["subset_long"]) == {">4s"} and set(difference["subset_short"]) == {"<=4s"}
    part = by_subset.set_index(["pooling", "layer", "effect", "subset"])["estimate"]
    for row in difference.itertuples():
        key = (row.pooling, row.layer, row.effect)
        assert row.estimate_long == part[(*key, ">4s")]
        assert row.estimate_short == part[(*key, "<=4s")]
        assert row.estimate == row.estimate_long - row.estimate_short
        assert row.file_ci_low <= row.file_ci_high and row.spk_ci_low <= row.spk_ci_high
    # Paired: the interval is that of the difference inside every resample, which
    # is not the difference of the two intervals.
    row = difference.iloc[0]
    bounds = by_subset.set_index(["pooling", "layer", "effect", "subset"])
    key = (row["pooling"], row["layer"], row["effect"])
    naive = bounds.loc[(*key, ">4s"), "spk_ci_low"] - bounds.loc[(*key, "<=4s"), "spk_ci_low"]
    assert row["spk_ci_low"] != naive


def test_duration_without_speakers_and_units(duration, runs):
    _, by_subset, difference = duration
    no_speakers, _ = E.itw_by_duration_ci(resamples_of(runs["a"], speakers=None), "trim40",
                                          world.durations_of_itw(), layers=(3,),
                                          poolings=("stats",))
    assert no_speakers["spk_ci_low"].isna().all() and no_speakers["file_ci_low"].notna().all()
    units = replication.duration_units(by_subset, condition="trim40", layers=world.CROSS_LAYERS)
    # Two subsets; neither the whole-set rows nor the difference table add a unit.
    assert len(units) == (by_subset["subset"] != "all").sum() == len(by_subset) * 2 // 3
    assert "all" not in set(units["subset"])
    assert "subset" not in difference.columns


def test_a_subset_that_loses_a_class_stops(duration):
    resamples, _, _ = duration
    entry = resamples.get("itw")
    bonafide = np.flatnonzero(entry["labels"] == 0)
    durations = {str(i): 9.0 for i in entry["ids"]}
    durations[str(entry["ids"][bonafide[0]])] = 1.0          # one short bonafide file
    for index in np.flatnonzero(entry["labels"] == 1)[:20]:
        durations[str(entry["ids"][index])] = 1.0
    with pytest.raises(ValueError, match="no bonafide file"):
        E.itw_by_duration_ci(resamples, "trim40", durations, layers=(3,), poolings=("stats",))
    counts = np.array([[1, 0, 2], [0, 0, 3]], dtype=np.int16)
    E.require_both_classes(np.array([0, 0, 1]), counts[:1], "x")
    with pytest.raises(ValueError, match="1 resamples have no bonafide"):
        E.require_both_classes(np.array([0, 0, 1]), counts, "x")
    with pytest.raises(ValueError, match="no spoof"):
        E.require_both_classes(np.array([0, 1, 0]), counts, "x")


# Threshold difference of the rank agreement ----------------------------------------------------

def test_tau_difference_is_computed_inside_every_resample(runs):
    resamples = resamples_of(runs["a"])
    statistics = {}
    table = R.tau_difference_ci(resamples, "trim20", layers=world.LAYERS, statistics=statistics)
    rank40, _ = R.rank_agreement_ci(resamples, "trim40", layers=world.LAYERS, statistics=statistics)
    rank20, _ = R.rank_agreement_ci(resamples, "trim20", layers=world.LAYERS, statistics=statistics)
    assert len(statistics) == 8 and len(table) == 4          # nothing computed twice
    keys = ["test_set", "pooling"]
    rank40, rank20 = rank40.set_index(keys), rank20.set_index(keys)
    for row in table.itertuples():
        key = (row.test_set, row.pooling)
        assert row.tau_a == rank40.loc[key, "kendall_tau"]
        assert row.tau_b == rank20.loc[key, "kendall_tau"]
        assert row.estimate == row.tau_a - row.tau_b
        assert row.rho_estimate == (rank40.loc[key, "spearman_rho"]
                                    - rank20.loc[key, "spearman_rho"])
        a = R.rank_statistics(resamples, "trim40", row.test_set, row.pooling, world.LAYERS)
        b = R.rank_statistics(resamples, "trim20", row.test_set, row.pooling, world.LAYERS)
        values = a["taus"] - b["taus"]
        assert row.spk_ci_low == np.percentile(values, 2.5)
        assert row.spk_ci_high == np.percentile(values, 97.5)
        assert row.spk_excludes_zero == bool(row.spk_ci_low > 0 or row.spk_ci_high < 0)
        assert row.rho_spk_ci_high == np.percentile(a["rhos"] - b["rhos"], 97.5)
    assert set(table["reference"]) == {"trim40"} and set(table["condition"]) == {"trim20"}
    units = replication.tau_difference_units(table, "trim20")
    assert len(units) == 4
    with pytest.raises(ValueError):
        R.tau_difference_ci(resamples, "trim40", layers=world.LAYERS)


# Paired encoder differences ---------------------------------------------------------------------

def test_an_encoder_minus_itself_is_zero(runs):
    a, again = resamples_of(runs["a"]), resamples_of(runs["a"])
    table = E.encoder_differences(a, again, "trim40", world.CROSS_LAYERS)
    assert len(table) == 2 * 2 * len(world.CROSS_LAYERS) * (4 + len(cfg.EFFECTS))
    for column in ("estimate", "file_ci_low", "file_ci_high", "spk_ci_low", "spk_ci_high"):
        assert (table[column] == 0).all()
    tau = R.encoder_tau_differences(a, again, "trim40", layers=world.LAYERS)
    assert (tau[["estimate", "spk_ci_low", "spk_ci_high", "rho_estimate"]] == 0).all().all()


def test_encoder_differences_are_paired(runs):
    a, b = resamples_of(runs["a"]), resamples_of(runs["b"], encoder=OTHER)
    table = E.encoder_differences(a, b, "trim40", world.CROSS_LAYERS)
    assert set(table["encoder_a"]) == {world.ENCODER} and set(table["encoder_b"]) == {OTHER}
    keys = ["test_set", "pooling", "layer", "effect"]
    effects_a = E.effects_ci(a, "trim40", world.CROSS_LAYERS).set_index(keys)
    effects_b = E.effects_ci(b, "trim40", world.CROSS_LAYERS).set_index(keys)
    for row in table[table["quantity"].isin(cfg.EFFECTS)].itertuples():
        key = (row.test_set, row.pooling, row.layer, row.quantity)
        assert row.estimate_a == effects_a.loc[key, "estimate"]
        assert row.estimate_b == effects_b.loc[key, "estimate"]
        assert row.estimate == row.estimate_a - row.estimate_b
    cell = table[(table["quantity"] == "eer_ct") & (table["test_set"] == "eval")
                 & (table["pooling"] == "stats") & (table["layer"] == 4)].iloc[0]
    assert cell["estimate_a"] == effects_a.loc[("eval", "stats", 4, "delta_test"), "eer_ct"]
    stats_a = E.cell_statistics(a, "trim40", "stats", 4, "eval")
    stats_b = E.cell_statistics(b, "trim40", "stats", 4, "eval")
    for level in ("file", "spk"):
        values = stats_a[level]["ct"] - stats_b[level]["ct"]
        assert cell[f"{level}_ci_low"] == np.percentile(values, 2.5)
        assert cell[f"{level}_ci_high"] == np.percentile(values, 97.5)
    assert (table["estimate"] != 0).any()

    tau = R.encoder_tau_differences(a, b, "trim40", layers=world.LAYERS)
    rank_a = R.rank_agreement_ci(a, "trim40", layers=world.LAYERS)[0].set_index(["test_set", "pooling"])
    rank_b = R.rank_agreement_ci(b, "trim40", layers=world.LAYERS)[0].set_index(["test_set", "pooling"])
    for row in tau.itertuples():
        key = (row.test_set, row.pooling)
        assert row.tau_a == rank_a.loc[key, "kendall_tau"]
        assert row.tau_b == rank_b.loc[key, "kendall_tau"]
        assert row.estimate == row.tau_a - row.tau_b
        values = (R.rank_statistics(a, "trim40", *key, world.LAYERS)["taus"]
                  - R.rank_statistics(b, "trim40", *key, world.LAYERS)["taus"])
        assert row.spk_ci_low == np.percentile(values, 2.5)
    assert (tau["estimate"] != 0).any()


def test_encoders_that_are_not_paired_stop(runs):
    a = resamples_of(runs["a"])
    with pytest.raises(ValueError, match="file orders"):
        E.encoder_differences(a, resamples_of(runs["reversed"], encoder=OTHER), "trim40", (3,))
    with pytest.raises(ValueError, match="number of resamples"):
        E.encoder_differences(a, resamples_of(runs["b"], encoder=OTHER, n_boot=N_BOOT + 1),
                              "trim40", (3,))
    with pytest.raises(ValueError, match="number of resamples or seed"):
        R.encoder_tau_differences(a, resamples_of(runs["b"], encoder=OTHER, seed=7), "trim40",
                                  layers=world.LAYERS)
    other_speakers = {"eval": SPEAKERS["eval"],
                      "itw": {k: "one" for k in SPEAKERS["itw"]}}
    with pytest.raises(ValueError, match="speakers"):
        E.encoder_differences(a, resamples_of(runs["b"], encoder=OTHER, speakers=other_speakers),
                              "trim40", (3,), test_sets=("itw",))
    B.require_paired(a, resamples_of(runs["b"], encoder=OTHER), "eval")
