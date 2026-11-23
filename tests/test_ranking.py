"""Rank agreement, BCa intervals and the threshold summary, with known answers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import bootstrap, kendalltau, norm, spearmanr

import world
from ssl_silence import bootstrap as B
from ssl_silence import effects as E
from ssl_silence import metrics as M
from ssl_silence import ranking as R
from ssl_silence import selection as S

N_BOOT = 30


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    runs_dir = world.write_runs(str(tmp_path_factory.mktemp("world") / "runs"))
    speakers = {"eval": world.speakers_of("eval"), "itw": world.speakers_of("itw")}
    resamples = B.Resamples(B.ScoreStore(runs_dir, seeds=world.SEEDS), speakers, n_boot=N_BOOT)
    summary, detail = R.rank_agreement_ci(resamples, "trim40", layers=world.LAYERS)
    return runs_dir, resamples, summary, detail


# BCa ---------------------------------------------------------------------------------

def test_bca_interval_equals_scipy_on_a_skewed_mean():
    from scipy.stats._resampling import BootstrapResult, ConfidenceInterval

    rng = np.random.default_rng(0)
    x = rng.gamma(1.5, 2.0, 40)
    boots = x[rng.integers(0, x.size, (2000, x.size))].mean(axis=1)
    jack = np.array([np.delete(x, i).mean() for i in range(x.size)])
    mine = R.bca_interval(x.mean(), boots, jack)
    previous = BootstrapResult(confidence_interval=ConfidenceInterval(np.nan, np.nan),
                               bootstrap_distribution=boots, standard_error=boots.std(ddof=1))
    reference = bootstrap((x,), np.mean, n_resamples=0, bootstrap_result=previous, method="BCa",
                          rng=1)
    assert mine["low"] == pytest.approx(float(reference.confidence_interval.low), abs=1e-12)
    assert mine["high"] == pytest.approx(float(reference.confidence_interval.high), abs=1e-12)
    assert mine["z0"] == pytest.approx(norm.ppf(np.mean(boots < x.mean())))
    d = jack.mean() - jack
    assert mine["a"] == pytest.approx((d ** 3).sum() / (6 * (d ** 2).sum() ** 1.5))
    assert mine["z0"] != 0 and mine["a"] != 0 and mine["tie_share"] == 0


def test_bca_interval_special_cases():
    symmetric = np.linspace(-1, 1, 1001)
    flat = R.bca_interval(0.0, symmetric, np.zeros(5))   # no bias, no acceleration: percentile
    assert flat["z0"] == pytest.approx(0.0, abs=1e-12) and flat["a"] == 0
    assert flat["low"] == pytest.approx(np.percentile(symmetric, 2.5))
    assert flat["high"] == pytest.approx(np.percentile(symmetric, 97.5))
    tied = np.r_[np.full(40, -1.0), np.zeros(20), np.full(40, 1.0)]
    t = R.bca_interval(0.0, tied, np.array([0.1, 0.2, 0.3]))   # 40 % below, 20 % equal
    assert t["tie_share"] == pytest.approx(0.2) and t["z0"] == pytest.approx(0.0, abs=1e-12)
    degenerate = R.bca_interval(-5.0, symmetric, np.array([0.1, 0.2, 0.3]))  # nothing below
    assert np.isnan(degenerate["low"]) and np.isnan(degenerate["high"])
    assert degenerate["z0"] == -np.inf


def test_jackknife_leaves_one_speaker_out_within_class():
    labels = np.array([0, 0, 0, 1, 1, 1, 1])
    groups = np.array(["b", "a", "a", "a", "c", "c", "b"])
    counts, units = R.jackknife_speaker_counts(labels, groups)
    assert units == [(0, "a"), (0, "b"), (1, "a"), (1, "b"), (1, "c")]
    assert counts.tolist() == [[1, 0, 0, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 1, 1, 1],
                               [1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0, 1]]
    assert counts.dtype == np.int16


# Exact p of Kendall's tau ---------------------------------------------------------------

def test_exact_kendall_p_for_twelve_layers():
    a = np.arange(12.0)
    assert kendalltau(a, a, method="exact").pvalue == pytest.approx(2 / 479001600)   # 2 / 12!
    assert kendalltau(a, a[::-1], method="exact").statistic == pytest.approx(-1.0, abs=1e-12)
    # With 12 layers there are 66 pairs: 17 discordant pairs give tau = 32 / 66.
    b = np.array([5.0, 4, 3, 2, 1, 0, 7, 6, 9, 8, 10, 11])   # 15 + 1 + 1 inversions
    result = kendalltau(a, b, method="exact")
    assert result.statistic == pytest.approx(32 / 66, abs=1e-12)
    assert 0 < result.pvalue < 0.05
    with pytest.raises(ValueError):                           # no exact p with ties
        kendalltau(a, np.r_[0.0, 0.0, a[2:]], method="exact")


def test_exact_kendall_p_equals_the_enumeration_of_all_orders():
    from itertools import permutations

    a = np.arange(6.0)
    b = np.array([1.0, 0, 2, 4, 3, 5])                        # 2 discordant of 15 pairs
    observed = kendalltau(a, b, method="exact")
    assert observed.statistic == pytest.approx(11 / 15, abs=1e-12)
    taus = np.array([kendalltau(a, np.array(p, dtype=float))[0] for p in permutations(range(6))])
    share = np.mean(np.abs(taus) >= abs(observed.statistic) - 1e-12)
    assert observed.pvalue == pytest.approx(share, abs=1e-12)


# Rank agreement ------------------------------------------------------------------------

def test_rank_agreement_tables(setup):
    _, resamples, summary, detail = setup
    assert len(summary) == 4 and len(detail) == 4 * len(world.LAYERS)
    assert set(summary["n_boot"]) == {N_BOOT} and set(summary["n_layers"]) == {len(world.LAYERS)}
    row = summary[(summary["test_set"] == "eval") & (summary["pooling"] == "stats")].iloc[0]
    labels = world.labels_of("eval")
    curve = lambda c: np.array([np.mean([M.eer(labels, world.scores_of(c, c, "stats", layer, s, "eval"))
                                         for s in world.SEEDS]) for layer in world.LAYERS])
    clean, trimmed = curve("clean"), curve("trim40")
    assert row["kendall_tau"] == kendalltau(clean, trimmed)[0]
    assert row["spearman_rho"] == spearmanr(clean, trimmed)[0]
    assert row["best_layer_clean"] == world.LAYERS[int(np.argmin(clean))]
    assert row["best_layer_trimmed"] == world.LAYERS[int(np.argmin(trimmed))]
    assert row["tau_ci_low"] <= row["tau_ci_high"] and -1 <= row["tau_ci_low"]
    part = detail[(detail["test_set"] == "eval") & (detail["pooling"] == "stats")]
    assert np.array_equal(part["eer_clean"].to_numpy(), clean)
    assert part["p_best_clean"].sum() == pytest.approx(1.0)
    assert part["p_best_trimmed"].sum() == pytest.approx(1.0)
    n_units = sum(B.speakers_per_label(labels, np.array([world.speakers_of("eval")[a]
                                                         for a in world.ids_of("eval")])).values())
    assert row["n_jackknife"] == n_units
    assert row["n_speakers"] == len(set(world.speakers_of("eval").values()))


def test_exact_p_only_without_ties(setup):
    _, resamples, summary, detail = setup
    for row in summary.itertuples():
        part = detail[(detail["test_set"] == row.test_set) & (detail["pooling"] == row.pooling)]
        no_ties = part["eer_clean"].nunique() == len(part) and part["eer_trimmed"].nunique() == len(part)
        assert row.kendall_exact_applies == no_ties
        if no_ties:
            expected = kendalltau(part["eer_clean"], part["eer_trimmed"], method="exact").pvalue
            assert row.kendall_p_exact == expected
        else:
            assert np.isnan(row.kendall_p_exact)


def test_rank_resamples_are_one_stream_per_test_set_and_pooling(setup):
    _, resamples, _, _ = setup
    assert R.RANK_STREAM_OFFSETS == {("itw", "stats"): 1, ("itw", "mean"): 2,
                                     ("eval", "stats"): 3, ("eval", "mean"): 4}
    counts = R.rank_counts(resamples, "eval", "mean")
    labels = world.labels_of("eval")
    groups = np.array([world.speakers_of("eval")[a] for a in world.ids_of("eval")])
    rng = np.random.default_rng(2026 + 4)
    cls_groups = B.class_groups(labels, groups)
    expected = np.stack([B.draw_speaker_counts(rng, labels, cls_groups) for _ in range(N_BOOT)])
    assert np.array_equal(counts, expected)
    assert not np.array_equal(counts, R.rank_counts(resamples, "eval", "stats"))
    # The same resamples serve every trimmed condition: the clean curve is the same.
    s40 = R.rank_statistics(resamples, "trim40", "eval", "mean", world.LAYERS)
    s20 = R.rank_statistics(resamples, "trim20", "eval", "mean", world.LAYERS)
    assert np.array_equal(s40["best"]["clean"], s20["best"]["clean"])
    no_speakers = B.Resamples(resamples.store, None, n_boot=N_BOOT)
    with pytest.raises(ValueError):
        R.rank_counts(no_speakers, "eval", "mean")


# Noise ceiling -----------------------------------------------------------------------------

def test_a_split_puts_half_of_the_speakers_of_each_class_on_each_side():
    labels = np.array([0] * 7 + [1] * 9)
    groups = np.array(["a", "a", "b", "c", "c", "c", "d", "a", "a", "b", "b", "c", "e", "e", "f", "g"])
    cls_groups = B.class_groups(labels, groups)
    seen = set()
    for seed in range(20):
        half = R.draw_speaker_halves(np.random.default_rng(seed), labels, cls_groups)
        assert half.dtype == np.int16 and set(half) == {0, 1}
        for label, n_speakers in ((0, 4), (1, 6)):
            sides = {name: set(half[(labels == label) & (groups == name)])
                     for name in np.unique(groups[labels == label])}
            assert all(len(side) == 1 for side in sides.values())      # a speaker is on one side
            assert sum(side == {1} for side in sides.values()) == n_speakers // 2
        seen.add(tuple(half))
    assert len(seen) > 5                                               # the splits differ
    # Speaker "a" has both classes: the classes are split separately.
    odd = R.draw_speaker_halves(np.random.default_rng(0), np.array([0, 0, 0, 1, 1]),
                                B.class_groups(np.array([0, 0, 0, 1, 1]), np.array(list("abcab"))))
    assert odd[:3].sum() == 1 and odd[3:].sum() == 1                   # floor(3 / 2) and 1 of 2
    with pytest.raises(ValueError, match="fewer than two speakers"):
        R.draw_speaker_halves(np.random.default_rng(0), np.array([0, 0, 1]),
                              B.class_groups(np.array([0, 0, 1]), np.array(list("abc"))))


def test_the_splits_are_one_stream_per_test_set_and_pooling(setup):
    _, resamples, _, _ = setup
    assert R.SPLIT_STREAM_OFFSETS == {("itw", "stats"): 11, ("itw", "mean"): 12,
                                      ("eval", "stats"): 13, ("eval", "mean"): 14}
    assert not set(R.SPLIT_STREAM_OFFSETS.values()) & set(R.RANK_STREAM_OFFSETS.values())
    halves = R.split_halves(resamples, "eval", "mean")
    labels = world.labels_of("eval")
    groups = np.array([world.speakers_of("eval")[a] for a in world.ids_of("eval")])
    rng = np.random.default_rng(2026 + 14)
    cls_groups = B.class_groups(labels, groups)
    expected = np.stack([R.draw_speaker_halves(rng, labels, cls_groups) for _ in range(N_BOOT)])
    assert np.array_equal(halves, expected) and halves.shape == (N_BOOT, labels.size)
    assert not np.array_equal(halves, R.split_halves(resamples, "eval", "stats"))
    assert np.array_equal(R.split_halves(resamples, "eval", "mean", n_splits=7), expected[:7])
    with pytest.raises(ValueError, match="speakers are needed"):
        R.split_halves(B.Resamples(resamples.store, None, n_boot=N_BOOT), "eval", "mean")


def test_the_noise_ceiling_equals_tau_between_the_halves_computed_file_by_file(setup):
    _, resamples, summary, _ = setup
    ceiling, splits = R.noise_ceiling(resamples, "trim40", layers=world.LAYERS)
    assert len(ceiling) == 4 and len(splits) == 4 * N_BOOT
    assert list(ceiling.columns) == [
        "encoder", "condition", "test_set", "pooling", "n_layers", "n_speakers_bonafide",
        "n_speakers_spoof", "n_splits", "observed_tau", "ceiling_median", "ceiling_low",
        "ceiling_high", "cross_median", "cross_low", "cross_high",
        "share_ceiling_at_or_below_observed"]
    assert list(splits.columns) == ["encoder", "condition", "test_set", "pooling", "split",
                                    "tau_ceiling", "tau_cross"]
    # The tau of all files is that of the rank agreement table.
    merged = ceiling.merge(summary, on=["test_set", "pooling"])
    assert np.array_equal(merged["observed_tau"], merged["kendall_tau"]) and len(merged) == 4
    for test_set, pooling in (("eval", "mean"), ("itw", "stats")):
        labels = world.labels_of(test_set)
        halves = R.split_halves(resamples, test_set, pooling)

        def curve(condition, mask):
            return [np.mean([M.eer(labels[mask], world.scores_of(condition, condition, pooling, layer, s,
                                                                 test_set)[mask])
                             for s in world.SEEDS]) for layer in world.LAYERS]
        tau_ceiling, tau_cross = [], []
        for half in halves:
            a, b = half == 1, half == 0
            tau_ceiling.append(kendalltau(curve("clean", a), curve("clean", b))[0])
            tau_cross.append(kendalltau(curve("clean", a), curve("trim40", b))[0])
        part = splits[(splits["test_set"] == test_set) & (splits["pooling"] == pooling)]
        assert part["split"].tolist() == list(range(N_BOOT))
        assert part["tau_ceiling"].to_numpy() == pytest.approx(np.array(tau_ceiling), abs=1e-12)
        assert part["tau_cross"].to_numpy() == pytest.approx(np.array(tau_cross), abs=1e-12)
        row = ceiling[(ceiling["test_set"] == test_set) & (ceiling["pooling"] == pooling)].iloc[0]
        assert row["ceiling_median"] == pytest.approx(np.median(tau_ceiling))
        assert row["ceiling_low"] == pytest.approx(np.percentile(tau_ceiling, 2.5))
        assert row["ceiling_high"] == pytest.approx(np.percentile(tau_ceiling, 97.5))
        assert row["cross_median"] == pytest.approx(np.median(tau_cross))
        assert row["cross_low"] == pytest.approx(np.percentile(tau_cross, 2.5))
        assert row["cross_high"] == pytest.approx(np.percentile(tau_cross, 97.5))
        assert row["share_ceiling_at_or_below_observed"] == pytest.approx(
            np.mean(np.array(tau_ceiling) <= row["observed_tau"] + 1e-12))
        groups = np.array([world.speakers_of(test_set)[a] for a in world.ids_of(test_set)])
        assert row["n_speakers_bonafide"] == np.unique(groups[labels == 0]).size
        assert row["n_speakers_spoof"] == np.unique(groups[labels == 1]).size
        assert row["n_splits"] == N_BOOT and row["n_layers"] == len(world.LAYERS)


def test_with_the_same_condition_on_both_sides_the_cross_tau_is_the_ceiling(setup):
    _, resamples, _, _ = setup
    ceiling, splits = R.noise_ceiling(resamples, "clean", layers=world.LAYERS, n_splits=12,
                                      test_sets=("eval",), poolings=("stats",))
    assert len(ceiling) == 1 and len(splits) == 12
    assert np.array_equal(splits["tau_ceiling"], splits["tau_cross"])
    row = ceiling.iloc[0]
    assert row["observed_tau"] == 1.0 and row["share_ceiling_at_or_below_observed"] == 1.0
    assert (row["ceiling_median"], row["ceiling_low"], row["ceiling_high"]) == \
        (row["cross_median"], row["cross_low"], row["cross_high"])


# Threshold summary -------------------------------------------------------------------------

def test_threshold_summary(setup):
    runs_dir, resamples, summary, detail = setup
    summary20, detail20 = R.rank_agreement_ci(resamples, "trim20", layers=world.LAYERS)
    rank, rank_detail = pd.concat([summary, summary20]), pd.concat([detail, detail20])
    effects = pd.concat([E.effects_ci(resamples, "trim40", world.LAYERS),
                         E.effects_ci(resamples, "trim20", world.CROSS_LAYERS)])
    fallback = {"trim40": pd.DataFrame({"split": ["train", "eval"], "label": ["all", "all"],
                                        "n_fallback": [8, 21], "pct_fallback": [0.03, 0.029]}),
                "trim20": pd.DataFrame({"split": ["train", "train"], "label": ["0", "all"],
                                        "n_fallback": [5, 80], "pct_fallback": [0.2, 0.3]})}
    dev = S.select_on_dev(S.seed_summary(S.metrics_per_run(runs_dir)))
    table = R.threshold_summary(rank, rank_detail, effects, fallback, dev,
                                conditions=("trim40", "trim20"), layers=(3, 4),
                                splits=("train", "eval"))
    assert len(table) == 4 and table["top_db"].tolist() == [40, 40, 20, 20]
    row = table[(table["condition"] == "trim20") & (table["pooling"] == "stats")].iloc[0]
    r = summary20[(summary20["test_set"] == "itw") & (summary20["pooling"] == "stats")].iloc[0]
    assert row["kendall_tau_itw"] == r["kendall_tau"] and row["tau_ci_high_itw"] == r["tau_ci_high"]
    best = int(r["best_layer_trimmed"])
    d = detail20[(detail20["test_set"] == "itw") & (detail20["pooling"] == "stats")].set_index("layer")
    assert row["best_layer_itw_trimmed"] == best and row["p_best_itw_trimmed"] == d.loc[best, "p_best_trimmed"]
    e = effects[(effects["condition"] == "trim20") & (effects["test_set"] == "eval")
                & (effects["pooling"] == "stats") & (effects["layer"] == 4)].set_index("effect")
    assert row["delta_total_eval_L4"] == e.loc["delta_total", "estimate"]
    assert row["interaction_eval_L4_spk_ci_low"] == e.loc["interaction", "spk_ci_low"]
    assert row["n_fallback_train"] == 80 and np.isnan(row["n_fallback_eval"])
    chosen = dev[(dev["condition"] == "trim20") & (dev["pooling"] == "stats")]["layer"].iloc[0]
    assert row["dev_selected_layer"] == chosen
    assert table[table["condition"] == "trim40"]["n_fallback_eval"].tolist() == [21, 21]
