"""Agreement between the layer rankings of two conditions.

For a test set and a pooling method, the seed-mean EER of every layer in the
clean-to-clean cell and in the trimmed-to-trimmed cell gives two curves.
Their agreement is Kendall's tau and Spearman's rho, with speaker-level
percentile and BCa bootstrap intervals; the exact two-sided p of tau is given
when neither curve has ties. In every resample the best layer (lowest EER;
the lower layer on a tie) is recorded.

Resamples: one generator per (test set, pooling), seeded with the bootstrap
seed plus RANK_STREAM_OFFSETS, drawing ``n_boot`` class-stratified speaker
resamples one after the other. The same resamples are used for both
conditions, for every layer and for every trimmed condition.

The noise ceiling (``noise_ceiling``) asks how well two rankings agree when
only the speakers differ: the speakers are split into two halves within
each class, and tau is computed between the clean curves of the two halves
and, in the same splits, between the clean curve of one half and the
trimmed curve of the other. Its splits come from their own streams
(SPLIT_STREAM_OFFSETS).
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from . import config as cfg
from .bootstrap import (Resamples, class_groups, draw_speaker_counts, interval_columns,
                        require_paired, seed_mean_eer, seed_mean_resampled_eer, sequential_counts)
from .effects import BASE, EFFECT_WEIGHTS

RANK_STREAM_OFFSETS = {("itw", "stats"): 1, ("itw", "mean"): 2,
                       ("eval", "stats"): 3, ("eval", "mean"): 4}
# Streams of the speaker-disjoint halves of the noise ceiling.
SPLIT_STREAM_OFFSETS = {("itw", "stats"): 11, ("itw", "mean"): 12,
                        ("eval", "stats"): 13, ("eval", "mean"): 14}
RANK_TEST_SETS = ("itw", "eval")
RANK_POOLINGS = ("stats", "mean")
SUMMARY_LAYERS = (3, 4, 5)


# BCa ---------------------------------------------------------------------------------

def jackknife_speaker_counts(labels: np.ndarray, groups: np.ndarray
                             ) -> tuple[np.ndarray, list[tuple[int, str]]]:
    """Leave-one-speaker-out within class, as multiplicities.

    Returns (counts (n_units, N) int16, units). Row i is 1 for every file
    except the files of unit i. A unit is (label, speaker): bonafide speakers
    first, then spoof speakers, each in sorted order.
    """
    labels = np.asarray(labels)
    groups = np.asarray(groups).astype(str)
    units, rows = [], []
    for label, group in zip((cfg.LABEL_BONAFIDE, cfg.LABEL_SPOOF), class_groups(labels, groups)):
        names = np.unique(groups[labels == label])
        for name, index in zip(names, group):
            row = np.ones(labels.size, dtype=np.int16)
            row[index] = 0
            rows.append(row)
            units.append((int(label), str(name)))
    return np.stack(rows), units


def bca_interval(observed: float, boots: np.ndarray, jackknife: np.ndarray,
                 alpha: float = 0.05) -> dict[str, float]:
    """Bias-corrected and accelerated bootstrap interval (Efron 1987).

    z0 is the normal quantile of the share of bootstrap values below
    ``observed``, ties counted one half (``tie_share`` is the share equal to
    it). The acceleration is a = sum(d^3) / (6 sum(d^2)^1.5) with d =
    mean(jackknife) - jackknife, and 0 when all jackknife values are equal.
    The bounds are percentiles of ``boots`` at the corrected levels, NaN when
    a level is not finite.

    Returns low, high, z0, a, tie_share, level_low, level_high.
    """
    from scipy.stats import norm

    boots = np.asarray(boots, dtype=np.float64)
    jack = np.asarray(jackknife, dtype=np.float64)
    n_b = boots.size
    below = int(np.count_nonzero(boots < observed))
    ties = int(np.count_nonzero(boots == observed))
    p0 = (below + 0.5 * ties) / n_b
    z0 = float(norm.ppf(p0))
    d = jack.mean() - jack
    den = float(np.sum(d ** 2)) ** 1.5
    a = float(np.sum(d ** 3) / (6.0 * den)) if den > 0 else 0.0
    z = norm.ppf([alpha / 2, 1 - alpha / 2])
    with np.errstate(invalid="ignore", divide="ignore"):
        num = z0 + z
        levels = norm.cdf(z0 + num / (1 - a * num))
    if np.isfinite(levels).all():
        low, high = (float(v) for v in np.percentile(boots, 100 * levels))
    else:
        low = high = float("nan")
    return {"low": low, "high": high, "z0": z0, "a": a, "tie_share": ties / n_b,
            "level_low": float(levels[0]), "level_high": float(levels[1])}


# Rank agreement ------------------------------------------------------------------------

def rank_counts(resamples: Resamples, test_set: str, pooling: str) -> np.ndarray:
    """Speaker-level resamples (n_boot, N) of the rank agreement of one test set and pooling."""
    entry = resamples.get(test_set)
    if entry["groups"] is None:
        raise ValueError("the rank agreement is a speaker-level bootstrap: speakers are needed")
    labels = entry["labels"]
    cls_groups = class_groups(labels, entry["groups"])
    rng = np.random.default_rng(resamples.seed + RANK_STREAM_OFFSETS[(test_set, pooling)])
    return sequential_counts(rng, resamples.n_boot, labels,
                             lambda r: draw_speaker_counts(r, labels, cls_groups))


def rank_statistics(resamples: Resamples, condition: str, test_set: str, pooling: str,
                    layers: Sequence[int], base: str = BASE) -> dict:
    """Curves and rank statistics of one condition, test set and pooling.

    Returns full (condition -> seed-mean EER per layer), tau, p_tau, rho,
    p_rho, taus and rhos (bootstrap values), best (condition -> best layer in
    every resample), jack_tau and jack_rho (jackknife values) and n_units.
    """
    from scipy.stats import kendalltau, spearmanr

    entry = resamples.get(test_set)
    labels = entry["labels"]
    layers = [int(layer) for layer in layers]
    conditions = (base, condition)
    matrices = {c: [resamples.matrix(c, c, pooling, layer, test_set) for layer in layers]
                for c in conditions}
    full = {c: np.array([seed_mean_eer(m, labels) for m in matrices[c]]) for c in conditions}
    tau, p_tau = kendalltau(full[base], full[condition])
    rho, p_rho = spearmanr(full[base], full[condition])
    counts = rank_counts(resamples, test_set, pooling)
    curves = {c: np.stack([seed_mean_resampled_eer(m, labels, counts) for m in matrices[c]], axis=1)
              for c in conditions}                                  # (n_boot, n_layers)
    n_boot = counts.shape[0]
    taus = np.array([kendalltau(curves[base][b], curves[condition][b])[0] for b in range(n_boot)])
    rhos = np.array([spearmanr(curves[base][b], curves[condition][b])[0] for b in range(n_boot)])
    best = {c: np.array(layers)[np.argmin(curves[c], axis=1)] for c in conditions}
    jack_counts, units = jackknife_speaker_counts(labels, entry["groups"])
    jack = {c: np.stack([seed_mean_resampled_eer(m, labels, jack_counts) for m in matrices[c]],
                        axis=1) for c in conditions}                # (n_units, n_layers)
    jack_tau = np.array([kendalltau(jack[base][i], jack[condition][i])[0]
                         for i in range(len(units))])
    jack_rho = np.array([spearmanr(jack[base][i], jack[condition][i])[0]
                         for i in range(len(units))])
    return {"layers": layers, "full": full, "tau": float(tau), "p_tau": float(p_tau),
            "rho": float(rho), "p_rho": float(p_rho), "taus": taus, "rhos": rhos, "best": best,
            "jack_tau": jack_tau, "jack_rho": jack_rho, "n_units": len(units)}


def cached_statistics(statistics: dict | None, resamples: Resamples, condition: str,
                      test_set: str, pooling: str, layers: Sequence[int], base: str = BASE) -> dict:
    """``rank_statistics``, kept in ``statistics`` (when given) so that the
    tables that need the same bootstrap values do not compute them again."""
    key = (resamples.store.encoder, condition, test_set, pooling,
           tuple(int(layer) for layer in layers), base)
    if statistics is None:
        return rank_statistics(resamples, condition, test_set, pooling, layers, base)
    if key not in statistics:
        statistics[key] = rank_statistics(resamples, condition, test_set, pooling, layers, base)
    return statistics[key]


def rank_agreement_ci(resamples: Resamples, condition: str,
                      layers: Sequence[int] = tuple(range(1, 13)),
                      poolings: Sequence[str] = RANK_POOLINGS,
                      test_sets: Sequence[str] = RANK_TEST_SETS, base: str = BASE,
                      statistics: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank agreement between the clean and the ``condition`` layer curves.

    ``statistics``: an optional dict in which the bootstrap values are kept
    for ``tau_difference_ci`` and ``encoder_tau_differences``.

    Returns (summary, detail). summary, one row per test set and pooling:
    encoder, condition, test_set, pooling, n_layers, n_speakers, n_jackknife,
    n_boot, kendall_tau, tau_ci_low, tau_ci_high, tau_bca_low, tau_bca_high,
    z0_tau, a_tau, tie_share_tau, kendall_p, kendall_p_exact,
    kendall_exact_applies, spearman_rho, rho_ci_low, rho_ci_high,
    rho_bca_low, rho_bca_high, z0_rho, a_rho, spearman_p, best_layer_clean,
    best_layer_trimmed. detail, one row per layer: encoder, condition,
    test_set, pooling, layer, eer_clean, eer_trimmed, p_best_clean,
    p_best_trimmed (share of resamples in which the layer is best).
    """
    from scipy.stats import kendalltau

    summary, detail = [], []
    for test_set in test_sets:
        n_speakers = int(np.unique(resamples.get(test_set)["groups"]).size) \
            if resamples.get(test_set)["groups"] is not None else 0
        for pooling in poolings:
            s = cached_statistics(statistics, resamples, condition, test_set, pooling, layers, base)
            full, n_layers = s["full"], len(s["layers"])
            tau_bca = bca_interval(s["tau"], s["taus"], s["jack_tau"])
            rho_bca = bca_interval(s["rho"], s["rhos"], s["jack_rho"])
            no_ties = (np.unique(full[base]).size == n_layers
                       and np.unique(full[condition]).size == n_layers)
            p_exact = (float(kendalltau(full[base], full[condition], method="exact").pvalue)
                       if no_ties else float("nan"))
            fixed = {"encoder": resamples.store.encoder, "condition": condition,
                     "test_set": test_set, "pooling": pooling}
            summary.append({
                **fixed, "n_layers": n_layers, "n_speakers": n_speakers,
                "n_jackknife": s["n_units"], "n_boot": resamples.n_boot,
                "kendall_tau": s["tau"],
                "tau_ci_low": float(np.percentile(s["taus"], 2.5)),
                "tau_ci_high": float(np.percentile(s["taus"], 97.5)),
                "tau_bca_low": tau_bca["low"], "tau_bca_high": tau_bca["high"],
                "z0_tau": tau_bca["z0"], "a_tau": tau_bca["a"],
                "tie_share_tau": tau_bca["tie_share"], "kendall_p": s["p_tau"],
                "kendall_p_exact": p_exact, "kendall_exact_applies": bool(no_ties),
                "spearman_rho": s["rho"],
                "rho_ci_low": float(np.percentile(s["rhos"], 2.5)),
                "rho_ci_high": float(np.percentile(s["rhos"], 97.5)),
                "rho_bca_low": rho_bca["low"], "rho_bca_high": rho_bca["high"],
                "z0_rho": rho_bca["z0"], "a_rho": rho_bca["a"], "spearman_p": s["p_rho"],
                "best_layer_clean": s["layers"][int(np.argmin(full[base]))],
                "best_layer_trimmed": s["layers"][int(np.argmin(full[condition]))]})
            for k, layer in enumerate(s["layers"]):
                detail.append({**fixed, "layer": layer, "eer_clean": float(full[base][k]),
                               "eer_trimmed": float(full[condition][k]),
                               "p_best_clean": float(np.mean(s["best"][base] == layer)),
                               "p_best_trimmed": float(np.mean(s["best"][condition] == layer))})
    return pd.DataFrame(summary), pd.DataFrame(detail)


def _paired_rank_columns(a: dict, b: dict) -> dict:
    """Estimates and speaker-level percentile intervals of tau and rho of
    ``a`` minus ``b``, both computed in the same resamples."""
    tau = interval_columns("spk", a["taus"] - b["taus"])
    rho = interval_columns("rho_spk", a["rhos"] - b["rhos"])
    return {"tau_a": a["tau"], "tau_b": b["tau"], "estimate": a["tau"] - b["tau"], **tau,
            "rho_a": a["rho"], "rho_b": b["rho"], "rho_estimate": a["rho"] - b["rho"], **rho}


def tau_difference_ci(resamples: Resamples, condition: str, reference: str = "trim40",
                      layers: Sequence[int] = tuple(range(1, 13)),
                      poolings: Sequence[str] = RANK_POOLINGS,
                      test_sets: Sequence[str] = RANK_TEST_SETS, base: str = BASE,
                      statistics: dict | None = None) -> pd.DataFrame:
    """Rank agreement of ``reference`` minus rank agreement of ``condition``.

    tau(clean, reference) - tau(clean, condition), and the same for rho, per
    test set and pooling. Both values are computed in each of the same
    speaker-level resamples (the resamples of ``rank_counts``); percentile
    interval of the differences.

    Columns: encoder, condition, reference, test_set, pooling, n_layers,
    n_boot, tau_a (reference), tau_b (condition), estimate, spk_ci_low,
    spk_ci_high, spk_boot_se, spk_excludes_zero, rho_a, rho_b, rho_estimate,
    rho_spk_ci_low, rho_spk_ci_high, rho_spk_boot_se, rho_spk_excludes_zero.
    """
    if condition == reference:
        raise ValueError("the condition must differ from the reference condition")
    rows = []
    for test_set in test_sets:
        for pooling in poolings:
            a = cached_statistics(statistics, resamples, reference, test_set, pooling, layers, base)
            b = cached_statistics(statistics, resamples, condition, test_set, pooling, layers, base)
            if not (np.array_equal(a["full"][base], b["full"][base])
                    and np.array_equal(a["best"][base], b["best"][base])):
                raise ValueError("the clean curves of the two conditions differ: the resamples "
                                 "are not paired")
            rows.append({"encoder": resamples.store.encoder, "condition": condition,
                         "reference": reference, "test_set": test_set, "pooling": pooling,
                         "n_layers": len(a["layers"]), "n_boot": resamples.n_boot,
                         **_paired_rank_columns(a, b)})
    return pd.DataFrame(rows)


def encoder_tau_differences(resamples_a: Resamples, resamples_b: Resamples, condition: str,
                            layers: Sequence[int] = tuple(range(1, 13)),
                            poolings: Sequence[str] = RANK_POOLINGS,
                            test_sets: Sequence[str] = RANK_TEST_SETS, base: str = BASE,
                            statistics: dict | None = None) -> pd.DataFrame:
    """Rank agreement of encoder a minus rank agreement of encoder b, paired.

    Both encoders are evaluated on the same files and the same speaker-level
    resamples (``bootstrap.require_paired``).

    Columns: encoder_a, encoder_b, condition, test_set, pooling, n_layers,
    n_boot, tau_a, tau_b, estimate, spk_ci_low, spk_ci_high, spk_boot_se,
    spk_excludes_zero, rho_a, rho_b, rho_estimate and the rho interval columns.
    """
    rows = []
    for test_set in test_sets:
        require_paired(resamples_a, resamples_b, test_set)
        for pooling in poolings:
            a = cached_statistics(statistics, resamples_a, condition, test_set, pooling, layers, base)
            b = cached_statistics(statistics, resamples_b, condition, test_set, pooling, layers, base)
            rows.append({"encoder_a": resamples_a.store.encoder,
                         "encoder_b": resamples_b.store.encoder, "condition": condition,
                         "test_set": test_set, "pooling": pooling, "n_layers": len(a["layers"]),
                         "n_boot": resamples_a.n_boot, **_paired_rank_columns(a, b)})
    return pd.DataFrame(rows)


# Noise ceiling of the rank agreement -----------------------------------------------------------

def draw_speaker_halves(rng: np.random.Generator, labels: np.ndarray,
                        cls_groups: Sequence[Sequence[np.ndarray]]) -> np.ndarray:
    """One random split of the speakers into two halves, within each class.

    Returns (N,) int16: 1 for the files of half A, 0 for those of half B. In
    each class the speakers are put in a random order; the first
    floor(n / 2) go to half A and the others to half B, each with all its
    files of that class. A class with fewer than two speakers raises.
    """
    half = np.zeros(labels.size, dtype=np.int16)
    for group in cls_groups:
        if len(group) < 2:
            raise ValueError("a class with fewer than two speakers cannot be split into two halves")
        order = rng.permutation(len(group))
        for i in order[:len(group) // 2]:
            half[group[i]] = 1
    return half


def split_halves(resamples: Resamples, test_set: str, pooling: str,
                 n_splits: int | None = None) -> np.ndarray:
    """The splits (n_splits, N) of one test set and pooling: 1 for half A.

    One generator per (test set, pooling), seeded with the bootstrap seed
    plus SPLIT_STREAM_OFFSETS; ``n_splits`` is the number of resamples
    unless given.
    """
    entry = resamples.get(test_set)
    if entry["groups"] is None:
        raise ValueError("the noise ceiling splits the speakers: speakers are needed")
    labels = entry["labels"]
    cls_groups = class_groups(labels, entry["groups"])
    rng = np.random.default_rng(resamples.seed + SPLIT_STREAM_OFFSETS[(test_set, pooling)])
    return sequential_counts(rng, resamples.n_boot if n_splits is None else int(n_splits), labels,
                             lambda r: draw_speaker_halves(r, labels, cls_groups))


def noise_ceiling(resamples: Resamples, condition: str,
                  layers: Sequence[int] = tuple(range(1, 13)),
                  poolings: Sequence[str] = RANK_POOLINGS,
                  test_sets: Sequence[str] = RANK_TEST_SETS, base: str = BASE,
                  n_splits: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Noise ceiling of Kendall's tau between the layer rankings.

    The speakers of a test set are split at random into two halves within
    each class (``draw_speaker_halves``). On each half the curve is the
    seed-mean official EER per layer. In every split:

    - the ceiling is tau between the clean-to-clean curves of half A and of
      half B (the same condition on disjoint speakers);
    - the cross-condition tau is tau between the clean-to-clean curve of
      half A and the ``condition``-to-``condition`` curve of half B.

    Returns (summary, splits). summary, one row per test set and pooling:
    encoder, condition, test_set, pooling, n_layers, n_speakers_bonafide,
    n_speakers_spoof, n_splits, observed_tau (all files, clean against
    ``condition``), ceiling_median, ceiling_low, ceiling_high, cross_median,
    cross_low, cross_high (median and 2.5th and 97.5th percentiles over the
    splits) and share_ceiling_at_or_below_observed. splits, one row per
    split: encoder, condition, test_set, pooling, split, tau_ceiling,
    tau_cross.
    """
    from scipy.stats import kendalltau

    layers = [int(layer) for layer in layers]
    summary, detail = [], []
    for test_set in test_sets:
        entry = resamples.get(test_set)
        labels = entry["labels"]
        for pooling in poolings:
            half_a = split_halves(resamples, test_set, pooling, n_splits)
            half_b = (1 - half_a).astype(np.int16)
            clean = [resamples.matrix(base, base, pooling, layer, test_set) for layer in layers]
            trimmed = [resamples.matrix(condition, condition, pooling, layer, test_set)
                       for layer in layers]
            curve = lambda matrices, counts: np.stack(
                [seed_mean_resampled_eer(m, labels, counts) for m in matrices], axis=1)
            clean_a, clean_b, trimmed_b = curve(clean, half_a), curve(clean, half_b), \
                curve(trimmed, half_b)                                  # (n_splits, n_layers)
            n = half_a.shape[0]
            ceiling = np.array([kendalltau(clean_a[s], clean_b[s])[0] for s in range(n)])
            cross = np.array([kendalltau(clean_a[s], trimmed_b[s])[0] for s in range(n)])
            if not (np.isfinite(ceiling).all() and np.isfinite(cross).all()):
                raise ValueError(f"tau is not defined in a split of {test_set}/{pooling}: a curve "
                                 "has the same EER at every layer")
            observed = float(kendalltau([seed_mean_eer(m, labels) for m in clean],
                                        [seed_mean_eer(m, labels) for m in trimmed])[0])
            fixed = {"encoder": resamples.store.encoder, "condition": condition,
                     "test_set": test_set, "pooling": pooling}
            summary.append({
                **fixed, "n_layers": len(layers),
                "n_speakers_bonafide": int(entry["n_spk"]["n_speakers_bonafide"]),
                "n_speakers_spoof": int(entry["n_spk"]["n_speakers_spoof"]), "n_splits": n,
                "observed_tau": observed,
                "ceiling_median": float(np.median(ceiling)),
                "ceiling_low": float(np.percentile(ceiling, 2.5)),
                "ceiling_high": float(np.percentile(ceiling, 97.5)),
                "cross_median": float(np.median(cross)),
                "cross_low": float(np.percentile(cross, 2.5)),
                "cross_high": float(np.percentile(cross, 97.5)),
                "share_ceiling_at_or_below_observed": float(np.mean(ceiling <= observed + 1e-12))})
            detail.append(pd.DataFrame({**fixed, "split": np.arange(n), "tau_ceiling": ceiling,
                                        "tau_cross": cross}))
    return pd.DataFrame(summary), pd.concat(detail, ignore_index=True)


# Thresholds side by side ------------------------------------------------------------------

def threshold_summary(rank: pd.DataFrame, rank_detail: pd.DataFrame, effects: pd.DataFrame,
                      fallback: Mapping[str, pd.DataFrame], dev_selected: pd.DataFrame,
                      conditions: Sequence[str] = ("trim40", "trim20", "trim15"),
                      layers: Sequence[int] = SUMMARY_LAYERS,
                      test_sets: Sequence[str] = cfg.TEST_SETS,
                      splits: Sequence[str] = cfg.SPLITS) -> pd.DataFrame:
    """The trimming thresholds side by side, one row per condition and pooling.

    ``rank`` and ``rank_detail`` are the tables of ``rank_agreement_ci`` of
    every condition, ``effects`` the table of ``effects.effects_ci``,
    ``fallback`` maps a condition to its ``shards.fallback_counts`` and
    ``dev_selected`` is ``selection.select_on_dev``.

    Columns: condition, top_db, pooling; per test set t: kendall_tau_<t>,
    tau_ci_low_<t>, tau_ci_high_<t>, spearman_rho_<t>, rho_ci_low_<t>,
    rho_ci_high_<t>, best_layer_<t>_clean, p_best_<t>_clean,
    best_layer_<t>_trimmed, p_best_<t>_trimmed; dev_selected_layer; per test
    set and layer k: delta_total_<t>_L<k> and interaction_<t>_L<k> with
    ``_spk_ci_low`` and ``_spk_ci_high``; per split: n_fallback_<split>,
    pct_fallback_<split>. Missing values are NaN.
    """
    assert {"delta_total", "interaction"} <= set(EFFECT_WEIGHTS)
    rows = []
    for condition in conditions:
        r_all = rank[rank["condition"] == condition]
        d_all = rank_detail[rank_detail["condition"] == condition]
        e_all = effects[effects["condition"] == condition]
        for pooling in sorted(set(r_all["pooling"])):
            row = {"condition": condition, "top_db": cfg.trim_top_db(condition), "pooling": pooling}
            for t in test_sets:
                r = r_all[(r_all["test_set"] == t) & (r_all["pooling"] == pooling)].iloc[0]
                d = d_all[(d_all["test_set"] == t) & (d_all["pooling"] == pooling)].set_index("layer")
                for column in ("kendall_tau", "tau_ci_low", "tau_ci_high", "spearman_rho",
                               "rho_ci_low", "rho_ci_high"):
                    row[f"{column}_{t}"] = r[column]
                for name in ("clean", "trimmed"):
                    best = int(r[f"best_layer_{name}"])
                    row[f"best_layer_{t}_{name}"] = best
                    row[f"p_best_{t}_{name}"] = float(d.loc[best, f"p_best_{name}"])
            chosen = dev_selected[(dev_selected["condition"] == condition)
                                  & (dev_selected["pooling"] == pooling)]
            row["dev_selected_layer"] = int(chosen["layer"].iloc[0]) if len(chosen) else np.nan
            for t in test_sets:
                for layer in layers:
                    for effect in ("delta_total", "interaction"):
                        e = e_all[(e_all["test_set"] == t) & (e_all["pooling"] == pooling)
                                  & (e_all["layer"].astype(int) == int(layer))
                                  & (e_all["effect"] == effect)]
                        key = f"{effect}_{t}_L{int(layer)}"
                        row[key] = float(e["estimate"].iloc[0]) if len(e) else np.nan
                        for bound in ("spk_ci_low", "spk_ci_high"):
                            row[f"{key}_{bound}"] = float(e[bound].iloc[0]) if len(e) else np.nan
            counts = fallback.get(condition)
            counts = (counts[counts["label"].astype(str) == "all"].set_index("split")
                      if counts is not None else pd.DataFrame())
            for split in splits:
                found = split in counts.index
                row[f"n_fallback_{split}"] = int(counts.loc[split, "n_fallback"]) if found else np.nan
                row[f"pct_fallback_{split}"] = (float(counts.loc[split, "pct_fallback"])
                                                if found else np.nan)
            rows.append(row)
    return pd.DataFrame(rows)
