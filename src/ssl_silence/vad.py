"""The supplementary condition ``vad``.

Not in the analysis plan and not pre-registered: WavLM Base only, a
sensitivity analysis of the trimming rule. The leading and trailing
non-speech found by the WebRTC voice activity detector is removed
(``trimming.vad_interval``; the rule is in the experiment configuration),
and the audio is then extracted, trained and analysed like a trimmed
condition, for the layers of the configuration only.

Runs. Model weights are not stored, so the models trained on untrimmed audio
are trained again to be scored on the new condition. Their scores on
untrimmed audio are compared with those of the existing runs
(``clean_scores_check``), and the cell trained and tested on untrimmed audio
is read from the existing runs (``bootstrap.ScoreStore`` with
``base_runs_dir``). The runs of this condition have their own folder
(``config.vad_runs_dir``).

Tables (``config.vad_tables_dir``): the six effects with their cells, the
effects side by side with those of the condition trimmed at 40 dB, the
audio removed per split and class, the fallback counts and the per-attack
effect.
"""

from __future__ import annotations

import os
import time
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from . import config as cfg
from . import encoders, verify
from .extraction import encode_unpadded, prepare_file, sample_positions, trim_rule
from .training import PlannedRun, load_scores, plan_runs, read_runs

CONDITION = "vad"
ENCODER = "wavlm-base"
EXTRACTION = "unpadded"
BASE = "clean"
COMPARED_WITH = "trim40"          # the condition of the analysis shown next to this one
SECTION = "supplementary"


# Runs ----------------------------------------------------------------------------------------

def planned_runs(seeds: Sequence[int] = cfg.SEEDS) -> list[PlannedRun]:
    """The runs of the supplementary condition: the models trained on
    untrimmed audio first, then those trained on the new condition."""
    planned = plan_runs(ENCODER, EXTRACTION, seeds, section=SECTION)
    conditions = {p.spec.condition for p in planned}
    if conditions != {BASE, CONDITION}:
        raise ValueError(f"the supplementary runs train on {sorted(conditions)}, "
                         f"expected {BASE} and {CONDITION}")
    return sorted(planned, key=lambda p: p.spec.condition != BASE)


def runs_of(planned: Sequence[PlannedRun], condition: str) -> list[PlannedRun]:
    """The planned runs trained in ``condition``."""
    return [p for p in planned if p.spec.condition == condition]


def layers(planned: Sequence[PlannedRun] | None = None) -> tuple[int, ...]:
    """The layers of the supplementary runs; every layer has both training
    conditions and both poolings."""
    planned = planned_runs(cfg.SEEDS[:1]) if planned is None else planned
    found = sorted({p.spec.layer for p in planned})
    for layer in found:
        have = {(p.spec.condition, p.spec.pooling) for p in planned if p.spec.layer == layer}
        if have != {(c, pooling) for c in (BASE, CONDITION) for pooling in cfg.POOLINGS}:
            raise ValueError(f"layer {layer} does not have both training conditions and poolings")
    return tuple(found)


def clean_scores_check(planned: Sequence[PlannedRun], runs_dir: str, base_runs_dir: str,
                       tolerance: float = verify.SCORE_TOLERANCE) -> pd.DataFrame:
    """The models trained again on untrimmed audio against the existing runs.

    For every planned run trained in the base condition, every score set in
    the base condition (the selection set and the two test sets) is compared
    with the same set of the run with the same id under ``base_runs_dir``.
    One row per run and set: n, ids_equal, labels_equal, max_abs_diff,
    exactly_zero, within_tolerance (``verify.compare_score_sets``),
    best_epoch of both runs and best_epoch_equal. ``ok`` needs the scores
    within ``tolerance`` and the same selected epoch; a missing run or set is
    a failing row.
    """
    existing = read_runs(base_runs_dir).set_index("run_id")
    have_new = os.path.exists(os.path.join(runs_dir, "runs.csv"))
    new_runs = read_runs(runs_dir).set_index("run_id") if have_new else None
    rows = []
    for run in runs_of(planned, BASE):
        run_id = run.spec.run_id
        names = [name for name in (run.spec.selection_set,) + run.score_set_names
                 if name.partition("@")[2] == BASE]
        new_path = os.path.join(runs_dir, "scores", f"{run_id}.npz")
        old_path = os.path.join(base_runs_dir, "scores", f"{run_id}.npz")
        present = (new_runs is not None and run_id in new_runs.index and run_id in existing.index
                   and os.path.exists(new_path) and os.path.exists(old_path))
        new_sets = load_scores(new_path)[0] if present else {}
        old_sets = load_scores(old_path)[0] if present else {}
        for name in names:
            row = {"run_id": run_id, "score_set": name, "n": 0, "ids_equal": False,
                   "labels_equal": False, "max_abs_diff": float("nan"), "exactly_zero": False,
                   "within_tolerance": False, "best_epoch": -1, "existing_best_epoch": -1,
                   "best_epoch_equal": False}
            if present and name in new_sets and name in old_sets:
                row.update(verify.compare_score_sets(new_sets[name], old_sets[name], tolerance))
                row["best_epoch"] = int(new_runs.loc[run_id, "best_epoch"])
                row["existing_best_epoch"] = int(existing.loc[run_id, "best_epoch"])
                row["best_epoch_equal"] = row["best_epoch"] == row["existing_best_epoch"]
            row["ok"] = bool(row["within_tolerance"] and row["best_epoch_equal"])
            rows.append(row)
    return pd.DataFrame(rows)


# Extraction ----------------------------------------------------------------------------------

def short_check(items: pd.DataFrame, split: str, *, model, device: str | torch.device,
                n: int = cfg.SHORT_CHECK_FILES, seed=cfg.REEXTRACTION_SEED,
                condition: str = CONDITION) -> pd.DataFrame:
    """Encode ``n`` files of a split in the new condition and check what the
    extraction relies on.

    The files are those of ``extraction.sample_positions`` and go through
    ``prepare_file`` and ``encode_unpadded``, one at a time. One row per
    file: the samples before trimming, the kept interval, the fallback flag,
    the number of frames against the output length of the convolutional
    encoder, the shape and finiteness of the pooled features, and whether the
    interval and the number of real samples are those of the rule
    (``interval_problems``). ``seconds`` is the time of loading, trimming and
    encoding the file; one file is encoded before the timing starts.
    """
    top_db, rule = trim_rule(condition)
    if rule is None:
        raise ValueError(f"{condition!r} is not trimmed by voice activity detection")
    n_layers = int(model.config.num_hidden_layers)
    shape = (n_layers, int(model.config.hidden_size))
    kernels, strides = encoders.conv_layers(model)
    positions = sample_positions(len(items), min(n, len(items)), seed)
    cuda = torch.device(device).type == "cuda"
    encode_unpadded(model, prepare_file(items.iloc[positions[0]]["path"], top_db, EXTRACTION,
                                        rule)["x"], n_layers, device)
    rows = []
    for position in positions:
        item = items.iloc[position]
        start = time.perf_counter()
        prepared = prepare_file(item["path"], top_db, EXTRACTION, rule)
        encoded = encode_unpadded(model, prepared["x"], n_layers, device)
        if cuda:
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        row = {"split": split, "condition": condition, "position": int(position),
               "audio_id": str(item["audio_id"]), "label": int(item["label"]),
               "n_samples": prepared["n_samples"], "kept_start": prepared["kept_start"],
               "kept_end": prepared["kept_end"], "trim_fallback": prepared["trim_fallback"],
               "n_real": prepared["n_real"], "too_short": prepared["too_short"],
               "n_input": int(len(prepared["x"])), "n_frames": int(encoded["n_frames"]),
               "conv_frames": int(encoders.conv_output_length(len(prepared["x"]), kernels, strides)),
               "shape_ok": tuple(encoded["mean"].shape) == shape == tuple(encoded["std"].shape),
               "finite": bool(torch.isfinite(encoded["mean"]).all()
                              and torch.isfinite(encoded["std"]).all()),
               "seconds": seconds}
        row["interval_ok"] = not interval_problems(pd.DataFrame([row]), rule).any(axis=None)
        row["ok"] = bool(row["n_frames"] == row["conv_frames"] and row["shape_ok"]
                         and row["finite"] and row["interval_ok"])
        rows.append(row)
    return pd.DataFrame(rows)


def interval_problems(kept: pd.DataFrame, rule: Mapping[str, int]) -> pd.DataFrame:
    """What the rule excludes, per file (one boolean column per problem).

    ``kept`` has n_samples, kept_start, kept_end, n_real and trim_fallback.
    outside: the interval is not inside the file; off_frame: start or end is
    not at a frame boundary of the detector (a file without fallback);
    short_without_fallback: fewer than ``config.MIN_TRIMMED_SAMPLES`` kept
    without the flag; fallback_not_whole: flagged, but the interval is not
    the whole file; n_real_differs: the number of real samples is not the
    kept length inside the window.
    """
    n, start, end = (kept[key].to_numpy().astype(np.int64)
                     for key in ("n_samples", "kept_start", "kept_end"))
    fallback = kept["trim_fallback"].to_numpy().astype(bool)
    frame = cfg.SAMPLE_RATE * int(rule["frame_ms"]) // 1000
    return pd.DataFrame({
        "outside": (start < 0) | (end < start) | (end > n),
        "off_frame": ~fallback & ((start % frame != 0) | (end % frame != 0)),
        "short_without_fallback": ~fallback & (end - start < cfg.MIN_TRIMMED_SAMPLES),
        "fallback_not_whole": fallback & ((start != 0) | (end != n)),
        "n_real_differs": kept["n_real"].to_numpy().astype(np.int64)
        != np.minimum(end - start, cfg.WINDOW)}, index=kept.index)


def shards_check(kept: pd.DataFrame, reference: pd.DataFrame,
                 rule: Mapping[str, int] | None = None,
                 splits: Sequence[str] = cfg.SPLITS) -> pd.DataFrame:
    """The files of the extracted shards against the shards of the untrimmed
    condition and against the rule.

    ``kept`` is ``shards.kept_intervals`` of the new shards, ``reference``
    ``shards.shard_lengths`` of the untrimmed condition. One row per split:
    the files on both sides, whether the ids are the same in the same order
    and the labels equal, the number of files with each problem of
    ``interval_problems``, the files with more real samples than the
    untrimmed file has, and the fallback files.
    """
    rule = cfg.vad_rule(CONDITION) if rule is None else rule
    rows = []
    for split in splits:
        a = kept[kept["split"] == split].reset_index(drop=True)
        b = reference[reference["split"] == split].reset_index(drop=True)
        same_ids = len(a) > 0 and a["audio_id"].astype(str).tolist() == b["audio_id"].astype(str).tolist()
        same_labels = bool(same_ids and np.array_equal(a["label"].to_numpy().astype(int),
                                                       b["label"].to_numpy().astype(int)))
        row = {"split": split, "n_files": len(a), "n_reference": len(b),
               "same_ids_in_order": bool(same_ids), "same_labels": same_labels}
        problems = interval_problems(a, rule)
        row.update({f"n_{name}": int(problems[name].sum()) for name in problems.columns})
        row["n_longer_than_untrimmed"] = int(
            (a["n_real"].to_numpy() > b["n_real"].to_numpy()).sum()) if same_ids else -1
        row["n_fallback"] = int(a["trim_fallback"].sum())
        row["ok"] = bool(same_ids and same_labels and not problems.any(axis=None)
                         and row["n_longer_than_untrimmed"] == 0)
        rows.append(row)
    return pd.DataFrame(rows)


# Tables --------------------------------------------------------------------------------------

def removed_per_file(kept: pd.DataFrame, sr: int = cfg.SAMPLE_RATE) -> pd.DataFrame:
    """Seconds removed from every file, from ``shards.kept_intervals``:
    orig_s, removed_lead_s, removed_trail_s, removed_total_s and kept_s, with
    split, audio_id, system_id, label and fallback. A fallback file has 0
    removed."""
    n, start, end = (kept[key].to_numpy().astype(np.int64)
                     for key in ("n_samples", "kept_start", "kept_end"))
    return pd.DataFrame({
        "split": kept["split"].to_numpy(), "audio_id": kept["audio_id"].astype(str).to_numpy(),
        "system_id": kept["system_id"].astype(str).to_numpy(),
        "label": kept["label"].to_numpy().astype(int),
        "fallback": kept["trim_fallback"].to_numpy().astype(bool),
        "orig_s": n / sr, "removed_lead_s": start / sr, "removed_trail_s": (n - end) / sr,
        "removed_total_s": (n - end + start) / sr, "kept_s": (end - start) / sr})


def summarise_removed(per_file: pd.DataFrame) -> pd.DataFrame:
    """Removed seconds per split and label from ``removed_per_file``.

    One row per split (in order of appearance) and label (ascending), with
    the columns of ``silence.summarise_removed`` for a threshold: n,
    n_fallback and, for the total, the leading and the trailing removed
    seconds, the median, the quartiles and the mean; removed_frac_median is
    the median of removed_total_s / orig_s (0 for an empty file). Fallback
    files are included with 0 removed.
    """
    rows = []
    for split in per_file["split"].unique():
        part = per_file[per_file["split"] == split]
        for label in sorted(part["label"].unique()):
            g = part[part["label"] == label]
            row = {"condition": CONDITION, "split": split, "label": int(label), "n": len(g),
                   "n_fallback": int(g["fallback"].sum())}
            for m in ("total", "lead", "trail"):
                s = g[f"removed_{m}_s"]
                row[f"removed_{m}_median_s"] = float(s.median())
                row[f"removed_{m}_q25_s"] = float(s.quantile(0.25))
                row[f"removed_{m}_q75_s"] = float(s.quantile(0.75))
                row[f"removed_{m}_mean_s"] = float(s.mean())
            frac = (g["removed_total_s"] / g["orig_s"].where(g["orig_s"] > 0)).fillna(0.0)
            row["removed_frac_median"] = float(frac.median())
            rows.append(row)
    return pd.DataFrame(rows)


CELL_COLUMNS = ["eer_cc", "eer_ct", "eer_tc", "eer_tt"]


def cells_from_effects(effects: pd.DataFrame) -> pd.DataFrame:
    """The four cells per test set, pooling and layer, as the effects table
    holds them (seed-mean EER in percent): cc, ct, tc and tt are named
    train then test, with c the untrimmed condition and t the condition of
    the row. Raises if the rows of one configuration hold different cells."""
    keys = ["encoder", "condition", "test_set", "pooling", "layer"]
    if (effects.groupby(keys)[CELL_COLUMNS].nunique() > 1).any(axis=None):
        raise ValueError("the rows of one configuration hold different cells")
    return effects[keys + CELL_COLUMNS + ["n_files"]].drop_duplicates(keys).reset_index(drop=True)


SIDE_KEYS = ["test_set", "pooling", "layer", "effect"]
SIDE_COLUMNS = ["estimate", "file_ci_low", "file_ci_high", "file_excludes_zero", "spk_ci_low",
                "spk_ci_high", "spk_excludes_zero", "eer_cc", "eer_ct", "eer_tc", "eer_tt"]


def side_by_side(effects: pd.DataFrame, reference: pd.DataFrame,
                 reference_condition: str = COMPARED_WITH) -> pd.DataFrame:
    """The effects of the new condition next to those of ``reference_condition``.

    ``effects`` is the effects table of the new condition and ``reference``
    the effects table of the analysis (``effects_2x2``). One row per test
    set, pooling, layer and effect of the new condition; the columns of
    ``SIDE_COLUMNS`` twice, with the two condition names in front, and
    ``same_sign`` and ``both_spk_exclude_zero``. A row that the reference
    does not have raises. ``eer_cc`` is the same cell in both; it is kept
    twice so that a difference would be seen.
    """
    new = effects[effects["condition"] == CONDITION]
    old = reference[reference["condition"] == reference_condition]
    if new.empty:
        raise ValueError(f"the effects table has no row of {CONDITION}")
    merged = new[["encoder"] + SIDE_KEYS + SIDE_COLUMNS].merge(
        old[SIDE_KEYS + SIDE_COLUMNS], on=SIDE_KEYS, how="left",
        suffixes=(f"_{CONDITION}", f"_{reference_condition}"), validate="one_to_one")
    missing = merged[f"estimate_{reference_condition}"].isna()
    if missing.any():
        raise KeyError(f"{int(missing.sum())} rows have no row of {reference_condition}; first: "
                       f"{merged.loc[missing, SIDE_KEYS].head(3).to_dict('records')}")
    a, b = merged[f"estimate_{CONDITION}"], merged[f"estimate_{reference_condition}"]
    merged["same_sign"] = np.sign(a) == np.sign(b)
    merged["both_spk_exclude_zero"] = (
        merged[f"spk_excludes_zero_{CONDITION}"].astype(bool)
        & merged[f"spk_excludes_zero_{reference_condition}"].astype(bool))
    return merged
