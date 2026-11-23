"""Deviation of padded from unpadded features.

Each file is zero-padded at the end to the 4 s window and encoded three ways:
padded without an attention mask, padded with an attention mask, and unpadded
(the reference: the real samples only). In all three the feature is pooled
over the frames of the real samples. For one layer and one sample of files,
the relative deviation is

    mean over files of (mean over dimensions of |padded - reference|)
    -----------------------------------------------------------------
    mean over files of (mean over dimensions of |reference|)

Files with fewer than ``config.MIN_SAMPLES`` real samples are skipped.
"""

from __future__ import annotations

import json
import os
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from . import config as cfg
from . import extraction

STATISTICS = ("mean", "std")
# name -> (encoding compared, encoding it is compared with)
COMPARISONS = {"no_mask": ("no_mask", "reference"), "mask": ("mask", "reference"),
               "mask_vs_no_mask": ("mask", "no_mask")}
PAD_BINS = ((0, 0, "0"), (0, 16000, "<1 s"), (16000, 32000, "1-2 s"),
            (32000, 48000, "2-3 s"), (48000, 64001, ">3 s"))
SAMPLE_SIZE = 500
SAMPLE_SEED = 0
# Generator stream of the file-level bootstrap; the same for every encoder.
INTERVAL_SEED = (cfg.BOOTSTRAP_SEED, 101)
# Prediction (a) of the layer-normalisation check: with an attention mask the
# deviation is at most this factor times the noise floor.
FLOOR_FACTOR = 10.0
# The encoder of the predictions about the noise floor (analysis plan, Section 7a).
PREDICTED_ENCODER = "wavlm-large"
# The encoder whose two samples are the samples of every encoder.
REFERENCE_ENCODER = "wavlm-base"
# The tables and the check tables of gate A4 (the reference encoder against
# the original check). The per-file table is the reference of the samples of
# every other run; none of them is written again once it exists.
GATE_TABLES = ("padding_deviation", "padding_deviation_files", "padding_deviation_by_amount")
GATE_CHECKS = ("padding_encoder", "padding_samples", "table_padding_deviation",
               "table_padding_deviation_by_amount", "padding_printed_numbers")
# The commit at which gate A4 passed on Colab and wrote these tables.
GATE_COMMIT = "637b4b8"
# Files of the timed check before the deviation and the noise floor.
TIMING_FILES = 25
TIMING = "timing"
NOT_REDRAWN = "not redrawn"


# Samples of files ----------------------------------------------------------------

def random_items(items: pd.DataFrame, n: int = SAMPLE_SIZE, seed: int = SAMPLE_SEED) -> pd.DataFrame:
    """``n`` files drawn without replacement with a fixed seed, in the order of the list."""
    if n > len(items):
        raise ValueError(f"cannot sample {n} of {len(items)} items")
    index = np.sort(np.random.default_rng(seed).choice(len(items), size=n, replace=False))
    return items.iloc[index]


def most_padded_items(items: pd.DataFrame, padding: Mapping[str, int],
                      n: int = SAMPLE_SIZE) -> pd.DataFrame:
    """The ``n`` files with the most zero padding.

    ``padding`` maps audio_id to the padding in samples (window minus the
    real samples of the condition). Files without padding are excluded; the
    order is largest padding first, then ascending audio id.
    """
    table = items.assign(pad=items["audio_id"].map(padding))
    if table["pad"].isna().any():
        raise KeyError("some items have no padding value")
    table = table[table["pad"] > 0].sort_values(["pad", "audio_id"], ascending=[False, True])
    return table.head(n)


def padding_from_lengths(lengths: pd.DataFrame, window: int = cfg.WINDOW) -> dict[str, int]:
    """audio_id -> padding in samples from the stored input lengths (``shards.shard_lengths``)."""
    return {str(a): int(window - n) for a, n in zip(lengths["audio_id"], lengths["n_real"])}


def pad_bin(pad_samples: int) -> str:
    """Label of the padding bin of PAD_BINS (samples at 16 kHz)."""
    pad_samples = int(pad_samples)
    if pad_samples <= 0:
        return PAD_BINS[0][2]
    for lo, hi, label in PAD_BINS[1:]:
        if lo <= pad_samples < hi:
            return label
    raise ValueError(f"padding {pad_samples} outside the 4 s window")


# The files of an earlier run -------------------------------------------------------

SAMPLES = ("random", "most_padded")
# The comparisons of the published values: each padded encoding against the reference.
PUBLISHED_COMPARISONS = ("no_mask", "mask")


def sample_ids(per_file: pd.DataFrame, condition: str, sample: str) -> list[str]:
    """Audio ids of one condition and sample of a per-file table, in its order.

    ``condition`` is the label the table uses. Raises if the table has no
    such rows or repeats an id.
    """
    part = per_file[(per_file["condition"].astype(str) == condition)
                    & (per_file["sample"].astype(str) == sample)]
    ids = [str(a) for a in part["audio_id"]]
    if not ids:
        raise KeyError(f"the per-file table has no rows of {condition!r}, {sample!r}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"the per-file table repeats audio ids of {condition!r}, {sample!r}")
    return ids


def items_with_ids(items: pd.DataFrame, ids: Sequence[str]) -> pd.DataFrame:
    """The rows of a file list with the given audio ids, in the order of ``ids``."""
    table = items.assign(audio_id=items["audio_id"].astype(str)).set_index("audio_id", drop=False)
    missing = [a for a in ids if a not in table.index]
    if missing:
        raise KeyError(f"{len(missing)} audio ids are not in the file list (first: {missing[0]})")
    return table.loc[list(ids)].reset_index(drop=True)


def compare_samples(chosen: pd.DataFrame, ids: Sequence[str]) -> dict[str, object]:
    """A sample drawn by this package against the ids of an earlier run:
    n_files, n_reference, n_common, same_order and ok (the same set of files)."""
    mine, reference = [str(a) for a in chosen["audio_id"]], [str(a) for a in ids]
    common = len(set(mine) & set(reference))
    return {"n_files": len(mine), "n_reference": len(reference), "n_common": common,
            "same_order": mine == reference,
            "ok": bool(common == len(mine) == len(reference) and len(mine) > 0)}


def gate_files_check(sha256: Mapping[str, str], commit: str = GATE_COMMIT) -> pd.DataFrame:
    """The tables of gate A4 against their meta files.

    ``sha256`` maps the path of each table to its sha256 now. A table is ok
    if its ``<name>.meta.json`` records a commit that starts with ``commit``
    and is not "-dirty", and records the sha256 the table has now.

    Returns file, git_commit, recorded_sha256, sha256, ok ("missing" for a
    value that the meta file does not have, or without a meta file).
    """
    rows = []
    for path, digest in sha256.items():
        meta_path = os.path.splitext(path)[0] + ".meta.json"
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as fh:
                meta = json.load(fh)
        recorded = str(meta.get("git_commit", "missing"))
        recorded_sha256 = str(meta.get("sha256", "missing"))
        rows.append({"file": path, "git_commit": recorded, "recorded_sha256": recorded_sha256,
                     "sha256": digest,
                     "ok": bool(recorded.startswith(commit) and not recorded.endswith("-dirty")
                                and recorded_sha256 == digest)})
    return pd.DataFrame(rows, columns=["file", "git_commit", "recorded_sha256", "sha256", "ok"])


# The samples of the reference encoder, for another encoder --------------------------------------

def stored_lengths(lengths: pd.DataFrame, window: int = cfg.WINDOW) -> dict[str, int]:
    """audio_id -> real samples inside the window, from the stored input
    lengths of an unpadded extraction (``shards.shard_lengths``)."""
    return {str(a): int(min(int(n), window)) for a, n in zip(lengths["audio_id"], lengths["n_real"])}


def lengths_from_audio(items: pd.DataFrame, condition: str) -> dict[str, int]:
    """audio_id -> real samples inside the window, from the audio itself:
    each file of ``items`` (columns path, audio_id) is loaded and prepared as
    in the padded extraction of ``condition``. No encoder is used."""
    top_db = cfg.trim_top_db(condition)
    return {str(item["audio_id"]): int(extraction.prepare_file(item["path"], top_db, "padded")["n_real"])
            for _, item in items.iterrows()}


def check_reference_sample(reference_files: pd.DataFrame, condition: str, sample: str,
                           lengths: Mapping[str, int], lengths_from: str,
                           drawn: pd.DataFrame | None = None) -> dict[str, object]:
    """One sample of the reference encoder against what another encoder has.

    ``reference_files`` is the per-file table of the reference encoder
    (``padding_deviation``: audio_id, condition, sample, n_real).
    ``lengths`` maps audio_id to the real samples inside the window as this
    encoder has them (its own shards, or the audio); a file is equal if its
    value is the ``n_real`` of the reference table. ``drawn`` is the sample
    drawn again for this encoder (None: it cannot be drawn); it must be the
    files of the reference table in the same order.

    Returns condition, sample, n_files, lengths_from, n_lengths_equal,
    redrawn ("same files in the same order", "other files" or NOT_REDRAWN)
    and ok.
    """
    ids = sample_ids(reference_files, condition, sample)
    part = reference_files[(reference_files["condition"].astype(str) == condition)
                           & (reference_files["sample"].astype(str) == sample)]
    expected = {str(a): int(n) for a, n in zip(part["audio_id"], part["n_real"])}
    equal = sum(1 for a in ids if a in lengths and int(lengths[a]) == expected[a])
    if drawn is None:
        redrawn, same = NOT_REDRAWN, True
    else:
        same = [str(a) for a in drawn["audio_id"]] == ids
        redrawn = "same files in the same order" if same else "other files"
    return {"condition": condition, "sample": sample, "n_files": len(ids),
            "lengths_from": lengths_from, "n_lengths_equal": int(equal), "redrawn": redrawn,
            "ok": bool(equal == len(ids) and same)}


def published_rows(table: pd.DataFrame) -> pd.DataFrame:
    """Rows of a deviation table for PUBLISHED_COMPARISONS and, where the
    table has the column ``statistic``, for the mean."""
    keep = table["comparison"].isin(PUBLISHED_COMPARISONS)
    if "statistic" in table.columns:
        keep &= table["statistic"] == "mean"
    return table[keep].reset_index(drop=True)


# Deviation -----------------------------------------------------------------------

def _encode_three_ways(model, x: torch.Tensor, n_real: int, n_layers: int, device):
    no_mask = extraction.encode_padded(model, x, n_real, n_layers, device)
    mask = extraction.encode_padded(model, x, n_real, n_layers, device, use_attention_mask=True)
    reference = extraction.encode_unpadded(model, x[:n_real], n_layers, device)
    if reference["n_frames"] != no_mask["n_real_frames"]:
        raise RuntimeError(f"the reference has {reference['n_frames']} frames but the padded "
                           f"input has {no_mask['n_real_frames']} real frames")
    pick = lambda encoded: {"mean": encoded["real_frames_mean"], "std": encoded["real_frames_std"]}
    return {"no_mask": pick(no_mask), "mask": pick(mask),
            "reference": {"mean": reference["mean"], "std": reference["std"]}}, no_mask


def _layer_table(sums, maxima, reference_sum, reference_max, n_files: int, n_skipped: int,
                 fixed: dict, comparisons: Sequence[str]) -> pd.DataFrame:
    rows = []
    n = max(n_files, 1)
    for comparison in comparisons:
        for statistic in STATISTICS:
            total, largest = sums[(comparison, statistic)], maxima[(comparison, statistic)]
            for layer in range(total.shape[0]):
                mean_value = reference_sum[statistic][layer] / n
                rows.append({**fixed, "comparison": comparison, "layer": layer + 1,
                             "statistic": statistic,
                             "max_abs_diff": largest[layer],
                             "mean_abs_diff": total[layer] / n,
                             "mean_abs_value": mean_value,
                             "rel_mean_diff": total[layer] / n / max(mean_value, 1e-12),
                             "rel_max_diff": largest[layer] / max(mean_value, 1e-12),
                             "max_abs_value": reference_max[statistic][layer],
                             "n_files": n_files, "n_skipped": n_skipped})
    return pd.DataFrame(rows)


def _accumulate(encodings: dict, pairs: Mapping[str, tuple[str, str]], n_layers: int, row: dict,
                sums, maxima, reference_sum, reference_max, reference_name: str) -> None:
    for statistic in STATISTICS:
        reference = encodings[reference_name][statistic]
        scale = reference.abs().mean(1).numpy()
        reference_sum[statistic] += scale
        reference_max[statistic] = np.maximum(reference_max[statistic],
                                              reference.abs().amax(1).numpy())
        for layer in range(n_layers):
            row[f"reference_{statistic}_L{layer + 1}"] = float(scale[layer])
        for comparison, (a, b) in pairs.items():
            diff = (encodings[a][statistic] - encodings[b][statistic]).abs()  # (L, D)
            key = (comparison, statistic)
            maxima[key] = np.maximum(maxima[key], diff.amax(1).numpy())
            per_layer = diff.mean(1).numpy()
            sums[key] += per_layer
            for layer in range(n_layers):
                row[f"{comparison}_{statistic}_L{layer + 1}"] = float(per_layer[layer])
            row[f"{comparison}_{statistic}_max"] = diff.max().item()


def padding_deviation(model, items: pd.DataFrame, device: str | torch.device, condition: str,
                      sample: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deviation of padded features (without and with an attention mask) from unpadded ones.

    ``items`` has the columns path and audio_id; ``sample`` labels the rows.

    Returns (per_layer, per_file). per_layer has one row per comparison x
    statistic x layer: condition, sample, comparison, layer, statistic,
    max_abs_diff, mean_abs_diff, mean_abs_value (of the reference),
    rel_mean_diff, rel_max_diff, max_abs_value, n_files, n_skipped. per_file
    has one row per file: audio_id, condition, sample, n_real, n_real_frames,
    pad, pad_s, pad_bin, ``reference_<statistic>_L<l>`` and, per comparison,
    ``<comparison>_<statistic>_L<l>`` and ``<comparison>_<statistic>_max``.
    """
    top_db = cfg.trim_top_db(condition)
    n_layers = int(model.config.num_hidden_layers)
    zeros = lambda: np.zeros(n_layers)
    sums = {(c, s): zeros() for c in COMPARISONS for s in STATISTICS}
    maxima = {(c, s): zeros() for c in COMPARISONS for s in STATISTICS}
    reference_sum = {s: zeros() for s in STATISTICS}
    reference_max = {s: zeros() for s in STATISTICS}
    per_file, n_files, n_skipped = [], 0, 0
    for _, item in items.iterrows():
        prepared = extraction.prepare_file(item["path"], top_db, "padded")
        x, n_real = prepared["x"], prepared["n_real"]
        if n_real < cfg.MIN_SAMPLES:
            n_skipped += 1
            continue
        encodings, no_mask = _encode_three_ways(model, x, n_real, n_layers, device)
        pad = cfg.WINDOW - n_real
        row = {"audio_id": item["audio_id"], "condition": condition, "sample": sample,
               "n_real": n_real, "n_real_frames": no_mask["n_real_frames"], "pad": pad,
               "pad_s": pad / cfg.SAMPLE_RATE, "pad_bin": pad_bin(pad)}
        _accumulate(encodings, COMPARISONS, n_layers, row, sums, maxima, reference_sum,
                    reference_max, "reference")
        per_file.append(row)
        n_files += 1
    per_layer = _layer_table(sums, maxima, reference_sum, reference_max, n_files, n_skipped,
                             {"condition": condition, "sample": sample}, list(COMPARISONS))
    return per_layer, pd.DataFrame(per_file)


def noise_floor(reference_model, reference_device, other_model, other_device,
                items: pd.DataFrame, condition: str, sample: str
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deviation between two unpadded extractions of the same files.

    The files are encoded unpadded by ``reference_model`` on
    ``reference_device`` and by ``other_model`` on ``other_device`` (the same
    weights on a GPU and on a CPU); the deviation of the second from the
    first uses the definition of ``padding_deviation``.

    Returns (per_layer, per_file) in the layout of ``padding_deviation`` with
    the single comparison "floor".
    """
    top_db = cfg.trim_top_db(condition)
    n_layers = int(reference_model.config.num_hidden_layers)
    pairs = {"floor": ("other", "reference")}
    zeros = lambda: np.zeros(n_layers)
    sums = {("floor", s): zeros() for s in STATISTICS}
    maxima = {("floor", s): zeros() for s in STATISTICS}
    reference_sum = {s: zeros() for s in STATISTICS}
    reference_max = {s: zeros() for s in STATISTICS}
    per_file, n_files, n_skipped = [], 0, 0
    for _, item in items.iterrows():
        prepared = extraction.prepare_file(item["path"], top_db, "unpadded")
        if prepared["n_real"] < cfg.MIN_SAMPLES:
            n_skipped += 1
            continue
        reference = extraction.encode_unpadded(reference_model, prepared["x"], n_layers,
                                               reference_device)
        other = extraction.encode_unpadded(other_model, prepared["x"], n_layers, other_device)
        row = {"audio_id": item["audio_id"], "condition": condition, "sample": sample,
               "n_real": prepared["n_real"]}
        _accumulate({"reference": reference, "other": other}, pairs, n_layers, row, sums, maxima,
                    reference_sum, reference_max, "reference")
        per_file.append(row)
        n_files += 1
    per_layer = _layer_table(sums, maxima, reference_sum, reference_max, n_files, n_skipped,
                             {"condition": condition, "sample": sample}, ["floor"])
    return per_layer, pd.DataFrame(per_file)


def deviation_by_padding(per_file: pd.DataFrame, per_layer: pd.DataFrame,
                         layers: Sequence[int] = (3, 4, 5)) -> pd.DataFrame:
    """Relative deviation by amount of zero padding.

    For every sample x condition x comparison x padding bin: n_files and, per
    statistic, the mean over the files of the bin of (file mean |difference|
    / mean |reference| of the layer) for ``layers``
    (``<statistic>_rel_L<l>``) and averaged over all layers
    (``<statistic>_rel_all``). The layer scales are those of ``per_layer``.
    """
    keys = ["condition", "sample", "statistic", "layer"]
    scale = per_layer.drop_duplicates(keys).set_index(keys)["mean_abs_value"].sort_index()
    n_layers = int(per_layer["layer"].max())
    labels = [b[2] for b in PAD_BINS]
    rows = []
    for (sample, condition), group in per_file.groupby(["sample", "condition"], sort=False):
        for comparison in COMPARISONS:
            for label in labels:
                part = group[group["pad_bin"] == label]
                if part.empty:
                    continue
                row = {"sample": sample, "condition": condition, "comparison": comparison,
                       "pad_bin": label, "n_files": len(part)}
                for statistic in STATISTICS:
                    relative = []
                    for layer in range(1, n_layers + 1):
                        s = scale.loc[(condition, sample, statistic, layer)]
                        r = part[f"{comparison}_{statistic}_L{layer}"].to_numpy() / max(s, 1e-12)
                        relative.append(r.mean())
                        if layer in layers:
                            row[f"{statistic}_rel_L{layer}"] = float(r.mean())
                    row[f"{statistic}_rel_all"] = float(np.mean(relative))
                rows.append(row)
    return pd.DataFrame(rows)


# Intervals -----------------------------------------------------------------------

def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return numerator / np.maximum(denominator, 1e-12)


def deviation_intervals(per_file: pd.DataFrame, comparisons: Sequence[str] = ("no_mask", "mask"),
                        difference: tuple[str, str] | None = ("mask", "no_mask"),
                        n_boot: int = cfg.N_BOOT, seed=INTERVAL_SEED) -> pd.DataFrame:
    """File-level bootstrap intervals of the relative deviation.

    For every condition x sample of ``per_file``, the files are resampled
    with replacement ``n_boot`` times and the ratio (mean numerator over mean
    denominator) is recomputed in each resample; percentile 2.5 and 97.5. The
    same resamples are used for every layer, statistic and comparison, so
    ``difference`` (first minus second comparison) is paired.

    Returns one row per condition x sample x statistic x layer x quantity:
    condition, sample, statistic, layer, quantity (a comparison, or
    ``<a>_minus_<b>``), estimate, ci_low, ci_high, n_files, n_boot.
    """
    rows = []
    for (condition, sample), group in per_file.groupby(["condition", "sample"], sort=False):
        n = len(group)
        index = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
        layers = sorted(int(c.rsplit("_L", 1)[1]) for c in group.columns
                        if c.startswith("reference_mean_L"))
        for statistic in STATISTICS:
            for layer in layers:
                denominator = group[f"reference_{statistic}_L{layer}"].to_numpy(dtype=float)
                boot_denominator = denominator[index].mean(axis=1)
                estimates, boots = {}, {}
                for comparison in comparisons:
                    numerator = group[f"{comparison}_{statistic}_L{layer}"].to_numpy(dtype=float)
                    estimates[comparison] = float(_ratio(numerator.mean(), denominator.mean()))
                    boots[comparison] = _ratio(numerator[index].mean(axis=1), boot_denominator)
                if difference is not None:
                    a, b = difference
                    name = f"{a}_minus_{b}"
                    estimates[name] = estimates[a] - estimates[b]
                    boots[name] = boots[a] - boots[b]
                for quantity, boot in boots.items():
                    rows.append({"condition": condition, "sample": sample, "statistic": statistic,
                                 "layer": layer, "quantity": quantity,
                                 "estimate": estimates[quantity],
                                 "ci_low": float(np.percentile(boot, 2.5)),
                                 "ci_high": float(np.percentile(boot, 97.5)),
                                 "n_files": n, "n_boot": n_boot})
    return pd.DataFrame(rows)


def mask_reduces_deviation(intervals: pd.DataFrame, statistic: str = "mean",
                           sample: str = "random") -> pd.DataFrame:
    """Per condition and layer, whether the attention mask reduces the deviation:
    the paired difference (with mask minus without mask) is negative and its
    interval excludes zero, on ``sample``."""
    part = intervals[(intervals["quantity"] == "mask_minus_no_mask")
                     & (intervals["statistic"] == statistic) & (intervals["sample"] == sample)]
    out = part[["condition", "sample", "layer", "estimate", "ci_low", "ci_high"]].copy()
    out["mask_reduces"] = (out["estimate"] < 0) & (out["ci_high"] < 0)
    return out.reset_index(drop=True)


def floor_predictions(intervals: pd.DataFrame, floor: pd.DataFrame, statistic: str = "mean",
                      sample: str = "random", factor: float = FLOOR_FACTOR) -> pd.DataFrame:
    """The two predictions about the noise floor, per condition and layer.

    (a) with an attention mask the deviation is at the floor: the upper bound
        of its interval is at most ``factor`` times the upper bound of the
        floor interval, computed in float64 exactly as
        ``upper <= 10.0 * floor_upper`` with no margin (with a floor upper
        bound of 0, the deviation must be 0);
    (b) without a mask the deviation stays above the floor: the lower bound
        of its interval is above the upper bound of the floor interval.

    ``intervals`` and ``floor`` are tables of ``deviation_intervals`` (the
    floor with the quantity "floor").

    Returns condition, sample, layer, mask_estimate, mask_ci_high,
    no_mask_ci_low, floor_ci_high, a_holds, b_holds.
    """
    keys = ["condition", "layer"]

    def pick(table, quantity):
        part = table[(table["quantity"] == quantity) & (table["statistic"] == statistic)
                     & (table["sample"] == sample)]
        return part.set_index(keys)

    mask, no_mask, base = pick(intervals, "mask"), pick(intervals, "no_mask"), pick(floor, "floor")
    if not (len(mask) == len(no_mask) == len(base) and mask.index.equals(base.index)
            and no_mask.index.equals(base.index)) or mask.empty:
        raise KeyError("the deviation and floor tables do not cover the same conditions and layers")
    out = pd.DataFrame({"mask_estimate": mask["estimate"], "mask_ci_high": mask["ci_high"],
                        "no_mask_ci_low": no_mask["ci_low"], "floor_ci_high": base["ci_high"]})
    at_floor = (out["mask_ci_high"].astype(np.float64)
                <= float(factor) * out["floor_ci_high"].astype(np.float64))
    zero_floor = out["floor_ci_high"] == 0
    out["a_holds"] = np.where(zero_floor, (out["mask_estimate"] == 0) & (out["mask_ci_high"] == 0),
                              at_floor)
    out["b_holds"] = out["no_mask_ci_low"] > out["floor_ci_high"]
    out = out.reset_index()
    out.insert(1, "sample", sample)
    return out


def prediction_verdict(predictions: pd.DataFrame, n_layers: int,
                       conditions: Sequence[str]) -> pd.DataFrame:
    """Whether each prediction about the noise floor holds: it is met at
    every layer in every condition.

    ``predictions`` is the table of ``floor_predictions``. It must have
    exactly one row per condition of ``conditions`` and layer 1..``n_layers``;
    anything else raises, so a missing layer cannot count as met.

    Returns one row per prediction: prediction ("a", "b"), statement,
    n_cells (conditions x layers), n_met, holds.
    """
    expected = {(c, layer) for c in conditions for layer in range(1, n_layers + 1)}
    found = list(zip(predictions["condition"], predictions["layer"].astype(int)))
    if len(found) != len(expected) or set(found) != expected:
        raise ValueError(f"the predictions have {len(found)} rows; expected one for each of "
                         f"{len(conditions)} conditions x {n_layers} layers")
    statements = {
        "a": f"with an attention mask: upper bound <= {FLOOR_FACTOR} x upper bound of the floor",
        "b": "without an attention mask: lower bound > upper bound of the floor"}
    rows = []
    for name, statement in statements.items():
        met = predictions[f"{name}_holds"].astype(bool)
        rows.append({"prediction": name, "statement": statement, "n_cells": len(expected),
                     "n_met": int(met.sum()), "holds": bool(met.all())})
    return pd.DataFrame(rows)
