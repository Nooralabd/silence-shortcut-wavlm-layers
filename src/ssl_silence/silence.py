"""Leading and trailing silence: measurements per file, summaries and baselines.

Silence is defined as in ``trimming``: a frame (400 samples, hop 160) is
silent when its RMS lies more than ``top_db`` dB below the loudest frame of
the same file. Only leading and trailing silence is measured. All lengths
are in samples at 16 kHz unless a name ends in ``_s`` (seconds).

Every file is loaded once and measured three ways:

    untrimmed audio   silence in the file and inside the 4 s window, at
                      SILENCE_TOP_DBS
    residual          the same measurements on the audio after trimming at
                      PRE_TRIM_TOP_DB (with the fallback rule), at
                      RESIDUAL_TOP_DBS
    removed amount    what the trimming rule removes at REMOVED_TOP_DBS

``measure_items`` writes one wide table per split; ``silence_table``,
``residual_table`` and ``removed_table`` return the three parts.
"""

from __future__ import annotations

import os
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from . import config as cfg
from . import metrics as M
from .trimming import (REMOVED_PER_DB_KEYS, apply_trim, bounds_from_db, frame_rms,
                       removed_amounts_of, resolve_trim, rms_to_db)

SILENCE_TOP_DBS = (15, 20, 30, 40, 50)
PRE_TRIM_TOP_DB = 40
RESIDUAL_TOP_DBS = (15, 20, 30, 40)
REMOVED_TOP_DBS = (40, 20, 15)
# librosa floors the frame RMS at this value before taking dB.
RMS_FLOOR = 1e-5
ITEM_COLUMNS = ["dataset", "split", "audio_id", "label", "system_id", "speaker"]
RESIDUAL_PREFIX = "residual_"
AMOUNT_PREFIX = "amount_"


# One waveform ------------------------------------------------------------------------

def window_view(n_samples: int, start: int, end: int, window: int = cfg.WINDOW) -> dict[str, int]:
    """What a centre crop to ``window`` samples, or zero padding at the end, contains.

    [start, end) is the non-silent interval of a file of ``n_samples``.
    Returns win_lead and win_trail (silence inside the window; padding not
    included), pad (zeros appended to a shorter file) and crop (samples
    discarded from a longer file, both sides together).
    """
    if n_samples > window:
        lo = (n_samples - window) // 2
        hi = lo + window
        win_lead = min(max(start - lo, 0), window)
        win_trail = min(max(hi - end, 0), window)
        pad, crop = 0, n_samples - window
    else:
        win_lead, win_trail = start, n_samples - end
        pad, crop = window - n_samples, 0
    return {"win_lead": int(win_lead), "win_trail": int(win_trail),
            "pad": int(pad), "crop": int(crop)}


def measure_silence(y: np.ndarray, top_dbs: Sequence[int] = SILENCE_TOP_DBS,
                    window: int = cfg.WINDOW) -> dict[str, int | float]:
    """Silence measurements of one mono 16 kHz waveform y (N,).

    Returns a flat dict: n_samples, all_zero (0/1), pad, crop and, for every
    threshold d: lead_d and trail_d (silence detected in the file),
    win_lead_d and win_trail_d (the same inside the window), trim_fallback_d
    (0/1: trimming at d keeps the file untrimmed), trimmed_len_d (length
    after trimming at d) and pad_trimmed_d (zeros a padded window would add
    after trimming at d); then peak_frame_rms, the RMS of the loudest frame.
    lead and trail report what was detected, whether or not the fallback
    applies.
    """
    y = np.asarray(y, dtype=np.float32)
    n = int(y.shape[0])
    all_zero = n == 0 or not np.any(y)
    rms = frame_rms(y) if n else np.zeros(0)
    db = rms_to_db(rms) if n else np.zeros(0)
    whole = window_view(n, 0, n, window)
    out: dict[str, int | float] = {"n_samples": n, "all_zero": int(all_zero),
                                   "pad": whole["pad"], "crop": whole["crop"]}
    for d in top_dbs:
        start, end = bounds_from_db(db, d, n)
        view = window_view(n, start, end, window)
        _, _, fallback = resolve_trim(n, start, end, all_zero)
        trimmed_len = n if fallback else end - start
        out[f"lead_{d}"] = start
        out[f"trail_{d}"] = n - end
        out[f"win_lead_{d}"] = view["win_lead"]
        out[f"win_trail_{d}"] = view["win_trail"]
        out[f"trim_fallback_{d}"] = int(fallback)
        out[f"trimmed_len_{d}"] = int(trimmed_len)
        out[f"pad_trimmed_{d}"] = int(max(0, window - trimmed_len))
    out["peak_frame_rms"] = float(rms.max()) if rms.size else 0.0
    return out


def measure_waveform(mono: torch.Tensor, top_dbs: Sequence[int] = SILENCE_TOP_DBS,
                     residual_top_dbs: Sequence[int] = RESIDUAL_TOP_DBS,
                     pre_trim_top_db: float = PRE_TRIM_TOP_DB,
                     removed_top_dbs: Sequence[float] = REMOVED_TOP_DBS) -> dict[str, int | float]:
    """All measurements of one mono 16 kHz waveform (1, N), in one flat dict.

    The keys of ``measure_silence`` for the untrimmed audio; with the prefix
    ``residual_``: fallback_trim<dB> (0/1, the pre-trim kept the file
    untrimmed), n_samples_orig and the keys of ``measure_silence`` for the
    audio trimmed at ``pre_trim_top_db``; with the prefix ``amount_``: the
    keys of ``trimming.removed_amounts_of``.
    """
    out = dict(measure_silence(mono[0].numpy(), top_dbs))
    trimmed, fallback = apply_trim(mono, pre_trim_top_db)
    residual = {f"fallback_trim{pre_trim_top_db:g}": int(fallback),
                "n_samples_orig": int(mono.shape[1])}
    residual.update(measure_silence(trimmed[0].numpy(), residual_top_dbs))
    out.update({RESIDUAL_PREFIX + k: v for k, v in residual.items()})
    out.update({AMOUNT_PREFIX + k: v
                for k, v in removed_amounts_of(mono, removed_top_dbs).items()})
    return out


def measure_file(path: str, **kwargs) -> dict[str, int | float]:
    """Load a file as the extraction does (mono, 16 kHz) and measure it:
    sr_orig, n_channels and the keys of ``measure_waveform``."""
    from .audio import load_audio, to_mono_16k

    wav, sr, _ = load_audio(path)
    out: dict[str, int | float] = {"sr_orig": int(sr), "n_channels": int(wav.shape[0])}
    out.update(measure_waveform(to_mono_16k(wav, sr), **kwargs))
    return out


# Whole splits, resumable ----------------------------------------------------------------

def read_measurements(path: str) -> pd.DataFrame:
    """Read a measurement table; ids, system ids and speakers stay strings and
    every number is read back exactly as it was written."""
    return pd.read_csv(path, dtype={"audio_id": str, "system_id": str, "speaker": str},
                       keep_default_na=False, float_precision="round_trip")


def measure_items(file_list: pd.DataFrame, out_dir: str, name: str, chunk_size: int = 2000,
                  progress: bool = True,
                  measure: Callable[[str], Mapping[str, int | float]] | None = None
                  ) -> pd.DataFrame:
    """Measure every file of ``file_list`` in resumable chunks.

    ``file_list`` has ITEM_COLUMNS and ``path`` (``datasets.items``). Chunk
    files ``<name>_chunk_<k>.csv`` are written under ``<out_dir>/chunks``; a
    finished chunk is not measured again. The whole table is written to
    ``<out_dir>/<name>.csv`` and returned: ITEM_COLUMNS and the measurement
    columns, in file-list order. Raises if the chunks do not match the file
    list.
    """
    measure = measure_file if measure is None else measure
    chunk_dir = os.path.join(out_dir, "chunks")
    cfg.makedirs(chunk_dir)
    n = len(file_list)
    n_chunks = (n + chunk_size - 1) // chunk_size
    parts = []
    for k in range(n_chunks):
        path = os.path.join(chunk_dir, f"{name}_chunk_{k:04d}.csv")
        if not os.path.exists(path):
            rows = []
            for _, item in file_list.iloc[k * chunk_size:(k + 1) * chunk_size].iterrows():
                row = {c: item[c] for c in ITEM_COLUMNS}
                row.update(measure(item["path"]))
                rows.append(row)
            tmp = path + ".tmp"
            pd.DataFrame(rows).to_csv(tmp, index=False)
            os.replace(tmp, path)
            if progress:
                print(f"[{name}] chunk {k + 1}/{n_chunks} measured")
        parts.append(read_measurements(path))
    full = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=ITEM_COLUMNS)
    if len(full) != n or list(full["audio_id"]) != [str(a) for a in file_list["audio_id"]]:
        raise RuntimeError(f"[{name}] the chunks do not match the file list; delete {chunk_dir} "
                           "and measure again (file order or chunk size changed?)")
    tmp = os.path.join(out_dir, f"{name}.csv.tmp")
    full.to_csv(tmp, index=False)
    os.replace(tmp, os.path.join(out_dir, f"{name}.csv"))
    return full


def _part(wide: pd.DataFrame, prefix: str | None, shared: Sequence[str] = ()) -> pd.DataFrame:
    keep = [c for c in ITEM_COLUMNS if c in wide.columns] + [c for c in shared if c in wide.columns]
    prefixes = (RESIDUAL_PREFIX, AMOUNT_PREFIX)
    if prefix is None:
        columns = {c: c for c in wide.columns if c not in keep and not c.startswith(prefixes)}
    else:
        columns = {c: c[len(prefix):] for c in wide.columns if c.startswith(prefix)}
        if not columns:
            raise KeyError(f"the table has no columns with the prefix {prefix!r}")
    return wide[keep + list(columns)].rename(columns=columns)


def silence_table(wide: pd.DataFrame) -> pd.DataFrame:
    """Measurements of the untrimmed audio from a table of ``measure_items``."""
    return _part(wide, None)


def residual_table(wide: pd.DataFrame) -> pd.DataFrame:
    """Measurements of the audio trimmed at PRE_TRIM_TOP_DB: sr_orig, n_channels,
    fallback_trim<dB>, n_samples_orig and the columns of ``measure_silence``."""
    return _part(wide, RESIDUAL_PREFIX, shared=("sr_orig", "n_channels"))


def removed_table(wide: pd.DataFrame) -> pd.DataFrame:
    """Removed amounts, one row per file: n_samples_orig, orig_s and the keys of
    ``trimming.REMOVED_PER_DB_KEYS`` with the suffix ``_<d>``."""
    return _part(wide, AMOUNT_PREFIX)


# Summaries -----------------------------------------------------------------------------

def silence_columns(top_db: int, with_duration: bool = False) -> list[str]:
    """The silence measurements at one threshold: lead, trail (whole file) and
    win_lead, win_trail (inside the window); with ``with_duration`` also
    n_samples and pad."""
    columns = [f"lead_{top_db}", f"trail_{top_db}", f"win_lead_{top_db}", f"win_trail_{top_db}"]
    return columns + ["n_samples", "pad"] if with_duration else columns


def summarise(table: pd.DataFrame, top_db: int) -> pd.DataFrame:
    """Per-class summary in seconds of ``silence_columns(top_db, with_duration=True)``.

    One row per label and measure: label, measure, n, mean_s, median_s,
    q25_s, q75_s, max_s.
    """
    rows = []
    for label, group in table.groupby("label"):
        for column in silence_columns(top_db, with_duration=True):
            s = group[column] / cfg.SAMPLE_RATE
            rows.append({"label": int(label), "measure": column, "n": len(s),
                         "mean_s": s.mean(), "median_s": s.median(),
                         "q25_s": s.quantile(0.25), "q75_s": s.quantile(0.75),
                         "max_s": s.max()})
    return pd.DataFrame(rows)


def residual_silence_share(tables: Mapping[str, pd.DataFrame],
                           pre_trim_top_db: float = PRE_TRIM_TOP_DB,
                           threshold_s: float = 0.05, top_db: float | None = None) -> pd.DataFrame:
    """Files in which silence remains after trimming at ``pre_trim_top_db``.

    ``tables`` maps a name to a ``residual_table``. A file counts when it was
    trimmed (no fallback) and more than ``threshold_s`` of leading (trailing)
    silence remains; silence is defined by ``top_db`` (default: the trimming
    threshold itself). The percentages are of all files.

    Columns: table, top_db, n_files, n_fallback, n_lead_over, pct_lead_over,
    n_trail_over, pct_trail_over.
    """
    pre = f"{pre_trim_top_db:g}"
    measured = pre if top_db is None else f"{top_db:g}"
    limit = threshold_s * cfg.SAMPLE_RATE
    rows = []
    for name, table in tables.items():
        trimmed = table[table[f"fallback_trim{pre}"] != 1]
        row = {"table": name, "top_db": float(measured), "n_files": len(table),
               "n_fallback": len(table) - len(trimmed)}
        for side in ("lead", "trail"):
            count = int((trimmed[f"{side}_{measured}"] > limit).sum())
            row[f"n_{side}_over"] = count
            row[f"pct_{side}_over"] = 100 * count / len(table) if len(table) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def untrimmable_below(top_db: float) -> float:
    """Loudest-frame RMS below which no frame of a file can be silent at ``top_db``:
    librosa floors the frame RMS at RMS_FLOOR, so nothing is trimmed."""
    return RMS_FLOOR * 10 ** (top_db / 20)


def quiet_file_counts(tables: Mapping[str, pd.DataFrame],
                      top_dbs: Sequence[float] = REMOVED_TOP_DBS) -> pd.DataFrame:
    """Files too quiet to be trimmed, per table and label ("0", "1", "all").

    ``tables`` maps a name to a ``silence_table``. Columns: table, label,
    n_files, n_all_zero and, for every threshold d, n_below_<d>: files whose
    peak_frame_rms is below ``untrimmable_below(d)`` (all-zero files included).
    """
    rows = []
    for name, table in tables.items():
        y = table["label"].to_numpy()
        peak = table["peak_frame_rms"].to_numpy(dtype=np.float64)
        zero = table["all_zero"].to_numpy() == 1
        for label, mask in (("0", y == 0), ("1", y == 1), ("all", np.ones(y.size, bool))):
            row = {"table": name, "label": label, "n_files": int(mask.sum()),
                   "n_all_zero": int(zero[mask].sum())}
            for d in top_dbs:
                row[f"n_below_{d:g}"] = int((peak[mask] < untrimmable_below(d)).sum())
            rows.append(row)
    return pd.DataFrame(rows)


def duration_map(table: pd.DataFrame) -> dict[str, float]:
    """audio id -> original duration in seconds: samples at 16 kHz / 16,000.

    ``table`` is a ``silence_table``. Raises if a file was not recorded at
    16 kHz (its length at 16 kHz would then be a resampled length).
    """
    other = table[table["sr_orig"] != cfg.SAMPLE_RATE]
    if len(other):
        raise ValueError(f"{len(other)} files are not at {cfg.SAMPLE_RATE} Hz "
                         f"(first: {other['audio_id'].iloc[0]}, {other['sr_orig'].iloc[0]} Hz)")
    return dict(zip(table["audio_id"].astype(str), table["n_samples"] / cfg.SAMPLE_RATE))


def check_duration_subsets(durations: Mapping[str, float],
                           expected: tuple[int, int] = cfg.ITW_DURATION_COUNTS,
                           window_s: float = 4.0) -> tuple[int, int]:
    """(files of at most ``window_s``, longer files); raises unless they equal ``expected``."""
    values = np.array(list(durations.values()), dtype=np.float64)
    found = (int((values <= window_s).sum()), int((values > window_s).sum()))
    if found != tuple(expected):
        raise ValueError(f"duration subsets {found}, expected {tuple(expected)}")
    return found


# Removed amounts ---------------------------------------------------------------------------

FALLBACK_COUNT_COLUMNS = ("n_files", "n_fallback", "n_too_short")


def removed_long(removed: pd.DataFrame, top_dbs: Sequence[float] = REMOVED_TOP_DBS
                 ) -> pd.DataFrame:
    """``removed_table`` -> one row per file and threshold.

    Columns: the item columns, top_db, n_samples_orig, orig_s and the keys of
    ``trimming.REMOVED_PER_DB_KEYS``. Thresholds in the order given; files in
    table order within a threshold.
    """
    keep = [c for c in ITEM_COLUMNS if c in removed.columns]
    parts = []
    for d in top_dbs:
        columns = {f"{k}_{d:g}": k for k in REMOVED_PER_DB_KEYS}
        missing = [c for c in columns if c not in removed.columns]
        if missing:
            raise KeyError(f"threshold {d:g} dB is not in the table: {missing}")
        part = removed[keep + ["n_samples_orig", "orig_s", *columns]].rename(columns=columns)
        part.insert(len(keep), "top_db", d)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def summarise_removed(long: pd.DataFrame) -> pd.DataFrame:
    """Removed seconds per threshold, split and label from ``removed_long`` rows.

    One row per (top_db, dataset, split, label) in order of appearance,
    labels ascending: n, n_fallback and, for the total, the leading and the
    trailing removed seconds, ``removed_<m>_median_s``, ``_q25_s``,
    ``_q75_s`` and ``_mean_s``; removed_frac_median is the median of
    removed_total_s / orig_s (0 for an empty file). Fallback files are
    included with 0 removed.
    """
    groups = [k for k in ("dataset", "split") if k in long.columns]
    rows = []
    for d in long["top_db"].unique():
        at_d = long[long["top_db"] == d]
        combos = ([tuple(t) for t in at_d[groups].drop_duplicates().itertuples(index=False)]
                  if groups else [()])
        for combo in combos:
            part = at_d
            for key, value in zip(groups, combo):
                part = part[part[key] == value]
            for label in sorted(part["label"].unique()):
                g = part[part["label"] == label]
                row = {"top_db": d, **dict(zip(groups, combo)), "label": label,
                       "n": len(g), "n_fallback": int(g["fallback"].sum())}
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


def fallback_counts_from_removed(long: pd.DataFrame) -> pd.DataFrame:
    """Fallback and too-short files per threshold, split and label ("0", "1",
    "all") from ``removed_long`` rows: top_db, split, label, n_files,
    n_fallback, n_too_short, pct_fallback (the layout of
    ``shards.fallback_counts`` with top_db in front)."""
    rows = []
    for d in long["top_db"].unique():
        for split in long.loc[long["top_db"] == d, "split"].unique():
            g = long[(long["top_db"] == d) & (long["split"] == split)]
            y = g["label"].to_numpy()
            fallback, short = g["fallback"].to_numpy(), g["too_short"].to_numpy()
            for label, m in (("0", y == 0), ("1", y == 1), ("all", np.ones(y.size, bool))):
                rows.append({"top_db": d, "split": split, "label": label,
                             "n_files": int(m.sum()), "n_fallback": int(fallback[m].sum()),
                             "n_too_short": int(short[m].sum()),
                             "pct_fallback": (float(100 * fallback[m].mean()) if m.any()
                                              else float("nan"))})
    return pd.DataFrame(rows)


def compare_fallback_counts(measured: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """The measured fallback counts of one threshold against the counts of the shards.

    ``measured``: ``fallback_counts_from_removed`` rows of exactly one
    threshold. ``reference``: ``shards.fallback_counts`` of the condition
    with that threshold. Returns the reference rows with every count of both
    tables side by side (``<column>`` from the shards, ``<column>_measured``)
    and ``ok`` (the three counts are equal). A reference row without a
    measured row raises.
    """
    columns = list(FALLBACK_COUNT_COLUMNS)
    if measured["top_db"].nunique() != 1:
        raise ValueError("`measured` must hold the rows of exactly one threshold")
    m = measured.assign(label=measured["label"].astype(str)).set_index(["split", "label"])
    ref = reference.assign(label=reference["label"].astype(str))
    lead = [c for c in ("condition", "split", "label") if c in ref.columns]
    out = ref[lead + columns].copy()
    for c in columns:
        values = []
        for split, label in zip(ref["split"], ref["label"]):
            if (split, label) not in m.index:
                raise KeyError(f"no measured row for split={split!r}, label={label!r}")
            values.append(int(m.at[(split, label), c]))
        out[f"{c}_measured"] = values
    out["ok"] = np.logical_and.reduce([out[c].to_numpy() == out[f"{c}_measured"].to_numpy()
                                       for c in columns])
    return out.reset_index(drop=True)


def trim_length_diffs(long: pd.DataFrame, lengths: pd.DataFrame, top_db: float,
                      window: int = cfg.WINDOW) -> pd.DataFrame:
    """The measured length after trimming against the length stored in the shards.

    The shards hold the length after trimming and after the centre crop
    (``shards.shard_lengths``: n_real), so the comparison is
    min(n_samples_trimmed, window) == n_real. ``long`` holds ``removed_long``
    rows; the rows of ``top_db`` are used.

    Returns one row per file (shard order within a split): split, audio_id,
    label, n_samples_trimmed, n_expected, n_real, abs_diff_samples. Raises
    KeyError when the files of a split differ or an audio id is repeated.
    """
    mine = long[long["top_db"] == top_db]
    if mine.empty:
        raise KeyError(f"no rows at {top_db:g} dB in the long table")
    parts = []
    for split in lengths["split"].unique():
        a = lengths[lengths["split"] == split][["split", "audio_id", "label", "n_real"]].copy()
        b = mine[mine["split"] == split][["audio_id", "n_samples_trimmed"]].copy()
        a["audio_id"], b["audio_id"] = a["audio_id"].astype(str), b["audio_id"].astype(str)
        if a["audio_id"].duplicated().any() or b["audio_id"].duplicated().any():
            raise KeyError(f"split {split!r}: duplicate audio_id")
        only_shards = set(a["audio_id"]) - set(b["audio_id"])
        only_measured = set(b["audio_id"]) - set(a["audio_id"])
        if only_shards or only_measured:
            raise KeyError(f"split {split!r}: {len(only_shards)} files only in the shards "
                           f"(e.g. {sorted(only_shards)[:3]}), {len(only_measured)} only in the "
                           f"measurement (e.g. {sorted(only_measured)[:3]})")
        m = a.merge(b, on="audio_id", how="left", validate="one_to_one")
        m["n_expected"] = np.minimum(m["n_samples_trimmed"].to_numpy(), window)
        m["abs_diff_samples"] = np.abs(m["n_expected"].to_numpy() - m["n_real"].to_numpy())
        parts.append(m[["split", "audio_id", "label", "n_samples_trimmed", "n_expected",
                        "n_real", "abs_diff_samples"]])
    return pd.concat(parts, ignore_index=True)


def summarise_trim_lengths(diffs: pd.DataFrame) -> pd.DataFrame:
    """``trim_length_diffs`` rows -> one row per split: split, n_files, n_equal,
    max_abs_diff_samples, ok (every file equal)."""
    rows = []
    for split in diffs["split"].unique():
        g = diffs[diffs["split"] == split]
        n_equal = int((g["abs_diff_samples"] == 0).sum())
        rows.append({"split": split, "n_files": int(len(g)), "n_equal": n_equal,
                     "max_abs_diff_samples": int(g["abs_diff_samples"].max()),
                     "ok": n_equal == len(g)})
    return pd.DataFrame(rows)


# Baselines without an encoder -------------------------------------------------------------------

SILENCE_ONLY = "silence-only"
FEATURE_SETS = {"sil": False, "sil+dur": True}      # name -> with n_samples and pad
BASE_CONDITION = "clean"


def mannwhitney_auc(bonafide: np.ndarray, spoof: np.ndarray) -> dict[str, float]:
    """Two-sided Mann-Whitney U test, bonafide against spoof, with the AUC.

    Returns auc (U of the spoof group / (n0 * n1): the probability that a
    spoof value exceeds a bonafide value, ties counted one half), u, p
    (two-sided, scipy default, tie-corrected), n_bonafide and n_spoof.
    """
    from scipy.stats import mannwhitneyu

    bonafide = np.asarray(bonafide, dtype=np.float64)
    spoof = np.asarray(spoof, dtype=np.float64)
    n0, n1 = bonafide.size, spoof.size
    if n0 == 0 or n1 == 0:
        raise ValueError("both groups need at least one value")
    result = mannwhitneyu(spoof, bonafide, alternative="two-sided")
    return {"auc": float(result.statistic) / (n0 * n1), "u": float(result.statistic),
            "p": float(result.pvalue), "n_bonafide": n0, "n_spoof": n1}


def baseline_score_path(scores_dir: str, features: str, top_db: float, condition: str) -> str:
    """Score file of one silence-only classifier."""
    name = f"{SILENCE_ONLY}_{features.replace('+', '-')}_{top_db:g}db_{condition}.npz"
    return os.path.join(scores_dir, name)


def silence_only_baseline(tables: Mapping[str, pd.DataFrame], top_db: int,
                          features: str = "sil", condition: str = BASE_CONDITION,
                          scores_dir: str | None = None, overwrite: bool = False) -> str:
    """Logistic regression on silence measurements only; saves its scores.

    ``tables`` maps a split to its measurement table (``silence_table`` for
    the untrimmed audio, ``residual_table`` for the audio of the condition
    trim40) and must hold "train"; the classifier is trained on it and scores
    every other split. ``features`` is "sil" (``silence_columns(top_db)``) or
    "sil+dur" (with n_samples and pad). ``condition`` names the audio that
    was measured; the audio itself is not used.

    Model: log(1 + samples) -> StandardScaler -> LogisticRegression(
    class_weight="balanced", max_iter=1000), deterministic.

    The scores (P(spoof)) are saved as the score sets ``<split>@<condition>``
    in ``baseline_score_path``; an existing file raises unless ``overwrite``.
    Returns the path.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler

    from .provenance import versions
    from .training import ScoreSet, save_scores, set_name

    if "train" not in tables:
        raise KeyError('the tables must hold the split "train"')
    cfg.trim_top_db(condition)
    columns = silence_columns(top_db, FEATURE_SETS[features])
    path = baseline_score_path(scores_dir or os.path.join(cfg.baselines_dir(), "scores"),
                               features, top_db, condition)
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite=True to replace it")
    train = tables["train"]
    model = make_pipeline(FunctionTransformer(np.log1p), StandardScaler(),
                          LogisticRegression(class_weight="balanced", max_iter=1000))
    model.fit(train[columns].to_numpy(dtype=np.float64), train["label"].to_numpy(dtype=np.int64))
    sets = {}
    for split, table in tables.items():
        if split == "train":
            continue
        p = model.predict_proba(table[columns].to_numpy(dtype=np.float64))[:, 1]
        sets[set_name(split, condition)] = ScoreSet(p.astype(np.float32),
                                                    table["label"].to_numpy(dtype=np.int64),
                                                    table["audio_id"].astype(str).to_numpy())
    save_scores(path, sets, {"model": SILENCE_ONLY, "features": features, "columns": columns,
                             "top_db": top_db, "condition": condition, "n_train": len(train),
                             "versions": versions()})
    return path


def silence_only_baselines(tables: Mapping[str, pd.DataFrame], top_dbs: Sequence[int],
                           condition: str = BASE_CONDITION, scores_dir: str | None = None,
                           overwrite: bool = False) -> pd.DataFrame:
    """``silence_only_baseline`` for every threshold and both feature sets.

    Returns one row per feature set and threshold: condition, features,
    top_db and one column per scored split with the official EER (%) computed
    from the saved score file.
    """
    from .training import load_scores, set_name

    rows = []
    for features in FEATURE_SETS:
        for d in sorted(top_dbs):
            path = silence_only_baseline(tables, d, features, condition, scores_dir, overwrite)
            sets, _ = load_scores(path)
            row = {"condition": condition, "features": features, "top_db": d}
            for split in tables:
                if split != "train":
                    s = sets[set_name(split, condition)]
                    row[split] = M.eer(s.labels, s.scores)
            rows.append(row)
    return pd.DataFrame(rows)


def pad_column(condition: str) -> str:
    """Column of a ``silence_table`` with the zeros a padded window would add in ``condition``."""
    top_db = cfg.trim_top_db(condition)
    return "pad" if top_db is None else f"pad_trimmed_{top_db:g}"


def padding_by_label(tables: Mapping[str, pd.DataFrame],
                     conditions: Sequence[str] = (BASE_CONDITION, "trim40"),
                     per_attack_table: str = "eval") -> pd.DataFrame:
    """Median and quartiles of the padding in seconds, by label, per table and
    condition, and per attack of ``per_attack_table``.

    Columns: table, condition, group ("all" or an attack), label, n,
    median_s, q25_s, q75_s, iqr_s, frac_padded (share of files with any
    padding), frac_over_2s.
    """
    rows = []
    for name, table in tables.items():
        for condition in conditions:
            pad = table[pad_column(condition)].astype(float) / cfg.SAMPLE_RATE
            groups: list[tuple[str, pd.Series]] = [("all", pd.Series(True, index=table.index))]
            if name == per_attack_table and "system_id" in table.columns:
                attacks = sorted(a for a in table["system_id"].astype(str).unique() if a != "-")
                groups += [(a, table["system_id"].astype(str) == a) for a in attacks]
            for group, selected in groups:
                for label in sorted(table.loc[selected, "label"].unique()):
                    s = pad[selected & (table["label"] == label)]
                    if s.empty:
                        continue
                    q25, q75 = s.quantile(0.25), s.quantile(0.75)
                    rows.append({"table": name, "condition": condition, "group": group,
                                 "label": int(label), "n": int(len(s)),
                                 "median_s": float(s.median()), "q25_s": float(q25),
                                 "q75_s": float(q75), "iqr_s": float(q75 - q25),
                                 "frac_padded": float((s > 0).mean()),
                                 "frac_over_2s": float((s > 2.0).mean())})
    return pd.DataFrame(rows)


def padding_only_baseline(tables: Mapping[str, pd.DataFrame],
                          conditions: Sequence[str] = (BASE_CONDITION, "trim40")) -> pd.DataFrame:
    """The amount of padding alone as a detector score, per table and condition.

    Nothing is trained: the score is +padding (more padding = more
    spoof-like) or -padding, each table scored on its own. eer_best is the
    lower of the two official EERs and sign_best names it; auc and
    p_mannwhitney come from ``mannwhitney_auc``.

    Columns: table, condition, n_bonafide, n_spoof, eer_plus_pad,
    eer_minus_pad, eer_best, sign_best, auc, p_mannwhitney.
    """
    rows = []
    for name, table in tables.items():
        for condition in conditions:
            pad = (table[pad_column(condition)].astype(float) / cfg.SAMPLE_RATE).to_numpy()
            y = table["label"].to_numpy(dtype=np.int64)
            eer_plus, eer_minus = M.eer(y, pad), M.eer(y, -pad)
            r = mannwhitney_auc(pad[y == 0], pad[y == 1])
            rows.append({"table": name, "condition": condition, "n_bonafide": r["n_bonafide"],
                         "n_spoof": r["n_spoof"], "eer_plus_pad": eer_plus,
                         "eer_minus_pad": eer_minus, "eer_best": min(eer_plus, eer_minus),
                         "sign_best": "+pad" if eer_plus <= eer_minus else "-pad",
                         "auc": r["auc"], "p_mannwhitney": r["p"]})
    return pd.DataFrame(rows)


# Gate A5: the measured tables against the original ones ---------------------------------------------

# Thresholds of the silence-only classifiers of the original tables, per condition.
BASELINE_TOP_DBS = {BASE_CONDITION: (30, 40, 50), "trim40": (15, 20, 30)}
# Folder of the original per-file tables (name in the mapping file) -> the part of the wide table.
ORIGINAL_PARTS = {"measurements_dir": silence_table, "residual_measurements_dir": residual_table,
                  "removed_measurements_dir": removed_table}
SUMMARY_TABLES = ("silence_summary", "residual_silence_summary", "trim_amount",
                  "padding_only_baseline", "padding_by_label")
# Tables of the silence-only classifiers (names of the mapping file) -> the measured audio.
BASELINE_TABLES = {"silence_only_baseline": BASE_CONDITION, "residual_silence_baseline": "trim40"}
# The seven original tables that gate A5 reproduces.
GATE_TABLES = SUMMARY_TABLES + tuple(BASELINE_TABLES)
# The check tables that decide gate A5, in the order in which they are required.
GATE_CHECKS = ("silence_per_file", *(f"table_{name}" for name in GATE_TABLES),
               "silence_fallback_counts", "silence_trim_lengths", "silence_itw_durations",
               "silence_printed_numbers", "silence_quiet_files")
REMAINING_THRESHOLD_S = 0.05
# Files per split of the timed check before the measurement.
TIMING_FILES = 100
# Files with more than 50 ms of leading silence after trimming at 40 dB in the
# original feasibility table (``strict_trim_feasibility.csv``, row
# ``lead40_gt50ms``). Reported next to the measured counts; not a check.
EARLIER_COUNTS = {"train": 1_046, "dev": 972, "eval": 2_382, "itw": 938}


def remaining_silence(tables: Mapping[str, pd.DataFrame],
                      pre_trim_top_db: float = PRE_TRIM_TOP_DB,
                      threshold_s: float = REMAINING_THRESHOLD_S) -> pd.DataFrame:
    """Leading silence that remains after trimming at ``pre_trim_top_db``, per table and label.

    ``tables`` maps a name to a ``residual_table``. Silence is defined by the
    trimming threshold itself. A file counts when more than ``threshold_s``
    of leading silence remains: in the whole file (``lead``) or inside the
    window (``win_lead``). Every file is counted, also a file that the
    fallback rule kept untrimmed; ``n_lead_over_trimmed`` counts the files
    without fallback only (the count of ``residual_silence_share``). The
    percentages are of all files of the row.

    One row per table and label ("0", "1", "all"): table, label, n_files,
    n_fallback, n_lead_over, pct_lead_over, n_win_lead_over,
    pct_win_lead_over, n_lead_over_trimmed, pct_lead_over_trimmed and
    lead_median_s, lead_q25_s, lead_q75_s: the remaining leading silence of
    the whole file, in seconds, over the files of n_lead_over (NaN if none).
    """
    d = f"{pre_trim_top_db:g}"
    limit = threshold_s * cfg.SAMPLE_RATE
    rows = []
    for name, table in tables.items():
        y = table["label"].to_numpy()
        fallback = table[f"fallback_trim{d}"].to_numpy() == 1
        lead = table[f"lead_{d}"].to_numpy(dtype=np.float64)
        win_lead = table[f"win_lead_{d}"].to_numpy(dtype=np.float64)
        for label, mask in (("0", y == 0), ("1", y == 1), ("all", np.ones(y.size, bool))):
            n = int(mask.sum())
            over = mask & (lead > limit)
            counts = {"n_lead_over": int(over.sum()),
                      "n_win_lead_over": int((mask & (win_lead > limit)).sum()),
                      "n_lead_over_trimmed": int((over & ~fallback).sum())}
            row = {"table": name, "label": label, "n_files": n,
                   "n_fallback": int((mask & fallback).sum())}
            for key, count in counts.items():
                row[key] = count
                row["pct" + key[1:]] = 100 * count / n if n else float("nan")
            seconds = pd.Series(lead[over] / cfg.SAMPLE_RATE)
            row["lead_median_s"] = float(seconds.median())
            row["lead_q25_s"] = float(seconds.quantile(0.25))
            row["lead_q75_s"] = float(seconds.quantile(0.75))
            rows.append(row)
    return pd.DataFrame(rows)


def quiet_files(tables: Mapping[str, pd.DataFrame],
                top_db: float = PRE_TRIM_TOP_DB) -> pd.DataFrame:
    """The files that are too quiet to be trimmed at ``top_db``, one row per file.

    ``tables`` maps a name to a ``silence_table``. A file is listed when its
    peak_frame_rms is below ``untrimmable_below(top_db)`` (the files that
    ``quiet_file_counts`` counts). Columns: table, audio_id, label,
    n_samples, all_zero, peak_frame_rms, top_db, below.
    """
    limit = untrimmable_below(top_db)
    parts = []
    for name, table in tables.items():
        part = table.loc[table["peak_frame_rms"].astype(float) < limit,
                         ["audio_id", "label", "n_samples", "all_zero", "peak_frame_rms"]].copy()
        part.insert(0, "table", name)
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    out["top_db"], out["below"] = float(top_db), limit
    return out


def compare_with_original_files(wide: pd.DataFrame, split: str, original_dir: str) -> pd.DataFrame:
    """The three parts of a wide table against the original per-file tables of ``split``.

    ``original_dir`` holds the folders of ORIGINAL_PARTS with one
    ``<split>.csv`` each (the migrated copies). Every column of an original
    table is compared file by file on ``audio_id`` (``verify.compare_tables``)
    with ``verify.SECONDS_TOLERANCE``: a count must be equal, and seconds may
    differ by the last digit that the original code lost when it read its
    tables with the default parser. Columns that only this package measures
    are not compared. The original tables are read with every number exactly
    as written.

    Returns the rows of ``verify.compare_tables`` with the leading columns
    ``part`` and ``split``.
    """
    from . import verify

    results = []
    for folder, part in ORIGINAL_PARTS.items():
        old = read_measurements(os.path.join(original_dir, folder, f"{split}.csv"))
        result = verify.compare_tables(part(wide), old, ["audio_id"],
                                       tolerance=verify.SECONDS_TOLERANCE)
        result.insert(0, "split", split)
        result.insert(0, "part", folder)
        results.append(result)
    return pd.concat(results, ignore_index=True)


def summary_tables(wide: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """The tables of SUMMARY_TABLES from the wide tables of the splits.

    ``wide`` maps a split to its table of ``measure_items``. The tables have
    the names and columns that the mapping file gives the original tables:
    ``silence_summary`` (``summarise`` at PRE_TRIM_TOP_DB, per split, with the
    column ``table``), ``residual_silence_summary`` (``summarise`` of the
    trimmed audio at RESIDUAL_TOP_DBS, with ``table`` and ``top_db``),
    ``trim_amount`` (``summarise_removed``), ``padding_only_baseline`` and
    ``padding_by_label``.
    """
    raw = {split: silence_table(table) for split, table in wide.items()}
    residual = {split: residual_table(table) for split, table in wide.items()}
    removed = pd.concat([removed_table(table) for table in wide.values()], ignore_index=True)
    return {
        "silence_summary": pd.concat(
            [summarise(table, PRE_TRIM_TOP_DB).assign(table=split) for split, table in raw.items()],
            ignore_index=True),
        "residual_silence_summary": pd.concat(
            [summarise(table, d).assign(table=split, top_db=d)
             for split, table in residual.items() for d in RESIDUAL_TOP_DBS], ignore_index=True),
        "trim_amount": summarise_removed(removed_long(removed)),
        "padding_only_baseline": padding_only_baseline(raw),
        "padding_by_label": padding_by_label(raw),
    }


def earlier_counts_report(share: pd.DataFrame,
                          earlier: Mapping[str, int] = EARLIER_COUNTS) -> pd.DataFrame:
    """The measured counts of ``remaining_silence`` next to the counts of EARLIER_COUNTS.

    One row per table (the rows of all files): table, n_files, n_fallback,
    n_lead_over (every file), n_lead_over_trimmed (without the fallback
    files), earlier and reproduced_by: "every file", "files without
    fallback", both joined by " and ", or "neither". Descriptive: the table
    has no column ``ok``.
    """
    rows = []
    for row in share[share["label"] == "all"].itertuples():
        names = [name for name, count in (("every file", row.n_lead_over),
                                          ("files without fallback", row.n_lead_over_trimmed))
                 if count == earlier[row.table]]
        rows.append({"table": row.table, "n_files": row.n_files, "n_fallback": row.n_fallback,
                     "n_lead_over": row.n_lead_over, "n_lead_over_trimmed": row.n_lead_over_trimmed,
                     "earlier": int(earlier[row.table]),
                     "reproduced_by": " and ".join(names) or "neither"})
    return pd.DataFrame(rows)


def duration_checks(table: pd.DataFrame,
                    expected: tuple[int, int] = cfg.ITW_DURATION_COUNTS,
                    window_s: float = 4.0) -> pd.DataFrame:
    """The sampling rate and the duration subsets of In-the-Wild as a check table.

    ``table`` is the ``silence_table`` of In-the-Wild. Rows: the number of
    files that were not recorded at 16 kHz (expected 0; the duration of a
    file is its number of samples at 16 kHz / 16,000, which is the original
    duration only then), the files of at most ``window_s`` and the longer
    files (``expected``). Columns: check, found, expected, ok.
    """
    seconds = table["n_samples"].to_numpy(dtype=np.float64) / cfg.SAMPLE_RATE
    rows = [("files not at %d Hz" % cfg.SAMPLE_RATE, int((table["sr_orig"] != cfg.SAMPLE_RATE).sum()), 0),
            (f"files of at most {window_s:g} s", int((seconds <= window_s).sum()), int(expected[0])),
            (f"files longer than {window_s:g} s", int((seconds > window_s).sum()), int(expected[1]))]
    return pd.DataFrame([{"check": name, "found": found, "expected": wanted, "ok": found == wanted}
                         for name, found, wanted in rows])


def compare_with_shards(long: pd.DataFrame, lengths: pd.DataFrame, top_db: float
                        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The measured trimming at ``top_db`` against the shards of the condition with that threshold.

    ``long``: ``removed_long`` rows of every split; ``lengths``:
    ``shards.shard_lengths`` of the feature folder. Returns
    (``compare_fallback_counts`` of the fallback and too-short counts per
    split and label, ``summarise_trim_lengths`` of the length of every
    file), each with a leading column ``top_db``.
    """
    from .shards import fallback_counts_of

    counts = compare_fallback_counts(fallback_counts_from_removed(long[long["top_db"] == top_db]),
                                     fallback_counts_of(lengths))
    same = summarise_trim_lengths(trim_length_diffs(long, lengths, top_db))
    for table in (counts, same):
        table.insert(0, "top_db", top_db)
    return counts, same
