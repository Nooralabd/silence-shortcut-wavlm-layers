"""Feature shards on disk.

A shard is one ``torch.save`` dict with the pooled features of up to
``shard_size`` consecutive files of a split:

    features        (n, L, D) float32 tensors; layer index 0 is transformer
                    layer 1. unpadded: mean, std. padded: all_frames_mean,
                    all_frames_std, real_frames_mean, real_frames_std
    n_real          (n,) long   input samples after trimming and the centre crop
    n_frames        (n,) long   frames of the encoder output
    n_real_frames   (n,) long   padded only: frames of the real samples
    trim_fallback   (n,) bool   the untrimmed audio was kept
    too_short       (n,) bool   fewer than config.MIN_SAMPLES real samples
    labels          (n,) long   0 = bonafide, 1 = spoof
    audio_ids, system_ids       list[str]
    lo, hi, n_total, shard_size int: the shard holds files lo..hi-1 of n_total
    layers          optional list[int]: the 1-based layers stored, when not all
    n_samples, kept_start, kept_end
                    optional (n,) long: samples of the file at 16 kHz before
                    trimming, and the interval [kept_start, kept_end) that
                    was kept (stored for a condition trimmed by voice
                    activity detection)
    meta            dict: encoder, checkpoint, versions, device, commit

``read_shard`` returns these keys only. Any other key of a file is ignored
and never exposed.
"""

from __future__ import annotations

import glob
import os
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from . import config as cfg

FEATURE_KEYS = {"unpadded": ("mean", "std"),
                "padded": ("all_frames_mean", "all_frames_std", "real_frames_mean",
                           "real_frames_std")}
PER_FILE_KEYS = {"unpadded": ("n_real", "n_frames", "trim_fallback", "too_short", "labels"),
                 "padded": ("n_real", "n_frames", "n_real_frames", "trim_fallback", "too_short",
                            "labels")}
LIST_KEYS = ("audio_ids", "system_ids")
RANGE_KEYS = ("lo", "hi", "n_total", "shard_size")
KEPT_KEYS = ("n_samples", "kept_start", "kept_end")
OPTIONAL_KEYS = ("layers",) + KEPT_KEYS
FRAME_SETS = ("all-frames", "real-frames")
# Keys of the shard meta that the local cache keeps (see ``build_layer_cache``).
CACHE_META_KEYS = ("encoder", "revision", "git_commit")


def schema(extraction: str) -> tuple[str, ...]:
    """Required keys of a shard of ``extraction``, besides ``meta``."""
    if extraction not in FEATURE_KEYS:
        raise KeyError(f"unknown extraction {extraction!r}; known: {sorted(FEATURE_KEYS)}")
    return FEATURE_KEYS[extraction] + PER_FILE_KEYS[extraction] + LIST_KEYS + RANGE_KEYS


def shard_path(feature_dir: str, split: str, k: int) -> str:
    """Path of shard ``k`` of a split."""
    return os.path.join(feature_dir, f"{split}_shard_{k:04d}.pt")


def n_shards(n_total: int, shard_size: int) -> int:
    """Number of shards that hold ``n_total`` files."""
    return (n_total + shard_size - 1) // shard_size


def write_shard(path: str, payload: dict[str, Any], extraction: str) -> None:
    """Write a shard atomically after checking its keys and lengths."""
    missing = [key for key in schema(extraction) + ("meta",) if key not in payload]
    unknown = [key for key in payload if key not in schema(extraction) + ("meta",) + OPTIONAL_KEYS]
    if missing or unknown:
        raise KeyError(f"shard keys: missing {missing}, unknown {unknown}")
    n = payload["hi"] - payload["lo"]
    kept = tuple(key for key in KEPT_KEYS if key in payload)
    if kept and kept != KEPT_KEYS:
        raise KeyError(f"shard keys: {kept} without the others of {KEPT_KEYS}")
    for key in FEATURE_KEYS[extraction] + PER_FILE_KEYS[extraction] + LIST_KEYS + kept:
        if len(payload[key]) != n:
            raise ValueError(f"{key} holds {len(payload[key])} files, expected {n}")
    cfg.require_drive(path)
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def read_shard(path: str, extraction: str = "unpadded", mmap: bool = False) -> dict[str, Any]:
    """Read a shard and return the keys of the schema only.

    With ``mmap`` the tensors are memory-mapped and not loaded.
    """
    cfg.require_drive(path)
    raw = torch.load(path, map_location="cpu", weights_only=False, mmap=mmap)
    missing = [key for key in schema(extraction) if key not in raw]
    if missing:
        raise KeyError(f"{path}: missing keys {missing}")
    out = {key: raw[key] for key in schema(extraction)}
    out["layers"] = [int(layer) for layer in raw["layers"]] if "layers" in raw else None
    out.update({key: raw[key] for key in KEPT_KEYS if key in raw})
    out["meta"] = dict(raw.get("meta", {}))
    return out


def split_shards(feature_dir: str, split: str, extraction: str = "unpadded",
                 mmap: bool = True) -> list[dict[str, Any]]:
    """All shards of a split in order; raises unless they cover the split without gaps."""
    paths = sorted(glob.glob(os.path.join(feature_dir, f"{split}_shard_*.pt")))
    if not paths:
        raise FileNotFoundError(f"no shards of split {split!r} in {feature_dir}")
    shards, expected_lo, n_total = [], 0, None
    for path in paths:
        shard = read_shard(path, extraction, mmap=mmap)
        if shard["lo"] != expected_lo:
            raise RuntimeError(f"gap before {path}: expected lo={expected_lo}, found {shard['lo']}")
        if n_total is not None and shard["n_total"] != n_total:
            raise RuntimeError(f"{path}: n_total {shard['n_total']} differs from {n_total}")
        expected_lo, n_total = shard["hi"], shard["n_total"]
        shards.append(shard)
    if expected_lo != n_total:
        raise RuntimeError(f"split {split!r} is incomplete: {expected_lo} of {n_total} files")
    return shards


def shard_lengths(feature_dir: str, splits: Sequence[str] = cfg.SPLITS,
                  extraction: str = "unpadded") -> pd.DataFrame:
    """One row per file in shard order: split, audio_id, label, n_real, trim_fallback, too_short."""
    parts = []
    for split in splits:
        for shard in split_shards(feature_dir, split, extraction):
            parts.append(pd.DataFrame({
                "split": split,
                "audio_id": [str(a) for a in shard["audio_ids"]],
                "label": np.asarray(shard["labels"]).astype(int),
                "n_real": np.asarray(shard["n_real"]).astype(int),
                "trim_fallback": np.asarray(shard["trim_fallback"]).astype(bool),
                "too_short": np.asarray(shard["too_short"]).astype(bool)}))
    return pd.concat(parts, ignore_index=True)


def kept_intervals(feature_dir: str, splits: Sequence[str] = cfg.SPLITS,
                   extraction: str = "unpadded") -> pd.DataFrame:
    """One row per file in shard order, for shards that store the kept
    interval: split, audio_id, system_id, label, n_samples, kept_start,
    kept_end, n_real, trim_fallback. Raises KeyError for shards without it."""
    parts = []
    for split in splits:
        for shard in split_shards(feature_dir, split, extraction):
            missing = [key for key in KEPT_KEYS if key not in shard]
            if missing:
                raise KeyError(f"the shards of {feature_dir} do not store {missing}")
            parts.append(pd.DataFrame({
                "split": split,
                "audio_id": [str(a) for a in shard["audio_ids"]],
                "system_id": [str(a) for a in shard["system_ids"]],
                "label": np.asarray(shard["labels"]).astype(int),
                **{key: np.asarray(shard[key]).astype(int) for key in KEPT_KEYS},
                "n_real": np.asarray(shard["n_real"]).astype(int),
                "trim_fallback": np.asarray(shard["trim_fallback"]).astype(bool)}))
    return pd.concat(parts, ignore_index=True)


def fallback_counts(feature_dir: str, splits: Sequence[str] = cfg.SPLITS,
                    extraction: str = "unpadded") -> pd.DataFrame:
    """Files that kept their untrimmed audio, per split and label.

    One row per split x label ("0", "1", "all"): split, label, n_files,
    n_fallback, n_too_short, pct_fallback.
    """
    return fallback_counts_of(shard_lengths(feature_dir, splits, extraction), splits)


def fallback_counts_of(lengths: pd.DataFrame, splits: Sequence[str] | None = None) -> pd.DataFrame:
    """``fallback_counts`` from the rows of ``shard_lengths`` (the shards are not read again).

    ``splits`` defaults to the splits of ``lengths`` in order of appearance.
    """
    splits = list(lengths["split"].unique()) if splits is None else splits
    rows = []
    for split in splits:
        part = lengths[lengths["split"] == split]
        y = part["label"].to_numpy()
        fallback, short = part["trim_fallback"].to_numpy(), part["too_short"].to_numpy()
        for label, mask in (("0", y == 0), ("1", y == 1), ("all", np.ones(y.size, bool))):
            rows.append({"split": split, "label": label, "n_files": int(mask.sum()),
                         "n_fallback": int(fallback[mask].sum()),
                         "n_too_short": int(short[mask].sum()),
                         "pct_fallback": float(100 * fallback[mask].mean()) if mask.any()
                         else float("nan")})
    return pd.DataFrame(rows)


def _layer_index(shard: dict[str, Any], layer: int, n_stored: int) -> int:
    stored = shard["layers"] if shard["layers"] is not None else list(range(1, n_stored + 1))
    if len(stored) != n_stored:
        raise ValueError(f"shard stores {n_stored} layers but lists {stored}")
    if layer not in stored:
        raise ValueError(f"layer {layer} is not stored; stored layers: {stored}")
    return stored.index(layer)


def load_layer(feature_dir: str, split: str, layer: int, pooling: str,
               extraction: str = "unpadded", frames: str | None = None
               ) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Features of one layer of a split.

    ``layer`` is 1-based. ``pooling`` is "mean" (D values) or "stats" (mean
    and standard deviation, 2 D values). For the padded extraction ``frames``
    is "all-frames" or "real-frames".

    Returns (features (N, D or 2 D) float32, labels (N,) long, audio ids).
    """
    if pooling not in cfg.POOLINGS:
        raise ValueError(f"pooling must be one of {cfg.POOLINGS}, got {pooling!r}")
    if extraction == "unpadded":
        if frames is not None:
            raise ValueError("frames applies to the padded extraction only")
        mean_key, std_key = "mean", "std"
    else:
        if frames not in FRAME_SETS:
            raise ValueError(f"frames must be one of {FRAME_SETS}, got {frames!r}")
        prefix = frames.replace("-", "_")
        mean_key, std_key = f"{prefix}_mean", f"{prefix}_std"
    xs, ys, ids = [], [], []
    for shard in split_shards(feature_dir, split, extraction):
        index = _layer_index(shard, layer, shard[mean_key].shape[1])
        mean = shard[mean_key][:, index, :]
        x = mean if pooling == "mean" else torch.cat([mean, shard[std_key][:, index, :]], dim=-1)
        xs.append(x.clone())
        ys.append(torch.as_tensor(shard["labels"]).clone())
        ids.extend(str(a) for a in shard["audio_ids"])
    return torch.cat(xs).float(), torch.cat(ys).long(), ids


# Local per-layer cache -------------------------------------------------------------

def build_layer_cache(feature_dir: str, split: str, cache_dir: str,
                      extraction: str = "unpadded") -> str:
    """Copy the shards of a split into arrays that are read one layer at a time.

    Reading one layer from the shards means reading every shard completely.
    This writes, once, one ``<split>_<key>.npy`` array of shape (L, N, D)
    float32 per feature key (layer first) and ``<split>_index.pt`` (labels,
    audio ids, system ids, stored layers, and under ``meta`` the distinct
    values of ``CACHE_META_KEYS`` that the shards record). The values are
    copied unchanged. Skipped when the index file exists and has ``meta``; it
    is written last.

    Returns the path of the index file.
    """
    index_path = os.path.join(cache_dir, f"{split}_index.pt")
    if os.path.exists(index_path) and "meta" in torch.load(index_path, map_location="cpu",
                                                           weights_only=False):
        return index_path
    keys = FEATURE_KEYS[extraction]
    os.makedirs(cache_dir, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    labels, audio_ids, system_ids, layers = [], [], [], None
    meta: dict[str, list[str]] = {key: [] for key in CACHE_META_KEYS}
    for shard in split_shards(feature_dir, split, extraction, mmap=False):
        for key in CACHE_META_KEYS:
            value = shard["meta"].get(key)
            if value is not None and str(value) not in meta[key]:
                meta[key].append(str(value))
        n_stored, dim = shard[keys[0]].shape[1:]
        stored = shard["layers"] if shard["layers"] is not None else list(range(1, n_stored + 1))
        if not arrays:
            layers = stored
            arrays = {key: np.lib.format.open_memmap(
                os.path.join(cache_dir, f"{split}_{key}.npy"), mode="w+", dtype=np.float32,
                shape=(n_stored, shard["n_total"], dim)) for key in keys}
        elif stored != layers:
            raise ValueError(f"shards of {split!r} store different layers")
        for key in keys:
            arrays[key][:, shard["lo"]:shard["hi"], :] = shard[key].numpy().transpose(1, 0, 2)
        labels.append(torch.as_tensor(shard["labels"]).clone())
        audio_ids.extend(str(a) for a in shard["audio_ids"])
        system_ids.extend(str(s) for s in shard["system_ids"])
    for array in arrays.values():
        array.flush()
    del arrays
    tmp = index_path + ".tmp"
    torch.save({"labels": torch.cat(labels).long(), "audio_ids": audio_ids,
                "system_ids": system_ids, "layers": layers, "keys": list(keys),
                "meta": meta}, tmp)
    os.replace(tmp, index_path)
    return index_path


def cache_meta(cache_dir: str, split: str) -> dict[str, list[str]]:
    """What the shards behind a cached split record: for each of
    ``CACHE_META_KEYS`` the distinct values, in shard order (empty when the
    shards do not record the key)."""
    index = torch.load(os.path.join(cache_dir, f"{split}_index.pt"), map_location="cpu",
                       weights_only=False)
    return {key: list(index["meta"].get(key, [])) for key in CACHE_META_KEYS}


def load_layer_from_cache(cache_dir: str, split: str, layer: int, pooling: str,
                          extraction: str = "unpadded", frames: str | None = None
                          ) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """``load_layer`` from the local cache written by ``build_layer_cache``."""
    if pooling not in cfg.POOLINGS:
        raise ValueError(f"pooling must be one of {cfg.POOLINGS}, got {pooling!r}")
    if extraction == "unpadded":
        if frames is not None:
            raise ValueError("frames applies to the padded extraction only")
        mean_key, std_key = "mean", "std"
    else:
        if frames not in FRAME_SETS:
            raise ValueError(f"frames must be one of {FRAME_SETS}, got {frames!r}")
        prefix = frames.replace("-", "_")
        mean_key, std_key = f"{prefix}_mean", f"{prefix}_std"
    index = torch.load(os.path.join(cache_dir, f"{split}_index.pt"), map_location="cpu",
                       weights_only=False)
    if layer not in index["layers"]:
        raise ValueError(f"layer {layer} is not stored; stored layers: {index['layers']}")
    row = index["layers"].index(layer)

    def read(key: str) -> torch.Tensor:
        array = np.load(os.path.join(cache_dir, f"{split}_{key}.npy"), mmap_mode="r")
        return torch.from_numpy(np.array(array[row]))

    x = read(mean_key) if pooling == "mean" else torch.cat([read(mean_key), read(std_key)], dim=-1)
    return x.float(), index["labels"].long(), list(index["audio_ids"])
