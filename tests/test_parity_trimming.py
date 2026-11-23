"""trimming against the original analysis code at 40, 20 and 15 dB.
Skipped without the original code."""

from __future__ import annotations

import numpy as np
import soundfile as sf
import torch

from signals import trimming_cases
from ssl_silence import config as cfg
from ssl_silence import trimming as T

TOP_DBS = (40, 20, 15)


def same(new, old) -> float:
    """0.0 if two outcomes are identical, else the largest absolute difference (inf if not comparable)."""
    if isinstance(new, dict):
        if not isinstance(old, dict) or list(new) != list(old):
            return np.inf
        return max(same(new[k], old[k]) for k in new)
    if isinstance(new, tuple):
        if not isinstance(old, tuple) or len(new) != len(old):
            return np.inf
        return max(same(a, b) for a, b in zip(new, old))
    if isinstance(new, torch.Tensor):
        if not isinstance(old, torch.Tensor) or new.shape != old.shape or new.dtype != old.dtype:
            return np.inf
        return 0.0 if torch.equal(new, old) else float((new.double() - old.double()).abs().max())
    if isinstance(new, np.ndarray):
        if not isinstance(old, np.ndarray) or new.shape != old.shape or new.dtype != old.dtype:
            return np.inf
        return 0.0 if np.array_equal(new, old) else float(np.max(np.abs(new - old)))
    if type(new) is not type(old):
        return np.inf
    return 0.0 if new == old else float(abs(new - old))


def compare(parity_record, name, pairs):
    worst = max(same(new, old) for new, old in pairs)
    parity_record("trimming", name, len(pairs), worst)
    assert worst == 0.0, f"{name}: max abs diff {worst}"


def test_interval_trim_and_removed_amount(original, parity_record):
    O = original("trim")
    interval, trimmed, removed = [], [], []
    for wav in trimming_cases().values():
        for top_db in TOP_DBS:
            interval.append((T.trim_interval(wav, top_db), O.trim_interval(wav, top_db)))
            trimmed.append((T.apply_trim(wav, top_db), O.apply_trim(wav, top_db)))
            removed.append((T.removed_amount_of(wav, top_db), O.removed_amount_of(wav, top_db)))
    compare(parity_record, "trim_interval", interval)
    compare(parity_record, "apply_trim", trimmed)
    compare(parity_record, "removed_amount_of", removed)


def test_frame_levels_and_bounds(original, parity_record):
    O = original("trim")
    levels, from_levels, bounds = [], [], []
    for wav in trimming_cases().values():
        y = wav[0].numpy()
        if y.size == 0 or not np.any(y):
            continue
        new_db, old_db = T.frame_db(y), O.frame_db(y)
        levels.append((new_db, old_db))
        for top_db in TOP_DBS:
            from_levels.append((T.bounds_from_db(new_db, top_db, y.size),
                                O.bounds_from_db(old_db, top_db, y.size)))
            bounds.append((T.trim_bounds(y, top_db), O.trim_bounds(y, top_db)))
    compare(parity_record, "frame_db", levels)
    compare(parity_record, "bounds_from_db", from_levels)
    compare(parity_record, "trim_bounds", bounds)


def test_resolve_trim(original, parity_record):
    O = original("trim")
    pairs = []
    for n in (0, 399, 8_000, 30_000):
        for start, end in ((0, 0), (0, n), (1_000, 8_999), (1_000, 9_000), (1_000, 9_001)):
            for all_zero in (False, True):
                pairs.append((T.resolve_trim(n, start, end, all_zero),
                              O.resolve_trim(n, start, end, all_zero)))
    compare(parity_record, "resolve_trim", pairs)


def test_removed_amount_from_files(original, parity_record, tmp_path):
    O = original("trim")
    rng = np.random.default_rng(21)
    single, several = [], []
    for k, (name, wav) in enumerate(trimming_cases().items()):
        if wav.shape[1] == 0:
            continue
        sr = (cfg.SAMPLE_RATE, cfg.SAMPLE_RATE, 22_050, 48_000)[k % 4]
        samples = (wav[0].numpy() * 32767).astype(np.int16)
        if k % 3 == 0:
            samples = np.stack([samples, (rng.normal(0, 30, samples.size)).astype(np.int16)], axis=1)
        path = tmp_path / f"{name}.flac"
        sf.write(path, samples, sr, subtype="PCM_16")
        several.append((T.removed_amounts(str(path), TOP_DBS), O.removed_amounts(str(path), TOP_DBS)))
        for top_db in TOP_DBS:
            single.append((T.removed_amount(str(path), top_db), O.removed_amount(str(path), top_db)))
    compare(parity_record, "removed_amount", single)
    compare(parity_record, "removed_amounts", several)
