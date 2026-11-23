"""Synthetic pooled features and feature shards for the training tests."""

from __future__ import annotations

import os

import numpy as np
import torch

SPLIT_SIZES = {"train": 420, "dev": 200, "eval": 300, "itw": 250}


def gaussian_features(n: int, dim: int, seed: int, shift: float = 1.0, p_spoof: float = 0.8):
    """(x (n, dim) float32, y (n,) long, ids): two Gaussian classes ``shift`` apart."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < p_spoof).astype(np.int64)
    y[0], y[1] = 0, 1
    x = rng.normal(0.0, 1.0, (n, dim)) + shift * y[:, None] * rng.normal(1.0, 0.2, (1, dim))
    return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y), [f"u{seed}_{k}" for k in range(n)]


def write_feature_split(folder: str, split: str, n: int, n_layers: int, dim: int, seed: int,
                        shard_size: int = 128, shift: float = 1.0) -> None:
    """Write the unpadded shards of one split: layer l holds features that separate
    the classes more with depth; the standard deviation tensor is independent noise."""
    os.makedirs(folder, exist_ok=True)
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.8).astype(np.int64)
    y[0], y[1] = 0, 1
    depth = np.linspace(0.5, 1.5, n_layers)[None, :, None]
    mean = rng.normal(0.0, 1.0, (n, n_layers, dim)) + shift * depth * y[:, None, None]
    std = np.abs(rng.normal(1.0, 0.3, (n, n_layers, dim))) + 0.2 * y[:, None, None]
    for k, lo in enumerate(range(0, n, shard_size)):
        hi = min(lo + shard_size, n)
        payload = {
            "mean": torch.from_numpy(mean[lo:hi].astype(np.float32)),
            "std": torch.from_numpy(std[lo:hi].astype(np.float32)),
            "n_real": torch.full((hi - lo,), 64_000, dtype=torch.long),
            "n_frames": torch.full((hi - lo,), 199, dtype=torch.long),
            "trim_fallback": torch.zeros(hi - lo, dtype=torch.bool),
            "too_short": torch.zeros(hi - lo, dtype=torch.bool),
            "labels": torch.from_numpy(y[lo:hi]),
            "audio_ids": [f"{split}_{j}" for j in range(lo, hi)],
            "system_ids": ["-" if y[j] == 0 else "A07" for j in range(lo, hi)],
            "lo": lo, "hi": hi, "n_total": n, "shard_size": shard_size,
            "meta": {"split": split},
        }
        torch.save(payload, os.path.join(folder, f"{split}_shard_{k:04d}.pt"))


def write_feature_condition(folder: str, seed: int, n_layers: int = 5, dim: int = 8) -> None:
    """Write train, dev, eval and itw shards of one condition."""
    for k, (split, n) in enumerate(SPLIT_SIZES.items()):
        write_feature_split(folder, split, n, n_layers, dim, seed=1000 * seed + k)
