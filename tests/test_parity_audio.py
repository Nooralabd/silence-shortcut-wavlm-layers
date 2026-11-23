"""audio against the original analysis code. Skipped without the original code."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf
import torch

from ssl_silence import audio
from ssl_silence import config as cfg

SAMPLE_RATES = (8_000, 16_000, 22_050, 44_100, 48_000)
# Lengths at 16 kHz: around the minimum, shorter than, equal to and longer than
# the window, with odd and even excess for the centre crop.
LENGTHS = (1, 2, 399, 400, 401, 8_000, 20_001, 63_999, 64_000, 64_001, 64_002, 64_003,
           100_001, 130_000)


def signals(n: int, rng) -> dict[str, np.ndarray]:
    """Named float32 signals of n samples."""
    t = np.arange(n, dtype=np.float64)
    impulse = np.zeros(n)
    impulse[n // 2] = -0.25
    return {"tone": 0.3 * np.sin(2 * np.pi * 220.0 * t / cfg.SAMPLE_RATE),
            "noise": rng.normal(0.0, 0.1, n),
            "silence": np.zeros(n),
            "impulse_negative_peak": impulse,
            "tiny": rng.normal(0.0, 1e-9, n),
            "full_scale": np.clip(rng.normal(0.0, 0.6, n), -1.0, 1.0)}


def same(new, old) -> float:
    """0.0 if two outcomes are identical, else the largest absolute difference (inf if not comparable)."""
    if isinstance(new, tuple):
        if not isinstance(old, tuple) or len(new) != len(old):
            return np.inf
        return max(same(a, b) for a, b in zip(new, old))
    if isinstance(new, torch.Tensor):
        if not isinstance(old, torch.Tensor) or new.shape != old.shape or new.dtype != old.dtype:
            return np.inf
        if torch.equal(new, old):
            return 0.0
        return float((new.double() - old.double()).abs().max())
    return 0.0 if (new == old and type(new) is type(old)) else np.inf


def compare(parity_record, name, pairs):
    worst = max(same(new, old) for new, old in pairs)
    parity_record("audio", name, len(pairs), worst)
    assert worst == 0.0, f"{name}: max abs diff {worst}"


def test_windows(original, original_function, parity_record):
    old_padded = original_function("audio", "padded_window")
    old_unpadded = original("features").unpadded_window
    rng = np.random.default_rng(11)
    padded, unpadded = [], []
    for n in LENGTHS:
        for values in signals(n, rng).values():
            wav = torch.from_numpy(values.astype(np.float32)).unsqueeze(0)
            padded.append((audio.padded_window(wav), old_padded(wav)))
            unpadded.append((audio.unpadded_window(wav), old_unpadded(wav)))
    compare(parity_record, "padded_window", padded)
    compare(parity_record, "unpadded_window", unpadded)


def test_to_mono_16k(original, parity_record):
    pytest.importorskip("torchaudio")
    O = original("audio")
    rng = np.random.default_rng(12)
    pairs = []
    for sr in SAMPLE_RATES:
        for seconds in (0.02, 0.5, 3.999, 4.0, 4.001, 6.5):
            n = int(round(seconds * sr)) | 1 if seconds == 6.5 else int(round(seconds * sr))
            for name, values in signals(n, rng).items():
                mono = torch.from_numpy(values.astype(np.float32)).unsqueeze(0)
                other = torch.from_numpy(rng.normal(0.0, 0.1, n).astype(np.float32)).unsqueeze(0)
                for wav in (mono, torch.cat([mono, other]), torch.cat([mono, -mono])):
                    pairs.append((audio.to_mono_16k(wav, sr), O.to_mono_16k(wav, sr)))
    compare(parity_record, "to_mono_16k", pairs)


def test_load_audio(original, parity_record, tmp_path):
    O = original("audio")
    rng = np.random.default_rng(13)
    pairs = []
    for sr in SAMPLE_RATES:
        for channels in (1, 2):
            for suffix in ("wav", "flac"):
                n = int(rng.integers(100, 5_000))
                samples = rng.integers(-32768, 32768, (n, channels)).astype(np.int16)
                path = tmp_path / f"{sr}_{channels}.{suffix}"
                sf.write(path, samples, sr, subtype="PCM_16")
                for backend in ("auto", "soundfile"):
                    pairs.append((audio.load_audio(str(path), backend),
                                  O.load_audio(str(path), backend)))
    compare(parity_record, "load_audio", pairs)


def test_load_to_window_pipeline(original, original_function, parity_record, tmp_path):
    """File -> load -> mono 16 kHz -> both windows, as the extraction chains them."""
    pytest.importorskip("torchaudio")
    O = original("audio")
    old_padded = original_function("audio", "padded_window")
    old_unpadded = original("features").unpadded_window
    rng = np.random.default_rng(14)
    pairs = []
    for sr in SAMPLE_RATES:
        for channels in (1, 2):
            for seconds in (0.01, 1.3, 4.0, 5.7):
                n = int(round(seconds * sr))
                samples = (rng.normal(0.0, 0.2, (n, channels)).clip(-1, 1) * 32767).astype(np.int16)
                path = tmp_path / f"{sr}_{channels}_{n}.flac"
                sf.write(path, samples, sr, subtype="PCM_16")
                new_wav, new_sr, _ = audio.load_audio(str(path))
                old_wav, old_sr, _ = O.load_audio(str(path))
                new_mono, old_mono = audio.to_mono_16k(new_wav, new_sr), O.to_mono_16k(old_wav, old_sr)
                pairs.append(((audio.padded_window(new_mono), audio.unpadded_window(new_mono)),
                              (old_padded(old_mono), old_unpadded(old_mono))))
    compare(parity_record, "load -> mono 16 kHz -> windows", pairs)
