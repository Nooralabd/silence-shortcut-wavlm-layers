"""Synthetic waveforms shared by the trimming tests."""

from __future__ import annotations

import numpy as np
import torch

SR = 16_000
WEAK = 10 ** (-30 / 20)  # 30 dB below the tone


def tone(n: int, amplitude: float = 0.5, hz: float = 220.0) -> np.ndarray:
    """n samples of a sine at 16 kHz."""
    return amplitude * np.sin(2 * np.pi * hz * np.arange(n) / SR)


def clip(lead: int, body: int, trail: int, amplitude: float = 0.5, weak: int = 0,
         floor: float = 0.0, seed: int = 0) -> torch.Tensor:
    """(1, N) float32: ``lead`` silent samples, a tone of ``body`` samples, a weak
    segment of ``weak`` samples 30 dB below the tone, ``trail`` silent samples.
    ``floor`` is the standard deviation of noise added everywhere (0 = exact zeros)."""
    parts = [np.zeros(lead), tone(body, amplitude), tone(weak, amplitude * WEAK), np.zeros(trail)]
    y = np.concatenate(parts)
    if floor:
        y = y + np.random.default_rng(seed).normal(0.0, floor, y.size)
    return torch.from_numpy(y.astype(np.float32)).unsqueeze(0)


MS = SR // 1000  # samples per millisecond


def voice(n: int, f0: float = 150.0, harmonics: int = 20, noise: float = 0.0,
          amplitude: float = 0.3, seed: int = 0) -> np.ndarray:
    """n samples of a harmonic complex at ``f0`` (harmonic k has amplitude 1/k),
    with white noise of standard deviation ``noise`` (relative to the amplitude)."""
    t = np.arange(n) / SR
    y = sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, harmonics + 1)
            if k * f0 < SR / 2)
    y = amplitude * y / np.max(np.abs(y))
    if noise:
        y = y + amplitude * noise * np.random.default_rng(seed).normal(0.0, 1.0, n)
    return y


def write_voiced_file_list(folder, n_files: int = 12, seed: int = 0, n_speakers: int = 4
                           ) -> "pd.DataFrame":
    """Write voiced files with leading and trailing silence, in the layout of
    ``datasets.items``. They differ in F0, noise, length (some longer than 4 s)
    and silence; one file is noise only, one is all zeros and one is very short."""
    import pandas as pd
    import soundfile as sf

    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_files):
        body = int(rng.integers(12_000, 60_000)) if k % 4 else 80_000
        y = voice(body, f0=float(rng.uniform(90, 260)), noise=float(rng.choice([0.0, 0.02, 0.2])),
                  seed=k)
        if k == 3:
            y = 0.1 * rng.normal(0.0, 1.0, body)
        if k == 5:
            y = np.zeros(body)
        if k == 7:
            y = voice(1_000, f0=120.0)
        y = np.concatenate([np.zeros(int(rng.integers(0, 12_000))), y,
                            np.zeros(int(rng.integers(0, 12_000)))])
        path = folder / f"voiced_{k:02d}.flac"
        sf.write(path, (np.clip(y, -1, 1) * 32767).astype(np.int16), SR, subtype="PCM_16")
        label = int(k % 3 > 0)
        rows.append({"dataset": "synthetic", "split": "train", "audio_id": f"voiced_{k:02d}",
                     "path": str(path), "label": label,
                     "system_id": f"A{1 + k % 2:02d}" if label else "-",
                     "speaker": f"spk{k % n_speakers}"})
    return pd.DataFrame(rows)


def weak_edges(weak_lead: int, weak_trail: int, lead: int = 8_000, body: int = 24_000,
               trail: int = 8_000, amplitude: float = 0.5) -> torch.Tensor:
    """(1, N) float32: silence, a weak segment 30 dB below the tone, the tone,
    a weak segment, silence. Trimming at 40 dB keeps the weak segments; at 20
    and 15 dB they are silence."""
    parts = [np.zeros(lead), tone(weak_lead, amplitude * WEAK), tone(body, amplitude),
             tone(weak_trail, amplitude * WEAK), np.zeros(trail)]
    return torch.from_numpy(np.concatenate(parts).astype(np.float32)).unsqueeze(0)


def residual_cases() -> dict[str, torch.Tensor]:
    """Waveforms that keep more than 50 ms of weak sound at an edge after trimming at 40 dB."""
    return {"weak_lead_80ms": weak_edges(80 * MS, 0),
            "weak_trail_200ms": weak_edges(0, 200 * MS),
            "weak_both_edges": weak_edges(200 * MS, 80 * MS)}


BLIP_CENTRE = 8_000  # a frame centre (50 hops)
BLIP_GAP = 1_600     # 100 ms of zeros between the blip and the tone


def blip_before_tone(level_db: float, gap: int = BLIP_GAP, amplitude: float = 0.5) -> torch.Tensor:
    """(1, N) float32: zeros, a blip of one frame length (400 samples, 2 kHz)
    centred on a frame centre, ``gap`` zeros, a tone, zeros. ``level_db`` is the
    level of the blip relative to the tone (negative).

    The frame centred on the blip holds all of it; its two neighbours hold 60 %
    (2.2 dB less). After a cut at the frame centre, the first frame holds the
    second half of the blip only (3 dB less).
    """
    blip = tone(400, amplitude * 10 ** (level_db / 20), hz=2_000.0)
    parts = [np.zeros(BLIP_CENTRE - 200), blip, np.zeros(gap), tone(24_000, amplitude),
             np.zeros(8_000)]
    return torch.from_numpy(np.concatenate(parts).astype(np.float32)).unsqueeze(0)


def blip_cases() -> dict[str, torch.Tensor]:
    """A blip 35 dB below the tone and one 39 dB below (one frame above a 40 dB
    threshold). In this order ``write_file_list`` writes the second at 16 kHz,
    so that its frames are not moved by resampling."""
    return {"blip_35db_before_tone": blip_before_tone(-35.0),
            "blip_39db_before_tone": blip_before_tone(-39.0)}


def trimming_cases() -> dict[str, torch.Tensor]:
    """Named waveforms covering the trimming rule and its edge cases."""
    cases = {
        **residual_cases(),
        **blip_cases(),
        "padded_tone": clip(8_000, 24_000, 12_000),
        "padded_tone_long_silence": clip(40_000, 30_000, 50_000),
        "weak_segment_after_tone": clip(8_000, 24_000, 12_000, weak=16_000),
        "weak_segment_noise_floor": clip(8_000, 24_000, 12_000, weak=16_000, floor=1e-5, seed=1),
        "leading_silence_only": clip(16_000, 32_000, 0),
        "trailing_silence_only": clip(0, 32_000, 16_000),
        "no_silence": clip(0, 48_000, 0),
        "all_zeros": torch.zeros(1, 32_000),
        "empty": torch.zeros(1, 0),
        "single_sample": torch.tensor([[0.3]]),
        "below_half_second_after_trim": clip(16_000, 4_800, 16_000),
        "half_second_tone_7900": clip(16_000, 7_900, 16_000),
        "half_second_tone_8000": clip(16_000, 8_000, 16_000),
        "half_second_tone_8200": clip(16_000, 8_200, 16_000),
        "half_second_tone_9000": clip(16_000, 9_000, 16_000),
        "short_file_no_silence": clip(0, 6_000, 0),
        "noise_floor_silence": clip(8_000, 24_000, 12_000, floor=1e-4, seed=2),
        "loud_noise_floor": clip(8_000, 24_000, 12_000, floor=2e-2, seed=3),
        "quiet_recording": clip(8_000, 24_000, 12_000, amplitude=1e-3),
        "full_scale": clip(8_000, 24_000, 12_000, amplitude=1.0),
        "noise_only": clip(0, 0, 40_000, floor=0.1, seed=4),
    }
    for n in (399, 400, 401, 63_999, 64_000, 64_001, 96_000):
        cases[f"tone_{n}"] = clip(0, n, 0)
        if n >= 20_000:
            cases[f"padded_to_{n}"] = clip(n // 4, n - n // 2, n // 4)
            cases[f"padded_to_{n}_weak"] = clip(n // 4, n // 4, n // 4, weak=n // 4)
    return cases


def write_file_list(folder, split: str = "eval", seed: int = 0, other_rates: bool = True
                    ) -> "pd.DataFrame":
    """Write the trimming cases as audio files and return a file list in the
    layout of ``datasets.items``: dataset, split, audio_id, path, label,
    system_id, speaker. Some files are stereo or at another sampling rate
    unless ``other_rates`` is false. Labels, attacks and speakers depend on ``seed``."""
    import pandas as pd
    import soundfile as sf

    rng = np.random.default_rng(seed)
    rows = []
    for k, (name, wav) in enumerate(trimming_cases().items()):
        if wav.shape[1] == 0:
            continue
        sr = (SR, SR, 22_050, 48_000)[k % 4] if other_rates else SR
        samples = (wav[0].numpy() * 32767).astype(np.int16)
        if other_rates and k % 5 == 0:
            samples = np.stack([samples, rng.normal(0, 30, samples.size).astype(np.int16)], axis=1)
        path = folder / f"{split}_{name}.flac"
        sf.write(path, samples, sr, subtype="PCM_16")
        label = int(rng.random() < 0.6)
        rows.append({"dataset": "synthetic", "split": split, "audio_id": f"{split}_{name}",
                     "path": str(path), "label": label,
                     "system_id": f"A{7 + k % 3:02d}" if label else "-",
                     "speaker": f"spk{int(rng.integers(0, 5))}"})
    table = pd.DataFrame(rows)
    table.loc[table.index[:2], "label"] = [0, 1]
    table.loc[table.index[:2], "system_id"] = ["-", "A07"]
    return table


def write_dataset(folder) -> "pd.DataFrame":
    """Write a small synthetic dataset and return its file list.

    Columns: path, audio_id, label, system_id. The files cover a padded tone,
    a weak segment, a file longer than the window, a file shorter than 400
    samples, an all-zero file, a file below 0.5 s, a stereo file and a file
    at another sampling rate.
    """
    import pandas as pd
    import soundfile as sf

    waves = {
        "padded_tone": clip(8_000, 24_000, 12_000),
        "weak_segment": clip(8_000, 24_000, 12_000, weak=16_000),
        "long_file": clip(10_000, 60_000, 10_000),
        "tiny_file": clip(0, 300, 0),
        "all_zeros": torch.zeros(1, 20_000),
        "short_tone": clip(4_000, 6_000, 4_000),
        "noise_floor": clip(8_000, 30_000, 8_000, floor=1e-3, seed=5),
    }
    rows = []
    for k, (name, wav) in enumerate(waves.items()):
        path = folder / f"{name}.flac"
        sf.write(path, (wav[0].numpy() * 32767).astype(np.int16), SR, subtype="PCM_16")
        rows.append({"path": str(path), "audio_id": name, "label": k % 2,
                     "system_id": "-" if k % 2 == 0 else "A07"})
    stereo = np.stack([clip(6_000, 20_000, 6_000)[0].numpy(),
                       clip(6_000, 20_000, 6_000, floor=1e-2, seed=6)[0].numpy()], axis=1)
    path = folder / "stereo.wav"
    sf.write(path, (stereo * 32767).astype(np.int16), SR, subtype="PCM_16")
    rows.append({"path": str(path), "audio_id": "stereo", "label": 1, "system_id": "A08"})
    path = folder / "other_rate.flac"
    sf.write(path, (clip(11_025, 33_075, 11_025)[0].numpy() * 32767).astype(np.int16), 22_050,
             subtype="PCM_16")
    rows.append({"path": str(path), "audio_id": "other_rate", "label": 0, "system_id": "-"})
    return pd.DataFrame(rows)
