"""Trimming of leading and trailing silence.

Silence is detected with ``librosa.effects.trim`` on the mono 16 kHz
waveform: a frame (400 samples, hop 160) is silent when its RMS lies more
than ``top_db`` dB below the loudest frame of the same file. Only leading and
trailing silence is removed; pauses inside the utterance stay.

librosa floors every frame RMS at 1e-5 before taking dB, so the threshold is
relative to the loudest frame only down to that floor: in a file whose
loudest frame has an RMS below 1e-5 * 10 ** (top_db / 20) (1e-3 at 40 dB,
1e-4 at 20 dB), no frame can be silent and nothing is trimmed. Trimming runs
before peak normalisation.

Fallback: a file that is all zeros, or that would keep fewer than 0.5 s
(8,000 samples), keeps its untrimmed audio and is flagged.

A supplementary condition uses a voice activity detector instead of the
energy rule (``vad_interval``): the WebRTC detector labels frames of the same
waveform as speech or not, the kept audio runs from the first to the last
run of consecutive speech frames, and the fallback is the same.

All lengths are in samples at 16 kHz.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from . import config as cfg

REMOVED_PER_DB_KEYS = ("n_samples_trimmed", "removed_lead_s", "removed_trail_s",
                       "removed_total_s", "fallback", "too_short")


def frame_rms(y: np.ndarray) -> np.ndarray:
    """RMS of every frame (400 samples, hop 160). y (N,) -> (n_frames,)."""
    import librosa

    return librosa.feature.rms(y=y, frame_length=cfg.TRIM_FRAME_LENGTH,
                               hop_length=cfg.TRIM_HOP_LENGTH)[0]


def rms_to_db(rms: np.ndarray) -> np.ndarray:
    """Frame RMS -> dB relative to the loudest frame (RMS floored at 1e-5 by librosa)."""
    import librosa

    return librosa.amplitude_to_db(rms, ref=np.max, top_db=None)


def frame_db(y: np.ndarray) -> np.ndarray:
    """Frame RMS in dB relative to the loudest frame. y (N,) -> (n_frames,).

    The computation inside ``librosa.effects.trim``, exposed so that several
    thresholds can be evaluated from one pass.
    """
    return rms_to_db(frame_rms(y))


def bounds_from_db(db: np.ndarray, top_db: float, n_samples: int) -> tuple[int, int]:
    """Non-silent interval [start, end) in samples from the frame levels of ``frame_db``."""
    nonzero = np.flatnonzero(db > -top_db)
    if nonzero.size == 0:
        return 0, 0
    start = int(nonzero[0]) * cfg.TRIM_HOP_LENGTH
    end = min(n_samples, (int(nonzero[-1]) + 1) * cfg.TRIM_HOP_LENGTH)
    return start, end


def trim_bounds(y: np.ndarray, top_db: float) -> tuple[int, int]:
    """Non-silent interval [start, end) of y (N,) from ``librosa.effects.trim``."""
    import librosa

    _, index = librosa.effects.trim(y, top_db=top_db, frame_length=cfg.TRIM_FRAME_LENGTH,
                                    hop_length=cfg.TRIM_HOP_LENGTH)
    return int(index[0]), int(index[1])


def resolve_trim(n_samples: int, start: int, end: int, all_zero: bool) -> tuple[int, int, bool]:
    """Apply the fallback rule to detected bounds.

    Returns (start, end, fallback). With fallback the interval is
    [0, n_samples): the file is all zeros or fewer than
    ``config.MIN_TRIMMED_SAMPLES`` would remain.
    """
    if all_zero or end - start < cfg.MIN_TRIMMED_SAMPLES:
        return 0, n_samples, True
    return start, end, False


def trim_interval(wav: torch.Tensor, top_db: float) -> tuple[int, int, bool]:
    """Interval [start, end) that ``apply_trim`` keeps, and its fallback flag.

    ``wav`` is (1, N), mono at 16 kHz. ``apply_trim`` and ``removed_amount_of``
    both use this function, so the measured removed amount is what the
    trimmed condition removes.
    """
    if wav.dim() != 2 or wav.shape[0] != 1:
        raise ValueError(f"expected (1, samples), got {tuple(wav.shape)}")
    y = wav[0].numpy()
    n = int(y.shape[0])
    all_zero = n == 0 or not np.any(y)
    start, end = (0, n) if all_zero else trim_bounds(y, top_db)
    return resolve_trim(n, start, end, all_zero)


def apply_trim(wav: torch.Tensor, top_db: float) -> tuple[torch.Tensor, bool]:
    """Trim leading and trailing silence with the fallback rule.

    ``wav`` is (1, N), mono at 16 kHz.

    Returns (waveform (1, M) with M <= N, fallback flag). With fallback the
    input is returned unchanged.
    """
    start, end, fallback = trim_interval(wav, top_db)
    return wav[:, start:end], fallback


def removed_amount_of(wav: torch.Tensor, top_db: float,
                      sr: int = cfg.SAMPLE_RATE) -> dict[str, float | int]:
    """What ``apply_trim`` removes from one untrimmed waveform (1, N).

    Returns a flat dict: top_db, n_samples_orig, n_samples_trimmed (samples
    that remain), orig_s, removed_lead_s, removed_trail_s, removed_total_s,
    fallback (0/1) and too_short (0/1: fewer than ``config.MIN_SAMPLES``
    samples remain). A fallback file has 0 removed.
    """
    start, end, fallback = trim_interval(wav, top_db)
    n = int(wav.shape[1])
    lead, trail = (0, 0) if fallback else (start, n - end)
    kept = n - lead - trail
    return {"top_db": top_db, "n_samples_orig": n, "n_samples_trimmed": kept,
            "orig_s": n / sr, "removed_lead_s": lead / sr, "removed_trail_s": trail / sr,
            "removed_total_s": (lead + trail) / sr, "fallback": int(fallback),
            "too_short": int(kept < cfg.MIN_SAMPLES)}


def removed_amount(path: str, top_db: float) -> dict[str, float | int]:
    """``removed_amount_of`` of a file, loaded as the extraction loads it."""
    from .audio import load_audio, to_mono_16k

    wav, sr, _ = load_audio(path)
    return removed_amount_of(to_mono_16k(wav, sr), top_db)


def removed_amounts_of(wav: torch.Tensor, top_dbs: Sequence[float]) -> dict[str, float | int]:
    """``removed_amount_of`` of one untrimmed waveform (1, N) at several thresholds.

    Returns one flat dict: n_samples_orig, orig_s and, for every threshold d,
    the keys of REMOVED_PER_DB_KEYS with the suffix ``_<d>``
    (removed_lead_s_40, fallback_20, ...).
    """
    out: dict[str, float | int] = {}
    for d in top_dbs:
        r = removed_amount_of(wav, d)
        out["n_samples_orig"], out["orig_s"] = r["n_samples_orig"], r["orig_s"]
        for k in REMOVED_PER_DB_KEYS:
            out[f"{k}_{d:g}"] = r[k]
    return out


def removed_amounts(path: str, top_dbs: Sequence[float]) -> dict[str, float | int]:
    """``removed_amounts_of`` of a file, loaded once as the extraction loads it."""
    from .audio import load_audio, to_mono_16k

    wav, sr, _ = load_audio(path)
    return removed_amounts_of(to_mono_16k(wav, sr), top_dbs)


# Voice activity detection -------------------------------------------------------------------

VAD_FRAME_MS = (10, 20, 30)          # the frame lengths the WebRTC detector accepts


def to_pcm16(y: np.ndarray) -> np.ndarray:
    """Float waveform (N,) in [-1, 1) -> 16-bit PCM, little endian. A sample
    that was read from 16-bit audio gets its integer value back; values
    outside the range are clipped."""
    return np.clip(np.round(np.asarray(y, dtype=np.float64) * 32768.0), -32768, 32767).astype("<i2")


def vad_speech_frames(y: np.ndarray, frame_ms: int, aggressiveness: int,
                      sr: int = cfg.SAMPLE_RATE) -> np.ndarray:
    """Speech decision of the WebRTC detector for every whole frame of y (N,).

    A new detector is created for every waveform and the frames are given to
    it in order (the detector adapts to what it has heard), so the decisions
    of a file depend on that file only. Samples after the last whole frame
    are not judged. Returns (n_frames,) bool.
    """
    import webrtcvad

    if frame_ms not in VAD_FRAME_MS:
        raise ValueError(f"frame_ms must be one of {VAD_FRAME_MS}, got {frame_ms}")
    if aggressiveness not in (0, 1, 2, 3):
        raise ValueError(f"aggressiveness must be 0, 1, 2 or 3, got {aggressiveness}")
    size = sr * frame_ms // 1000
    pcm = to_pcm16(y)
    detector = webrtcvad.Vad(int(aggressiveness))
    return np.array([detector.is_speech(pcm[k * size:(k + 1) * size].tobytes(), sr)
                     for k in range(len(pcm) // size)], dtype=bool)


def bounds_from_speech(speech: np.ndarray, run_frames: int, frame_samples: int) -> tuple[int, int]:
    """Interval [start, end) in samples from the frame decisions: from the
    first frame of the first run of ``run_frames`` consecutive speech frames
    to the end of the last frame of the last such run. (0, 0) without a run."""
    if run_frames < 1:
        raise ValueError("run_frames must be at least 1")
    speech = np.asarray(speech, dtype=bool)
    if speech.size < run_frames:
        return 0, 0
    windows = np.lib.stride_tricks.sliding_window_view(speech, run_frames).all(axis=1)
    starts = np.flatnonzero(windows)
    if starts.size == 0:
        return 0, 0
    return int(starts[0]) * frame_samples, (int(starts[-1]) + run_frames) * frame_samples


def vad_interval(wav: torch.Tensor, rule: dict) -> tuple[int, int, bool]:
    """Interval [start, end) that ``apply_vad`` keeps, and its fallback flag.

    ``wav`` is (1, N), mono at 16 kHz; ``rule`` has frame_ms, aggressiveness
    and run_frames (``config.vad_rule``). The fallback is that of the energy
    rule (``resolve_trim``): no run of speech frames, or fewer than
    ``config.MIN_TRIMMED_SAMPLES`` left.
    """
    if wav.dim() != 2 or wav.shape[0] != 1:
        raise ValueError(f"expected (1, samples), got {tuple(wav.shape)}")
    y = wav[0].numpy()
    n = int(y.shape[0])
    speech = vad_speech_frames(y, int(rule["frame_ms"]), int(rule["aggressiveness"]))
    start, end = bounds_from_speech(speech, int(rule["run_frames"]),
                                    cfg.SAMPLE_RATE * int(rule["frame_ms"]) // 1000)
    return resolve_trim(n, start, end, all_zero=end == 0)


def apply_vad(wav: torch.Tensor, rule: dict) -> tuple[torch.Tensor, bool, int, int]:
    """Remove leading and trailing non-speech with the fallback rule.

    Returns (waveform (1, M), fallback flag, start, end): the kept audio is
    ``wav[:, start:end]``; with fallback it is the input and the interval is
    [0, N).
    """
    start, end, fallback = vad_interval(wav, rule)
    return wav[:, start:end], fallback, start, end
