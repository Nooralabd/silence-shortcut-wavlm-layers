"""Audio loading and preprocessing.

    load -> mono -> resample to 16 kHz -> (trimmed conditions: trim) ->
    window -> peak normalisation x / (max|x| + 1e-8)

The waveform is passed to the encoder directly; no feature-extractor
normalisation is applied. The two windows are the inputs of the two
extractions: ``unpadded_window`` (primary) and ``padded_window`` (comparison).
"""

from __future__ import annotations

import torch

from . import config as cfg

BACKENDS = ("auto", "torchaudio", "soundfile")


def load_audio(path: str, backend: str = "auto") -> tuple[torch.Tensor, int, str]:
    """Read an audio file as float32 in [-1, 1].

    ``backend`` is "torchaudio", "soundfile" or "auto" (torchaudio if it can
    read the file, otherwise soundfile).

    Returns (waveform (channels, samples), sampling rate in Hz, backend used).
    """
    if backend not in BACKENDS:
        raise ValueError(backend)
    if backend in ("auto", "torchaudio"):
        try:
            import torchaudio

            wav, sr = torchaudio.load(path)
            return wav, int(sr), "torchaudio"
        except (ImportError, RuntimeError, OSError):
            if backend == "torchaudio":
                raise
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32", always_2d=True)  # (samples, channels)
    return torch.from_numpy(data.T.copy()), int(sr), "soundfile"


def to_mono_16k(wav: torch.Tensor, sr: int) -> torch.Tensor:
    """(channels, samples) at ``sr`` Hz -> (1, samples) at 16 kHz."""
    if wav.dim() != 2:
        raise ValueError(f"expected (channels, samples), got {tuple(wav.shape)}")
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != cfg.SAMPLE_RATE:
        import torchaudio

        wav = torchaudio.transforms.Resample(sr, cfg.SAMPLE_RATE)(wav)
    return wav


def padded_window(wav: torch.Tensor, window: int = cfg.WINDOW) -> tuple[torch.Tensor, int]:
    """Centre crop or zero padding at the end to ``window`` samples, then peak normalisation.

    ``wav`` is (1, samples), mono at 16 kHz.

    Returns (waveform (window,), number of real, unpadded samples).
    """
    n = wav.shape[1]
    if n > window:
        start = (n - window) // 2
        wav = wav[:, start:start + window]
        n_real = window
    elif n < window:
        wav = torch.nn.functional.pad(wav, (0, window - n))
        n_real = n
    else:
        n_real = window

    wav = wav / (wav.abs().max() + 1e-8)
    return wav.squeeze(0), n_real


def unpadded_window(wav: torch.Tensor, window: int = cfg.WINDOW,
                    min_samples: int = cfg.MIN_SAMPLES) -> tuple[torch.Tensor, int, bool]:
    """Real samples inside the window, without padding, then peak normalisation.

    A waveform longer than ``window`` is centre-cropped; a shorter one keeps
    its length. A waveform with fewer than ``min_samples`` samples is
    zero-padded to ``min_samples`` and flagged. ``wav`` is (1, samples), mono
    at 16 kHz.

    Returns (waveform (n,), number of real samples, too_short).
    """
    n = wav.shape[1]
    if n > window:
        start = (n - window) // 2
        wav = wav[:, start:start + window]
        n = window
    too_short = n < min_samples
    if too_short:
        wav = torch.nn.functional.pad(wav, (0, min_samples - n))
    wav = wav / (wav.abs().max() + 1e-8)
    return wav.squeeze(0), int(n), bool(too_short)
