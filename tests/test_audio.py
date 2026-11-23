"""Audio preprocessing on synthetic signals with known answers."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf
import torch

from ssl_silence import audio
from ssl_silence import config as cfg


def ramp(n: int) -> torch.Tensor:
    """(1, n) waveform whose sample i has the value i + 1."""
    return torch.arange(1, n + 1, dtype=torch.float32).unsqueeze(0)


def test_window_constants():
    assert cfg.SAMPLE_RATE == 16_000 and cfg.WINDOW == 64_000 and cfg.MIN_SAMPLES == 400


# Mono and sampling rate ----------------------------------------------------------

def test_stereo_is_the_mean_of_the_channels():
    wav = torch.tensor([[1.0, 2.0, 3.0], [3.0, 0.0, -3.0]])
    out = audio.to_mono_16k(wav, cfg.SAMPLE_RATE)
    assert out.shape == (1, 3)
    assert torch.equal(out, torch.tensor([[2.0, 1.0, 0.0]]))


def test_mono_at_16_khz_is_returned_unchanged():
    wav = ramp(10)
    assert audio.to_mono_16k(wav, cfg.SAMPLE_RATE) is wav


def test_wrong_shape_is_rejected():
    with pytest.raises(ValueError):
        audio.to_mono_16k(torch.zeros(100), cfg.SAMPLE_RATE)


@pytest.mark.parametrize("sr, n", [(8_000, 8_000), (22_050, 22_050), (44_100, 44_100),
                                   (48_000, 48_000)])
def test_resampling_gives_one_second_at_16_khz(sr, n):
    pytest.importorskip("torchaudio")
    t = torch.arange(n, dtype=torch.float32) / sr
    wav = torch.sin(2 * torch.pi * 440.0 * t).unsqueeze(0)
    out = audio.to_mono_16k(wav, sr)
    assert out.shape == (1, cfg.SAMPLE_RATE)
    # A 440 Hz tone stays a 440 Hz tone of the same amplitude away from the edges.
    target = torch.sin(2 * torch.pi * 440.0 * torch.arange(cfg.SAMPLE_RATE) / cfg.SAMPLE_RATE)
    assert torch.allclose(out[0, 2000:-2000], target[2000:-2000], atol=5e-3)


# Padded window -----------------------------------------------------------------

def test_padded_window_pads_zeros_at_the_end():
    out, n_real = audio.padded_window(ramp(5), window=8)
    assert n_real == 5
    assert torch.allclose(out, torch.tensor([1, 2, 3, 4, 5, 0, 0, 0]) / (5 + 1e-8))


def test_padded_window_exact_length_is_only_normalised():
    out, n_real = audio.padded_window(ramp(8), window=8)
    assert n_real == 8 and out.shape == (8,)
    assert torch.allclose(out, ramp(8)[0] / (8 + 1e-8))


@pytest.mark.parametrize("n, first", [(9, 1), (10, 2), (11, 2), (12, 3), (13, 3)])
def test_padded_window_centre_crop_starts_at_half_the_excess(n, first):
    # start = (n - window) // 2; sample values are index + 1.
    out, n_real = audio.padded_window(ramp(n), window=8)
    kept = torch.arange(first, first + 8, dtype=torch.float32)
    assert n_real == 8
    assert torch.allclose(out, kept / (kept.max() + 1e-8))


def test_peak_normalisation_uses_the_absolute_peak():
    wav = torch.tensor([[0.1, -0.5, 0.25]])
    out, _ = audio.padded_window(wav, window=3)
    assert out.abs().max().item() == pytest.approx(1.0, abs=1e-6)
    assert out[1].item() == pytest.approx(-1.0, abs=1e-6)
    assert torch.equal(out, wav[0] / (torch.tensor(0.5) + 1e-8))


def test_silence_stays_zero():
    out, n_real = audio.padded_window(torch.zeros(1, 5), window=8)
    assert n_real == 5 and torch.equal(out, torch.zeros(8))
    out, n_real, too_short = audio.unpadded_window(torch.zeros(1, 500), window=8_000)
    assert torch.equal(out, torch.zeros(500)) and not too_short


def test_default_window_is_four_seconds():
    out, n_real = audio.padded_window(torch.ones(1, 70_001))
    assert out.shape == (cfg.WINDOW,) and n_real == cfg.WINDOW


# Unpadded window ---------------------------------------------------------------

def test_unpadded_window_keeps_a_short_waveform_at_its_length():
    out, n_real, too_short = audio.unpadded_window(ramp(600), window=1_000)
    assert out.shape == (600,) and n_real == 600 and not too_short
    assert torch.allclose(out, ramp(600)[0] / (600 + 1e-8))


@pytest.mark.parametrize("n, first", [(9, 1), (10, 2), (11, 2), (12, 3)])
def test_unpadded_window_centre_crop(n, first):
    out, n_real, too_short = audio.unpadded_window(ramp(n), window=8, min_samples=4)
    kept = torch.arange(first, first + 8, dtype=torch.float32)
    assert n_real == 8 and not too_short
    assert torch.allclose(out, kept / (kept.max() + 1e-8))


@pytest.mark.parametrize("n, expected_length, flagged", [(399, 400, True), (400, 400, False),
                                                         (401, 401, False), (1, 400, True)])
def test_unpadded_window_minimum_length(n, expected_length, flagged):
    out, n_real, too_short = audio.unpadded_window(ramp(n))
    assert out.shape == (expected_length,) and n_real == n and too_short is flagged
    assert torch.equal(out[n:], torch.zeros(expected_length - n))


def test_both_windows_agree_on_the_real_samples():
    rng = np.random.default_rng(0)
    for n in (500, 63_999, 64_000, 64_001, 80_003):
        wav = torch.from_numpy(rng.normal(0, 0.1, (1, n)).astype(np.float32))
        padded, n_real_padded = audio.padded_window(wav)
        unpadded, n_real, too_short = audio.unpadded_window(wav)
        assert n_real == n_real_padded == min(n, cfg.WINDOW) and not too_short
        assert torch.equal(unpadded, padded[:n_real])
        assert torch.equal(padded[n_real:], torch.zeros(cfg.WINDOW - n_real))


# Loading -----------------------------------------------------------------------

@pytest.mark.parametrize("suffix", ["wav", "flac"])
def test_load_audio_reads_16_bit_pcm_exactly(tmp_path, suffix):
    samples = np.array([[0, 16384], [-16384, 32767], [-32768, 1]], dtype=np.int16)
    path = tmp_path / f"a.{suffix}"
    sf.write(path, samples, 22_050, subtype="PCM_16")
    wav, sr, backend = audio.load_audio(str(path))
    assert sr == 22_050 and backend in ("torchaudio", "soundfile")
    assert wav.dtype == torch.float32 and wav.shape == (2, 3)
    assert torch.equal(wav, torch.from_numpy(samples.T.astype(np.float32) / 32768.0))


def test_load_audio_backends(tmp_path):
    path = tmp_path / "a.wav"
    sf.write(path, np.zeros(10, dtype=np.int16), 16_000, subtype="PCM_16")
    wav, sr, backend = audio.load_audio(str(path), backend="soundfile")
    assert backend == "soundfile" and wav.shape == (1, 10) and sr == 16_000
    with pytest.raises(ValueError):
        audio.load_audio(str(path), backend="other")
