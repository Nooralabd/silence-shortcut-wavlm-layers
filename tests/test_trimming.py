"""The trimming rule on synthetic waveforms with known answers."""

from __future__ import annotations

import librosa
import numpy as np
import pytest
import soundfile as sf
import torch

from signals import clip, trimming_cases
from ssl_silence import config as cfg
from ssl_silence import trimming as T

TOP_DBS = (40, 20, 15)
EDGE = cfg.TRIM_FRAME_LENGTH  # detection resolution at a boundary, samples


def test_rule_constants():
    assert (cfg.TRIM_FRAME_LENGTH, cfg.TRIM_HOP_LENGTH) == (400, 160)
    assert cfg.MIN_TRIMMED_SAMPLES == 8_000 and cfg.MIN_SAMPLES == 400


def test_librosa_is_called_with_the_frame_settings(monkeypatch):
    seen = {}

    def fake_trim(y, **kwargs):
        seen.update(kwargs, n=y.shape[0], dtype=y.dtype)
        return y, np.array([10, 20])

    monkeypatch.setattr(librosa.effects, "trim", fake_trim)
    wav = clip(100, 9_000, 100)
    assert T.trim_bounds(wav[0].numpy(), 20) == (10, 20)
    assert seen == {"top_db": 20, "frame_length": 400, "hop_length": 160, "n": 9_200,
                    "dtype": np.float32}


@pytest.mark.parametrize("top_db", TOP_DBS)
def test_padded_tone_is_trimmed_to_the_tone(top_db):
    wav = clip(8_000, 24_000, 12_000)
    start, end, fallback = T.trim_interval(wav, top_db)
    assert not fallback
    assert abs(start - 8_000) <= EDGE and abs(end - 32_000) <= EDGE
    out, flag = T.apply_trim(wav, top_db)
    assert not flag and torch.equal(out, wav[:, start:end])


def test_weak_segment_is_kept_at_40_db_and_removed_at_20_and_15():
    # A segment 30 dB below the tone follows it: samples 32,000 to 48,000.
    wav = clip(8_000, 24_000, 12_000, weak=16_000)
    _, end_40, _ = T.trim_interval(wav, 40)
    assert abs(end_40 - 48_000) <= EDGE
    for top_db in (20, 15):
        _, end, fallback = T.trim_interval(wav, top_db)
        assert not fallback and abs(end - 32_000) <= EDGE


@pytest.mark.parametrize("top_db", TOP_DBS)
def test_silence_on_one_side_only(top_db):
    start, end, _ = T.trim_interval(clip(16_000, 32_000, 0), top_db)
    assert abs(start - 16_000) <= EDGE and end == 48_000
    start, end, _ = T.trim_interval(clip(0, 32_000, 16_000), top_db)
    assert start == 0 and abs(end - 32_000) <= EDGE
    assert T.trim_interval(clip(0, 48_000, 0), top_db) == (0, 48_000, False)


@pytest.mark.parametrize("top_db", TOP_DBS)
def test_the_rule_does_not_depend_on_the_recording_level_above_the_floor(top_db):
    reference = T.trim_interval(clip(8_000, 24_000, 12_000, weak=16_000), top_db)
    for amplitude in (0.02, 0.1, 1.0):
        assert T.trim_interval(clip(8_000, 24_000, 12_000, amplitude=amplitude, weak=16_000),
                               top_db) == reference


def test_very_quiet_file_is_not_trimmed_at_40_db():
    # librosa floors every frame RMS at 1e-5 before taking dB. A tone of
    # amplitude 1e-3 has an RMS of 7.1e-4, so exact silence lies only 37 dB
    # below it: nothing is silent at 40 dB, and the silence is found at 20 dB.
    wav = clip(8_000, 24_000, 12_000, amplitude=1e-3)
    assert T.trim_interval(wav, 40) == (0, 44_000, False)
    start, end, fallback = T.trim_interval(wav, 20)
    assert not fallback and abs(start - 8_000) <= EDGE and abs(end - 32_000) <= EDGE
    assert T.frame_db(wav[0].numpy()).min() == pytest.approx(
        20 * np.log10(1e-5 / (1e-3 / np.sqrt(2))), abs=0.05)


# Fallback --------------------------------------------------------------------

def test_resolve_trim_at_the_half_second_boundary():
    assert T.resolve_trim(30_000, 1_000, 8_999, False) == (0, 30_000, True)
    assert T.resolve_trim(30_000, 1_000, 9_000, False) == (1_000, 9_000, False)
    assert T.resolve_trim(30_000, 1_000, 9_001, False) == (1_000, 9_001, False)
    assert T.resolve_trim(30_000, 0, 30_000, True) == (0, 30_000, True)


@pytest.mark.parametrize("top_db", TOP_DBS)
def test_all_zero_empty_and_short_files_fall_back(top_db):
    for wav in (torch.zeros(1, 32_000), torch.zeros(1, 0), clip(0, 6_000, 0),
                clip(16_000, 4_800, 16_000)):
        n = wav.shape[1]
        assert T.trim_interval(wav, top_db) == (0, n, True)
        out, fallback = T.apply_trim(wav, top_db)
        assert fallback and torch.equal(out, wav)
        amount = T.removed_amount_of(wav, top_db)
        assert amount["fallback"] == 1 and amount["removed_total_s"] == 0.0
        assert amount["n_samples_trimmed"] == n


def test_all_zero_file_is_not_passed_to_librosa(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("librosa must not be called")

    monkeypatch.setattr(librosa.effects, "trim", fail)
    assert T.trim_interval(torch.zeros(1, 20_000), 40) == (0, 20_000, True)


def test_wrong_shape_is_rejected():
    for wav in (torch.zeros(1_000), torch.zeros(2, 1_000)):
        with pytest.raises(ValueError):
            T.trim_interval(wav, 40)


# Frame levels ------------------------------------------------------------------

@pytest.mark.parametrize("top_db", (50, 40, 30, 20, 15))
def test_bounds_from_frame_levels_equal_the_librosa_interval(top_db):
    for name, wav in trimming_cases().items():
        y = wav[0].numpy()
        if y.size == 0 or not np.any(y):
            continue
        db = T.frame_db(y)
        assert db.max() == 0.0
        assert T.bounds_from_db(db, top_db, y.size) == T.trim_bounds(y, top_db), name


def test_bounds_from_frame_levels_without_a_frame_above_the_threshold():
    assert T.bounds_from_db(np.array([-50.0, -60.0]), 40, 320) == (0, 0)
    assert T.bounds_from_db(np.array([-50.0, 0.0, -50.0]), 40, 400) == (160, 320)
    assert T.bounds_from_db(np.array([0.0, 0.0, 0.0]), 40, 400) == (0, 400)


# Removed amount ----------------------------------------------------------------

@pytest.mark.parametrize("top_db", TOP_DBS)
def test_removed_amount_is_what_apply_trim_removes(top_db):
    for name, wav in trimming_cases().items():
        out, fallback = T.apply_trim(wav, top_db)
        start, end, _ = T.trim_interval(wav, top_db)
        amount = T.removed_amount_of(wav, top_db)
        n = wav.shape[1]
        assert amount["top_db"] == top_db and amount["n_samples_orig"] == n, name
        assert amount["n_samples_trimmed"] == out.shape[1] == end - start, name
        assert amount["fallback"] == int(fallback), name
        assert amount["too_short"] == int(out.shape[1] < cfg.MIN_SAMPLES), name
        assert amount["orig_s"] == n / cfg.SAMPLE_RATE, name
        assert amount["removed_lead_s"] == start / cfg.SAMPLE_RATE, name
        assert amount["removed_trail_s"] == (n - end) / cfg.SAMPLE_RATE, name
        assert amount["removed_total_s"] == (start + n - end) / cfg.SAMPLE_RATE, name


def test_removed_amount_known_values():
    amount = T.removed_amount_of(clip(8_000, 24_000, 12_000), 40)
    assert amount["removed_lead_s"] == pytest.approx(0.5, abs=EDGE / cfg.SAMPLE_RATE)
    assert amount["removed_trail_s"] == pytest.approx(0.75, abs=EDGE / cfg.SAMPLE_RATE)
    assert amount["orig_s"] == 2.75 and amount["fallback"] == 0 and amount["too_short"] == 0
    short = T.removed_amount_of(clip(0, 399, 0), 40)
    assert short["fallback"] == 1 and short["too_short"] == 1 and short["n_samples_trimmed"] == 399
    assert T.removed_amount_of(clip(0, 400, 0), 40)["too_short"] == 0


def test_removed_amounts_from_a_file(tmp_path):
    wav = clip(8_000, 24_000, 12_000, weak=16_000)
    path = tmp_path / "a.flac"
    sf.write(path, (wav[0].numpy() * 32767).astype(np.int16), cfg.SAMPLE_RATE, subtype="PCM_16")
    wide = T.removed_amounts(str(path), TOP_DBS)
    assert wide["n_samples_orig"] == 60_000 and wide["orig_s"] == 3.75
    assert set(wide) == {"n_samples_orig", "orig_s"} | {
        f"{key}_{d}" for key in T.REMOVED_PER_DB_KEYS for d in TOP_DBS}
    for d in TOP_DBS:
        single = T.removed_amount(str(path), d)
        for key in T.REMOVED_PER_DB_KEYS:
            assert wide[f"{key}_{d}"] == single[key]
    # The weak segment (1 s) is removed at 20 and 15 dB only.
    assert wide["removed_trail_s_20"] - wide["removed_trail_s_40"] == pytest.approx(
        1.0, abs=2 * EDGE / cfg.SAMPLE_RATE)
    assert wide["removed_trail_s_15"] >= wide["removed_trail_s_20"]
