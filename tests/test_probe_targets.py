"""The probe sample and the acoustic targets, with known answers."""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from signals import SR, voice, write_voiced_file_list
from ssl_silence import audio as A
from ssl_silence import probes as P
from ssl_silence import trimming as T


def strata_table() -> pd.DataFrame:
    """10 bonafide files, 25 of attack A01 and 15 of attack A02, interleaved."""
    rows = ([{"label": 0, "system_id": "-"}] * 10 + [{"label": 1, "system_id": "A01"}] * 25
            + [{"label": 1, "system_id": "A02"}] * 15)
    table = pd.DataFrame(rows).sample(frac=1.0, random_state=3).reset_index(drop=True)
    table["audio_id"] = [f"f{k}" for k in range(len(table))]
    return table


def test_stratified_sample_by_hand():
    table = strata_table()
    sample = P.stratified_sample(table, n=7)
    # Exact shares 1.4, 3.5 and 2.1: floors 1, 3, 2 and the remaining file to the largest remainder.
    assert sample.groupby(["label", "system_id"]).size().to_dict() == {
        (0, "-"): 1, (1, "A01"): 4, (1, "A02"): 2}
    assert list(sample.index) == sorted(sample.index) and sample["audio_id"].is_unique
    assert sample.equals(table.loc[sample.index])                      # rows unchanged
    assert P.stratified_sample(table, n=7).equals(sample)              # fixed seed
    assert not P.stratified_sample(table, n=7, seed=1).index.equals(sample.index)
    assert P.stratified_sample(table, n=50).equals(table)
    proportional = P.stratified_sample(table, n=20).groupby("system_id").size().to_dict()
    assert proportional == {"-": 4, "A01": 10, "A02": 6}
    with pytest.raises(ValueError):
        P.stratified_sample(table, n=51)
    assert (P.N_SAMPLE, P.SAMPLE_SEED) == (5000, 0)


def test_real_window_is_the_centre_crop_without_padding():
    y = np.arange(100_000, dtype=np.float64)
    cropped = P.real_window(y)
    assert cropped.shape == (64_000,) and cropped[0] == 18_000 and cropped[-1] == 81_999
    assert P.real_window(y[:64_000]) is not None and P.real_window(y[:64_000]).shape == (64_000,)
    short = y[:300]
    assert P.real_window(short) is short
    assert np.array_equal(P.real_window(y, window=10), y[49_995:50_005])


def test_spectral_tilt_of_flat_and_falling_spectra():
    rng = np.random.default_rng(0)
    white = rng.normal(0.0, 1.0, 2 * SR)
    times = np.arange(0.05, 1.9, 0.01)
    assert abs(P.spectral_tilt(white, times)) < 0.15                   # flat: about 0 dB per kHz
    falling = np.cumsum(white)                                         # -6 dB per octave
    falling -= np.linspace(falling[0], falling[-1], falling.size)
    assert P.spectral_tilt(falling, times) < -2.0
    # A frame needs 200 samples on each side; a frame that does not fit is left out.
    assert np.isnan(P.spectral_tilt(white, np.array([0.0, 0.01, 1.999])))
    assert np.isnan(P.spectral_tilt(white, np.array([])))
    one = P.spectral_tilt(white, np.array([0.5]))
    assert P.spectral_tilt(white, np.array([0.0, 0.5, 1.9999])) == one
    # Known answer: the slope of the line through the spectrum of one frame.
    centre, half = 8_000, 200
    magnitude = np.abs(np.fft.rfft(white[centre - half:centre + half] * np.hamming(400), n=512))
    freqs = np.fft.rfftfreq(512, 1 / SR) / 1000.0
    assert one == pytest.approx(np.polyfit(freqs, 20 * np.log10(magnitude), 1)[0], abs=1e-9)


def test_acoustic_targets_of_a_harmonic_signal():
    clean = P.acoustic_targets(voice(SR, f0=150.0))
    assert clean["f0_mean_hz"] == pytest.approx(150.0, abs=0.5)
    assert clean["n_voiced_frames"] > 80 and clean["n_hnr_frames"] > 80
    assert clean["hnr_mean_db"] > 20 and clean["tilt_db_per_khz"] < -1.0
    assert list(clean) == [*P.ACOUSTIC_TARGETS, "n_voiced_frames", "n_hnr_frames"]
    other = P.acoustic_targets(voice(SR, f0=220.0))
    assert other["f0_mean_hz"] == pytest.approx(220.0, abs=0.5)
    noisy = P.acoustic_targets(voice(SR, f0=150.0, noise=0.2))
    assert noisy["f0_mean_hz"] == pytest.approx(150.0, abs=2.0)
    assert noisy["hnr_mean_db"] < clean["hnr_mean_db"] - 5             # noise lowers the HNR
    assert noisy["tilt_db_per_khz"] > clean["tilt_db_per_khz"]         # and flattens the spectrum


def test_files_without_enough_voiced_frames_have_no_target():
    silent = P.acoustic_targets(np.zeros(SR))
    assert silent["n_voiced_frames"] == 0 and all(np.isnan(silent[k]) for k in P.ACOUSTIC_TARGETS)
    short = P.acoustic_targets(voice(1_000, f0=120.0))                 # 62 ms: fewer than 10 frames
    assert 0 <= short["n_voiced_frames"] < P.MIN_VOICED_FRAMES
    assert np.isnan(short["f0_mean_hz"]) and np.isnan(short["tilt_db_per_khz"])
    assert P.MIN_VOICED_FRAMES == 10


def test_targets_need_the_pinned_praat(monkeypatch):
    assert P.require_praat() == {"parselmouth": "0.4.7", "praat": "6.1.38"}
    monkeypatch.setattr(P, "PRAAT_VERSION", "6.4.0")
    with pytest.raises(RuntimeError, match="probe targets need"):
        P.acoustic_targets(voice(SR))


@pytest.fixture(scope="module")
def sample(tmp_path_factory):
    return write_voiced_file_list(tmp_path_factory.mktemp("voiced"))


def test_targets_are_computed_on_the_audio_of_the_condition(sample):
    for item in sample.itertuples():
        wav, sr, _ = A.load_audio(item.path)
        mono = A.to_mono_16k(wav, sr)
        clean = P.targets_for_file(item.path)
        assert clean == pytest.approx(P.acoustic_targets(P.real_window(mono[0].numpy())), nan_ok=True)
        trimmed, _ = T.apply_trim(mono, 40)
        expected = P.acoustic_targets(P.real_window(trimmed[0].numpy()))
        assert P.targets_for_file(item.path, "trim40") == pytest.approx(expected, nan_ok=True)
    # Trimming removes the silence: fewer frames without harmonicity, the same F0.
    first = sample["path"].iloc[1]
    clean, trimmed = P.targets_for_file(first), P.targets_for_file(first, "trim40")
    assert trimmed["f0_mean_hz"] == pytest.approx(clean["f0_mean_hz"], abs=1.0)
    with pytest.raises(KeyError):
        P.targets_for_file(first, "trimmed")


def test_compute_targets_is_resumable_and_records_praat(sample, tmp_path, monkeypatch):
    out_dir = str(tmp_path / "clean")
    table = P.compute_targets(sample, out_dir, chunk_size=5, progress=False)
    assert list(table.columns) == ["audio_id", "label", "system_id", *P.ACOUSTIC_TARGETS,
                                   "n_voiced_frames", "n_hnr_frames"]
    assert list(table["audio_id"]) == list(sample["audio_id"])
    assert len(os.listdir(os.path.join(out_dir, "target_chunks"))) == 3
    assert table["f0_mean_hz"].isna().sum() >= 2 and table["f0_mean_hz"].notna().sum() >= 8
    assert set(table["system_id"]) == {"-", "A01", "A02"}
    stored = P.read_targets(os.path.join(out_dir, P.TARGETS_FILE))
    pd.testing.assert_frame_equal(stored, table)
    row = table.iloc[1]
    assert row["f0_mean_hz"] == P.targets_for_file(sample["path"].iloc[1])["f0_mean_hz"]   # exact
    meta = json.loads(open(os.path.join(out_dir, "probe_targets.meta.json")).read())
    assert (meta["parselmouth"], meta["praat"], meta["condition"]) == ("0.4.7", "6.1.38", "clean")
    assert meta["praat_settings"]["pitch_ceiling_hz"] == 600.0 and meta["rows"] == len(sample)

    monkeypatch.setattr(P, "targets_for_file", lambda *a: (_ for _ in ()).throw(AssertionError()))
    pd.testing.assert_frame_equal(P.compute_targets(sample, out_dir, chunk_size=5, progress=False),
                                  table)
    with pytest.raises(RuntimeError, match="do not match the sample"):
        P.compute_targets(sample.iloc[::-1], out_dir, chunk_size=5, progress=False)
