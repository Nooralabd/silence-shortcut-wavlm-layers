"""The probe sample and the acoustic targets against the original analysis code.
Skipped without the original code."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signals import SR, voice, write_voiced_file_list
from ssl_silence import probes as P
from test_parity_silence_measurements import frame_difference
from test_probe_targets import strata_table

OLD_CONDITIONS = {"clean": "clean", "trim40": "trimmed"}


def value_difference(new, old) -> float:
    """0.0 for equal values (NaN equals NaN), else the absolute difference."""
    if isinstance(new, dict):
        if not isinstance(old, dict) or list(new) != list(old):
            return np.inf
        return max(value_difference(new[k], old[k]) for k in new)
    if type(new) is not type(old):
        return np.inf
    if isinstance(new, float) and np.isnan(new) and np.isnan(old):
        return 0.0
    return float(abs(new - old))


def record(parity_record, name, differences):
    parity_record("probes", name, len(differences), max(differences))
    assert max(differences) == 0.0, f"{name}: max abs diff {max(differences)}"


@pytest.fixture(scope="module")
def sample(tmp_path_factory):
    return write_voiced_file_list(tmp_path_factory.mktemp("voiced"), n_files=16)


def test_settings_and_sample(original, parity_record):
    O = original("probes")
    assert P.PRAAT == O.PRAAT and P.MIN_VOICED_FRAMES == O.MIN_VOICED_FRAMES
    assert (P.N_SAMPLE, P.SAMPLE_SEED) == (O.N_SAMPLE, O.SAMPLE_SEED)
    assert P.ACOUSTIC_TARGETS == O.TARGETS
    table = strata_table()
    rng = np.random.default_rng(2)
    big = pd.DataFrame({"label": (rng.random(3_000) < 0.9).astype(int)})
    big["system_id"] = np.where(big["label"] == 1, rng.choice(["A01", "A02", "A03", "A04"], 3_000), "-")
    differences = []
    for frame, sizes in ((table, (1, 7, 20, 50)), (big, (100, 999, 2_500))):
        for n in sizes:
            for seed in (0, 5):
                new, old = P.stratified_sample(frame, n, seed), O.stratified_sample(frame, n, seed)
                differences.append(0.0 if new.equals(old) else np.inf)
    record(parity_record, "stratified_sample", differences)


def test_window_tilt_and_targets_of_a_waveform(original, parity_record):
    O = original("probes")
    rng = np.random.default_rng(1)
    waves = [voice(SR, 150.0), voice(SR, 220.0, noise=0.05), voice(3 * SR, 95.0, noise=0.3, seed=2),
             voice(90_000, 180.0), rng.normal(0, 0.1, SR), np.zeros(SR), voice(1_000, 120.0),
             voice(2_400, 300.0)]
    windows = [0.0 if np.array_equal(P.real_window(y), O.real_window(y)) else np.inf for y in waves]
    record(parity_record, "real_window", windows)
    times = np.arange(0.0, 1.0, 0.01)
    tilts = [value_difference(P.spectral_tilt(y, times), O.spectral_tilt(y, times)) for y in waves]
    record(parity_record, "spectral_tilt", tilts)
    targets = [value_difference(P.acoustic_targets(y), O.acoustic_targets(y)) for y in waves]
    assert any(np.isnan(P.acoustic_targets(y)["f0_mean_hz"]) for y in waves)
    record(parity_record, "acoustic_targets", targets)


def test_targets_of_files_and_of_a_sample(sample, original, parity_record, tmp_path):
    O = original("probes")
    files, tables = [], []
    for condition, old_condition in OLD_CONDITIONS.items():
        for path in sample["path"]:
            files.append(value_difference(P.targets_for_file(path, condition),
                                          O.targets_for_file(path, old_condition)))
        new = P.compute_targets(sample, str(tmp_path / "new" / condition), condition,
                                chunk_size=6, progress=False)
        old = O.compute_targets(sample, str(tmp_path / "old" / condition), chunk_size=6,
                                condition=old_condition, progress=False)
        assert list(new.columns) == list(old.columns)
        # The chunk files, as first written by both codes, hold the same text.
        chunks = sorted((tmp_path / "old" / condition / "target_chunks").iterdir())
        assert len(chunks) == 3
        for chunk in chunks:
            mine = tmp_path / "new" / condition / "target_chunks" / chunk.name
            tables.append(0.0 if mine.read_text() == chunk.read_text() else np.inf)
        # This package returns the numbers of the chunk files exactly.
        exact = pd.concat([P.read_targets(str(chunk)) for chunk in chunks], ignore_index=True)
        tables.append(frame_difference(new, exact))
        # The original code read its chunk files with the default parser of pandas
        # and wrote the result: its table can differ in the last digit.
        assert frame_difference(new, old) < 1e-12
    record(parity_record, "targets_for_file (clean and trimmed at 40 dB)", files)
    record(parity_record, "compute_targets (chunk files as written)", tables)
