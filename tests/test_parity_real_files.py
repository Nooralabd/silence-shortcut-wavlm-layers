"""Audio loading, windows and trimming on real dataset files against the
original analysis code. Runs on Colab: skipped unless the original code is
present and SSL_SILENCE_REAL_FILES names a CSV file with the column ``path``
(a few files of every split, written by notebooks/01_environment_check.ipynb).

Every outcome is compared with ``==``. The sha256 of each file is part of
the record, so the files that were compared can be identified later.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
import torch

from ssl_silence import audio
from ssl_silence import provenance
from ssl_silence import trimming
from test_parity_audio import same

TOP_DBS = (40, 20, 15)


@pytest.fixture(scope="module")
def real_files() -> pd.DataFrame:
    listing = os.environ.get("SSL_SILENCE_REAL_FILES")
    if not listing:
        pytest.skip("SSL_SILENCE_REAL_FILES is not set (a CSV file with the column path)")
    table = pd.read_csv(listing)
    assert len(table) > 0 and all(os.path.exists(p) for p in table["path"])
    return table


def test_real_files_are_read_and_trimmed_identically(real_files, original, original_function,
                                                     parity_record):
    O_audio, O_trim, O_features = original("audio"), original("trim"), original("features")
    old_padded = original_function("audio", "padded_window")
    loaded, mono, windows, trimmed = [], [], [], []
    rows = []
    for path in real_files["path"]:
        new_wav, new_sr, new_backend = audio.load_audio(path)
        old_wav, old_sr, old_backend = O_audio.load_audio(path)
        loaded.append(((new_wav, new_sr, new_backend), (old_wav, old_sr, old_backend)))
        new_mono, old_mono = audio.to_mono_16k(new_wav, new_sr), O_audio.to_mono_16k(old_wav, old_sr)
        mono.append((new_mono, old_mono))
        windows.append((audio.padded_window(new_mono), old_padded(old_mono)))
        windows.append((audio.unpadded_window(new_mono), O_features.unpadded_window(old_mono)))
        for top_db in TOP_DBS:
            trimmed.append((trimming.trim_interval(new_mono, top_db),
                            O_trim.trim_interval(old_mono, top_db)))
            a, b = trimming.apply_trim(new_mono, top_db), O_trim.apply_trim(old_mono, top_db)
            trimmed.append(((a[0], bool(a[1])), (b[0], bool(b[1]))))
        rows.append({"sha256": provenance.sha256_file(path), "sr_orig": int(new_sr),
                     "n_channels": int(new_wav.shape[0]), "n_samples_16k": int(new_mono.shape[1]),
                     "backend": new_backend})
    for name, pairs in (("load_audio (real files)", loaded), ("to_mono_16k (real files)", mono),
                        ("padded and unpadded window (real files)", windows),
                        ("trim_interval and apply_trim at 40, 20 and 15 dB (real files)", trimmed)):
        worst = max(same(new, old) for new, old in pairs)
        parity_record("real_files", name, len(pairs), worst)
        assert worst == 0.0, f"{name}: max abs diff {worst}"
    listing = os.environ.get("SSL_SILENCE_REAL_FILES_RECORD")
    if listing:
        pd.DataFrame(rows).to_csv(listing, index=False)
    assert len({row["sha256"] for row in rows}) == len(rows)
    assert isinstance(loaded[0][0][0], torch.Tensor) and np.isfinite(rows[0]["sr_orig"])
