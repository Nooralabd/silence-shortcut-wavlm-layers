"""Ridge probes, their targets and tables against the original analysis code.
Skipped without the original code."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ssl_silence import probes as P
from ssl_silence import shards
from synthetic import write_feature_split
from test_parity_silence_measurements import frame_difference
from test_probes import linear_problem, measurements, probe_inputs

OLD_CONDITIONS = {"clean": "clean", "trim40": "trimmed"}


def record(parity_record, name, differences):
    parity_record("probes", name, len(differences), max(differences))
    assert max(differences) == 0.0, f"{name}: max abs diff {max(differences)}"


def test_ridge_probe(original, parity_record):
    O = original("probes")
    assert np.array_equal(P.ALPHAS, O.ALPHAS) and (P.N_FOLDS, P.CV_SEED) == (O.N_FOLDS, O.CV_SEED)
    assert P.CV_SCHEMES == O.CV_SCHEMES
    differences = []
    for seed, (n, dim, n_speakers) in enumerate(((150, 6, 7), (240, 12, 20), (90, 4, 5))):
        x, y, groups = linear_problem(n, dim, n_speakers, seed)
        y = y + np.random.default_rng(seed).normal(0.0, 1.0 + seed, n)      # not a perfect fit
        for g in (None, groups):
            for target in (y, np.random.default_rng(0).permutation(y)):
                new, old = P.ridge_probe_r2(x, target, g), O.ridge_probe_r2(x, target, g)
                assert list(new) == list(old)
                differences += [abs(new["r2_mean"] - old["r2_mean"]), abs(new["r2_sd"] - old["r2_sd"]),
                                abs(new["alpha_median"] - old["alpha_median"]),
                                float(np.max(np.abs(np.subtract(new["r2_folds"], old["r2_folds"]))))]
    record(parity_record, "ridge_probe_r2 (random and speaker folds)", differences)


def test_probes_of_all_layers_and_table(original, parity_record):
    O = original("probes")
    features, targets, groups = probe_inputs()
    names = ("f0_mean_hz", "pad")
    new = P.probe_all_layers(features, targets, names, groups=groups, progress=False)
    old = O.probe_all_layers(features, targets, names, groups=groups, progress=False)
    assert list(new.columns) == list(old.columns) and len(new) == len(old) == 24
    record(parity_record, "probe_all_layers (real and shuffled targets)",
           [frame_difference(new, old)])
    new_table, old_table = P.probe_table(new), O.probe_table(old)
    assert list(new_table.columns) == list(old_table.columns)
    record(parity_record, "probe_table", [frame_difference(new_table, old_table)])


def test_silence_targets(original, parity_record):
    O = original("probes")
    table = measurements().assign(win_trail_40=[100, 200, 300])
    rng = np.random.default_rng(4)
    big = pd.DataFrame({"audio_id": [f"f{k}" for k in range(400)],
                        "win_lead_40": rng.integers(0, 20_000, 400),
                        "win_trail_40": rng.integers(0, 9_000, 400),
                        "pad": rng.integers(0, 50_000, 400),
                        "pad_trimmed_40": rng.integers(0, 60_000, 400),
                        "trim_fallback_40": (rng.random(400) < 0.1).astype(int)})
    differences = []
    for frame in (table, big):
        for condition, old_condition in OLD_CONDITIONS.items():
            new, old = P.condition_targets(frame, condition), O.condition_targets(frame, old_condition)
            assert list(new.index) == list(old.index)
            differences.append(frame_difference(new, old[list(new.columns)]))
            targets = pd.DataFrame({"audio_id": list(frame["audio_id"])[::-1][:3], "x": 1.0})
            joined = P.attach_silence_targets(targets, frame, condition)
            reference = O.attach_silence_targets(targets, frame, old_condition,
                                                 columns=P.SILENCE_TARGETS)
            assert list(joined.columns) == list(reference.columns)
            differences.append(frame_difference(joined, reference))
    record(parity_record, "condition_targets and attach_silence_targets", differences)


def test_sample_features(original, parity_record, tmp_path):
    O = original("probes")
    feature_dir, cache_dir = str(tmp_path / "features"), str(tmp_path / "cache")
    write_feature_split(feature_dir, "train", n=300, n_layers=4, dim=5, seed=7)
    shards.build_layer_cache(feature_dir, "train", cache_dir)
    rng = np.random.default_rng(0)
    differences = []
    for n in (1, 40, 300):
        ids = [f"train_{k}" for k in rng.permutation(300)[:n]]
        new, _ = P.load_sample_features(cache_dir, "train", ids)
        old = O.load_sample_features(cache_dir, "train", ids, key="mean")
        assert new.shape == old.shape and new.dtype == old.dtype
        differences.append(float(np.max(np.abs(new - old))))
    record(parity_record, "load_sample_features", differences)
