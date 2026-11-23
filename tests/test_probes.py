"""Ridge probes per layer, their targets and tables, with known answers."""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest
import torch

from signals import write_voiced_file_list
from ssl_silence import probes as P
from ssl_silence import shards
from ssl_silence import silence as S
from synthetic import write_feature_split


def linear_problem(n: int = 300, dim: int = 8, n_speakers: int = 12, seed: int = 0):
    """Features, a target that is linear in them, and speaker ids."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, (n, dim))
    y = x @ rng.normal(0.0, 1.0, dim) + 0.05 * rng.normal(0.0, 1.0, n)
    groups = np.array([f"s{k % n_speakers}" for k in range(n)])
    return x, y, groups


def test_a_linear_target_is_read_and_a_permuted_one_is_not():
    x, y, groups = linear_problem()
    for g in (None, groups):
        result = P.ridge_probe_r2(x, y, g)
        assert result["r2_mean"] > 0.99 and len(result["r2_folds"]) == 5
        assert result["r2_mean"] == pytest.approx(np.mean(result["r2_folds"]))
        assert result["r2_sd"] == pytest.approx(np.std(result["r2_folds"], ddof=1))
        assert result["alpha_median"] in P.ALPHAS
        shuffled = P.ridge_probe_r2(x, np.random.default_rng(0).permutation(y), g)
        assert shuffled["r2_mean"] < 0.05
        assert P.ridge_probe_r2(x, y, g) == result                       # seeded
    assert len(P.ALPHAS) == 13 and P.ALPHAS[0] == 1e-2 and P.ALPHAS[-1] == 1e4
    assert (P.N_FOLDS, P.CV_SEED) == (5, 0)


def test_speaker_folds_remove_the_speaker_leak():
    """A target that is one value per speaker, and features that identify the
    speaker: read with random folds, not with held-out speakers."""
    rng = np.random.default_rng(1)
    n_speakers, per_speaker, dim = 20, 15, 30
    identity = rng.normal(0.0, 1.0, (n_speakers, dim))
    value = rng.normal(0.0, 1.0, n_speakers)
    speaker = np.repeat(np.arange(n_speakers), per_speaker)
    x = identity[speaker] + 0.1 * rng.normal(0.0, 1.0, (speaker.size, dim))
    y = value[speaker] + 0.05 * rng.normal(0.0, 1.0, speaker.size)
    assert P.ridge_probe_r2(x, y)["r2_mean"] > 0.95
    assert P.ridge_probe_r2(x, y, speaker.astype(str))["r2_mean"] < 0.3


def test_padding_and_length_have_the_same_r2():
    """The length inside the window is the window minus the padding."""
    x, y, groups = linear_problem(n=200, seed=2)
    pad = 20_000 + 3_000 * y
    for g in (None, groups):
        a, b = P.ridge_probe_r2(x, pad, g), P.ridge_probe_r2(x, 64_000 - pad, g)
        assert a["r2_folds"] == pytest.approx(b["r2_folds"], abs=1e-9)
        assert a["alpha_median"] == b["alpha_median"]


def probe_inputs(n: int = 120, n_layers: int = 3, dim: int = 6):
    rng = np.random.default_rng(3)
    features = rng.normal(0.0, 1.0, (n, n_layers, dim)).astype(np.float32)
    targets = pd.DataFrame({"audio_id": [f"f{k}" for k in range(n)],
                            "f0_mean_hz": 150 + 20 * features[:, 0, 0] + rng.normal(0, 1, n),
                            "pad": 1_000 * features[:, 2, 1] + rng.normal(0, 50, n)})
    targets.loc[::10, "f0_mean_hz"] = np.nan
    groups = np.array([f"s{k % 8}" for k in range(n)])
    return features, targets, groups


def test_probe_all_layers():
    features, targets, groups = probe_inputs()
    r2 = P.probe_all_layers(features, targets, ("f0_mean_hz", "pad"), groups=groups, progress=False)
    assert list(r2.columns) == ["cv", "layer", "target", "control", "n_files", "n_groups", "r2_mean",
                                "r2_sd", "alpha_median", *[f"r2_fold_{k}" for k in range(1, 6)]]
    assert len(r2) == 2 * 2 * 3 * 2
    assert r2.groupby("target")["n_files"].first().to_dict() == {"f0_mean_hz": 108, "pad": 120}
    assert set(r2["n_groups"]) == {8}
    real = r2[(r2["control"] == "real") & (r2["cv"] == "speaker")].set_index(["target", "layer"])
    assert real.loc[("f0_mean_hz", 1), "r2_mean"] > 0.95 and real.loc[("f0_mean_hz", 2), "r2_mean"] < 0.2
    assert real.loc[("pad", 3), "r2_mean"] > 0.95
    assert (r2[r2["control"] == "shuffled"]["r2_mean"] < 0.2).all()
    # One row, recomputed directly: files without the target are left out and the
    # permutation of the control is that of the seed.
    ok = targets["f0_mean_hz"].notna().to_numpy()
    y = targets["f0_mean_hz"].to_numpy()[ok]
    direct = P.ridge_probe_r2(features[ok][:, 0, :].astype(np.float64),
                              np.random.default_rng(0).permutation(y), groups[ok])
    row = r2[(r2["cv"] == "speaker") & (r2["layer"] == 1) & (r2["target"] == "f0_mean_hz")
             & (r2["control"] == "shuffled")].iloc[0]
    assert row["r2_mean"] == direct["r2_mean"] and row["r2_fold_3"] == direct["r2_folds"][2]
    pd.testing.assert_frame_equal(
        P.probe_all_layers(features, targets, ("f0_mean_hz", "pad"), groups=groups, n_jobs=2,
                           progress=False), r2)
    labelled = P.probe_all_layers(features, targets, ("pad",), cv_schemes=("random",),
                                  layers=(4, 8, 12), progress=False)
    assert sorted(set(labelled["layer"])) == [4, 8, 12] and labelled["n_groups"].isna().all()


def test_probe_all_layers_rejects_inconsistent_input():
    features, targets, groups = probe_inputs()
    with pytest.raises(ValueError, match="speaker ids"):
        P.probe_all_layers(features, targets, ("pad",), progress=False)
    with pytest.raises(KeyError, match="hnr_mean_db"):
        P.probe_all_layers(features, targets, ("pad", "hnr_mean_db"), groups=groups, progress=False)
    with pytest.raises(ValueError, match="different lengths"):
        P.probe_all_layers(features[:-1], targets, ("pad",), groups=groups, progress=False)
    with pytest.raises(ValueError, match="different lengths"):
        P.probe_all_layers(features, targets, ("pad",), groups=groups[:-1], progress=False)
    with pytest.raises(ValueError, match="cv schemes"):
        P.probe_all_layers(features, targets, ("pad",), groups=groups, cv_schemes=("file",),
                           progress=False)
    with pytest.raises(ValueError, match="numbers of layers"):
        P.probe_all_layers(features, targets, ("pad",), groups=groups, layers=(1, 2), progress=False)


def measurements() -> pd.DataFrame:
    return pd.DataFrame({"audio_id": ["a", "b", "c"], "win_lead_40": [8_000, 0, 12_000],
                         "pad": [0, 24_000, 4_000], "pad_trimmed_40": [10_000, 30_000, 4_000],
                         "trim_fallback_40": [0, 0, 1], "pad_trimmed_20": [20_000, 40_000, 9_000],
                         "trim_fallback_20": [0, 1, 0]})


def test_condition_targets_by_hand():
    clean = P.condition_targets(measurements())
    assert list(clean.columns) == ["win_lead_40", "pad", "length_in_window"]
    assert clean.loc["a"].tolist() == [8_000, 0, 64_000] and clean.loc["b"].tolist() == [0, 24_000, 40_000]
    trimmed = P.condition_targets(measurements(), "trim40")
    assert trimmed["pad"].tolist() == [10_000, 30_000, 4_000]
    assert trimmed["length_in_window"].tolist() == [54_000, 34_000, 60_000]
    assert trimmed["win_lead_40"].tolist() == [0, 0, 12_000]         # only the fallback file keeps it
    assert P.condition_targets(measurements(), "trim20")["pad"].tolist() == [20_000, 40_000, 9_000]
    assert P.condition_targets(measurements(), "trim20")["win_lead_40"].tolist() == [0, 0, 0]
    with pytest.raises(KeyError):
        P.condition_targets(measurements(), "trimmed")
    assert P.TARGETS["clean"] == P.ACOUSTIC_TARGETS + ("win_lead_40", "pad", "length_in_window")
    assert P.TARGETS["trim40"] == P.ACOUSTIC_TARGETS + ("pad", "length_in_window")


def test_silence_targets_are_joined_by_audio_id():
    targets = pd.DataFrame({"audio_id": ["c", "a"], "f0_mean_hz": [100.0, 200.0]})
    joined = P.attach_silence_targets(targets, measurements(), "trim40", ("pad", "length_in_window"))
    assert list(joined.columns) == ["audio_id", "f0_mean_hz", "pad", "length_in_window"]
    assert joined["pad"].tolist() == [4_000, 10_000] and "pad" not in targets.columns
    with pytest.raises(KeyError, match="1 sampled files"):
        P.attach_silence_targets(pd.DataFrame({"audio_id": ["a", "z"]}), measurements())


def test_sample_features_from_the_layer_cache(tmp_path):
    feature_dir, cache_dir = str(tmp_path / "features"), str(tmp_path / "cache")
    write_feature_split(feature_dir, "train", n=300, n_layers=4, dim=5, seed=7)
    shards.build_layer_cache(feature_dir, "train", cache_dir)
    ids = ["train_250", "train_3", "train_128", "train_127", "train_3"]
    features, layers = P.load_sample_features(cache_dir, "train", ids)
    assert features.shape == (5, 4, 5) and features.dtype == np.float32 and layers == [1, 2, 3, 4]
    for layer in layers:
        x, _, all_ids = shards.load_layer_from_cache(cache_dir, "train", layer, "mean")
        rows = [all_ids.index(a) for a in ids]
        assert np.array_equal(features[:, layer - 1, :], x.numpy()[rows])
    with pytest.raises(KeyError, match="not in the train layer cache"):
        P.load_sample_features(cache_dir, "train", ["train_1", "dev_1"])


def r2_rows() -> pd.DataFrame:
    rows = []
    for target, base in (("pad", 0.9), ("f0_mean_hz", 0.5)):
        for cv, shift in (("random", 0.05), ("speaker", 0.0)):
            for layer in (1, 4, 8, 12):
                for control in ("real", "shuffled"):
                    rows.append({"cv": cv, "layer": layer, "target": target, "control": control,
                                 "r2_mean": (base + shift - layer / 100) if control == "real" else -0.01,
                                 "r2_sd": 0.0123})
    return pd.DataFrame(rows)


def test_probe_tables():
    table = P.probe_table(r2_rows())
    assert list(table.columns) == ["layer", "f0_mean_hz|random", "f0_mean_hz|speaker", "pad|random",
                                   "pad|speaker"]
    assert table["layer"].tolist() == [1, 4, 8, 12]
    assert table.loc[0, "pad|speaker"] == "0.890 +- 0.012" and table.loc[3, "f0_mean_hz|random"] == "0.430 +- 0.012"
    selected = P.selected_layers_table(r2_rows())
    assert list(selected.columns) == ["target", "cv", "L1", "L4", "L8", "L12"]
    assert selected["target"].tolist() == ["f0_mean_hz", "pad"] and set(selected["cv"]) == {"speaker"}
    assert selected.set_index("target").loc["pad", ["L1", "L12"]].tolist() == pytest.approx([0.89, 0.78])
    random = P.selected_layers_table(r2_rows(), layers=(4,), cv="random")
    assert random["L4"].tolist() == pytest.approx([0.51, 0.91])
    with pytest.raises(KeyError, match=r"\[5\]"):
        P.selected_layers_table(r2_rows(), layers=(1, 5))
    assert P.SELECTED_LAYERS == (1, 4, 8, 12)


def test_run_probes_from_files_to_tables(tmp_path):
    sample = write_voiced_file_list(tmp_path, n_files=20, n_speakers=5)
    wide = S.measure_items(sample, str(tmp_path / "measurements"), "train", progress=False)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    rng = np.random.default_rng(5)
    order = rng.permutation(len(sample))                       # the cache has another file order
    stored_layers = [1, 4, 8, 12]                              # a cache that holds four layers
    np.save(cache_dir / "train_mean.npy", rng.normal(0, 1, (4, len(sample), 4)).astype(np.float32))
    torch.save({"audio_ids": [sample["audio_id"].iloc[i] for i in order],
                "layers": stored_layers}, cache_dir / "train_index.pt")
    for condition, n_targets in (("clean", 6), ("trim40", 5)):
        out_dir = str(tmp_path / "probes" / condition)
        r2 = P.run_probes(sample, str(cache_dir), out_dir, S.silence_table(wide),
                          condition=condition, progress=False)
        assert list(r2.columns[:3]) == ["encoder", "extraction", "condition"]
        assert set(r2["encoder"]) == {"wavlm-base"} and set(r2["condition"]) == {condition}
        assert len(r2) == n_targets * 2 * 4 * 2 and sorted(set(r2["layer"])) == stored_layers
        assert tuple(pd.unique(r2["target"])) == P.TARGETS[condition]
        assert r2[r2["target"] == "pad"]["n_files"].eq(20).all()
        assert r2[r2["target"] == "f0_mean_hz"]["n_files"].lt(20).all()
        assert sorted(os.listdir(out_dir)) == [
            "probe_r2.csv", "probe_r2.meta.json", "probe_r2_selected.csv",
            "probe_r2_selected.meta.json", "probe_r2_table.csv", "probe_r2_table.meta.json",
            "probe_targets.csv", "probe_targets.meta.json", "target_chunks"]
        stored = pd.read_csv(os.path.join(out_dir, "probe_r2.csv"), float_precision="round_trip")
        pd.testing.assert_frame_equal(stored, r2, check_dtype=False)
        meta = json.loads(open(os.path.join(out_dir, "probe_r2.meta.json")).read())
        assert (meta["praat"], meta["condition"], meta["n_sample"]) == ("6.1.38", condition, 20)
        selected = pd.read_csv(os.path.join(out_dir, "probe_r2_selected.csv"))
        assert list(selected.columns) == ["target", "cv", "L1", "L4", "L8", "L12"]
        assert len(selected) == n_targets
        length = r2[(r2["target"] == "length_in_window")].reset_index(drop=True)
        pad = r2[(r2["target"] == "pad")].reset_index(drop=True)
        assert length["r2_mean"].to_numpy() == pytest.approx(pad["r2_mean"].to_numpy(), abs=1e-9)
    with pytest.raises(KeyError):
        P.run_probes(sample, str(cache_dir), str(tmp_path / "x"), S.silence_table(wide),
                     condition="trim20", progress=False)


# Gate A6 ---------------------------------------------------------------------------------------

PROBE_COLUMNS = {name: name for name in ("cv", "layer", "target", "control", "n_files", "n_groups", "r2_mean",
                                         "r2_sd", "alpha_median", *[f"r2_fold_{i}" for i in range(1, 6)])}
PROBE_KEYS = ["condition", "cv", "layer", "target", "control"]


def probe_legacy():
    """The two probe entries of the mapping file, as the real file has them."""
    return {"tables": {
        "probe_r2_clean": {"file": "clean.csv", "columns": PROBE_COLUMNS, "constants": {"condition": "clean"},
                           "keys": PROBE_KEYS},
        "probe_r2_trim40": {"file": "trim40.csv", "columns": PROBE_COLUMNS,
                            "where": {"target": ["f0_mean_hz", "hnr_mean_db", "tilt_db_per_khz", "pad",
                                                 "length_in_window"]},
                            "constants": {"condition": "trim40"}, "keys": PROBE_KEYS}}}


def small_r2(targets=("f0_mean_hz", "pad"), layers=(1, 2)):
    rows = []
    for cv in P.CV_SCHEMES:
        for layer in layers:
            for k, target in enumerate(targets):
                for control, value in (("real", 0.5 + 0.01 * layer + 0.1 * k), ("shuffled", -0.01)):
                    row = {"cv": cv, "layer": layer, "target": target, "control": control, "n_files": 40,
                           "n_groups": 4 if cv == "speaker" else np.nan, "r2_mean": value, "r2_sd": 0.02,
                           "alpha_median": 10.0}
                    row.update({f"r2_fold_{i}": value + 0.001 * i for i in range(1, 6)})
                    rows.append(row)
    return pd.DataFrame(rows)


def test_the_sample_and_its_speakers_are_compared_with_the_original():
    sample = pd.DataFrame({"audio_id": ["a", "b", "c"], "label": [0, 1, 1], "system_id": ["-", "A01", "A02"],
                           "speaker": ["s1", "s2", "s1"], "path": "x"})
    original = sample[["audio_id", "label", "system_id"]]
    row = P.compare_sample(sample, original).iloc[0]
    assert row.to_dict() == {"n_files": 3, "n_original": 3, "same_files_in_order": True, "same_labels": True,
                             "same_attacks": True, "ok": True}
    # Another order, another label, another attack or another number of files is not the same sample.
    assert not P.compare_sample(sample, original.iloc[[1, 0, 2]])["same_files_in_order"].iloc[0]
    assert not P.compare_sample(sample, original.assign(label=[0, 1, 0]))["ok"].iloc[0]
    assert not P.compare_sample(sample, original.assign(system_id=["-", "A01", "A03"]))["same_attacks"].iloc[0]
    assert not P.compare_sample(sample, original.iloc[:2])["ok"].iloc[0]
    speakers = P.speaker_check(sample, expected=2).iloc[0]
    assert (speakers["n_speakers"], speakers["n_without_speaker"], speakers["ok"]) == (2, 0, True)
    assert not P.speaker_check(sample)["ok"].iloc[0] and P.speaker_check(sample)["expected"].iloc[0] == 20
    assert P.N_SPEAKERS == 20
    missing = P.speaker_check(sample.assign(speaker=["s1", None, ""]), expected=1).iloc[0]
    assert (missing["n_without_speaker"], missing["ok"]) == (2, False)
    # The original targets are put in the order of the sample; other files are an error.
    targets = pd.DataFrame({"audio_id": ["c", "a", "b"], "f0_mean_hz": [3.0, 1.0, 2.0]})
    assert P.in_sample_order(targets, sample)["f0_mean_hz"].tolist() == [1.0, 2.0, 3.0]
    with pytest.raises(KeyError, match="same files"):
        P.in_sample_order(targets.iloc[:2], sample)
    with pytest.raises(KeyError, match="same files"):
        P.in_sample_order(pd.concat([targets, targets.iloc[:1]]), sample)


def test_targets_are_compared_and_only_a_target_that_differs_is_probed_again():
    from ssl_silence import verify

    original = pd.DataFrame({"audio_id": ["a", "b", "c"], "label": [0, 1, 1], "system_id": ["-", "A01", "A02"],
                             "f0_mean_hz": [100.0, np.nan, 120.0], "hnr_mean_db": [10.0, 11.0, 12.0],
                             "tilt_db_per_khz": [-1.0, -2.0, -3.0]})
    new = original.copy()
    new.loc[2, "hnr_mean_db"] += 1e-13                         # the last digit: inside the tolerance, not identical
    check = P.compare_targets(new, original, "clean")
    assert list(check.columns[:2]) == ["condition", "column"] and check["ok"].all()
    assert (check["tolerance"] == verify.TARGET_TOLERANCE).all() and verify.TARGET_TOLERANCE == 1e-6
    assert dict(zip(check["column"], check["identical"])) == {
        "label": True, "system_id": True, "f0_mean_hz": True, "hnr_mean_db": False, "tilt_db_per_khz": True}
    rerun = P.rerun_table(check)
    assert rerun["target"].tolist() == list(P.TARGETS["clean"])
    assert dict(zip(rerun["target"], rerun["probed_again"])) == {
        "f0_mean_hz": False, "hnr_mean_db": True, "tilt_db_per_khz": False, "win_lead_40": False, "pad": False,
        "length_in_window": False}
    assert rerun[rerun["target"] == "pad"]["source"].iloc[0].startswith("silence measurements")
    # A difference above the tolerance, a missing value on one side or a missing file fails.
    assert not P.compare_targets(new.assign(tilt_db_per_khz=[-1.0, -2.0, -3.001]), original, "clean")["ok"].all()
    one_side = P.compare_targets(new.assign(f0_mean_hz=[100.0, 110.0, 120.0]), original, "clean")
    assert not one_side.set_index("column").loc["f0_mean_hz", "ok"]
    assert not P.compare_targets(new.iloc[:2], original, "clean")["ok"].any()
    both = pd.concat([check, P.compare_targets(original, original, "trim40")], ignore_index=True)
    table = P.rerun_table(both)
    assert table[table["condition"] == "trim40"]["target"].tolist() == list(P.TARGETS["trim40"])
    assert not table[table["condition"] == "trim40"]["probed_again"].any()


def test_blocks_are_the_rows_of_probe_all_layers_and_a_restart_reads_them(tmp_path, monkeypatch):
    x, y, groups = linear_problem(n=60, dim=4, n_speakers=6)
    features = np.stack([x, x[:, ::-1]], axis=1).astype(np.float32)
    targets = pd.DataFrame({"audio_id": [f"f{k}" for k in range(60)], "f0_mean_hz": y, "pad": -y})
    whole = P.probe_all_layers(features, targets, ["f0_mean_hz", "pad"], groups=groups, layers=[1, 2],
                               progress=False)
    folder = str(tmp_path / "blocks")
    first = P.probe_blocks(features, targets, ["f0_mean_hz", "pad"], groups, [1, 2], folder, "clean__x",
                           progress=False)
    pd.testing.assert_frame_equal(first, whole)                # the same rows, in the same order, to the last digit
    assert sorted(os.listdir(folder)) == sorted(
        f"clean__x__{name}__{cv}.{ext}" for name in ("f0_mean_hz", "pad") for cv in P.CV_SCHEMES
        for ext in ("csv", "meta.json"))
    meta = json.load(open(os.path.join(folder, "clean__x__pad__speaker.meta.json"), encoding="utf-8"))
    assert set(meta) == {"n_files", "layers", "features_sha256", "groups_sha256", "target", "target_sha256", "cv"}
    # A restart computes nothing: finished blocks are read. A block without its meta file is computed again.
    calls = []
    real = P.probe_all_layers
    monkeypatch.setattr(P, "probe_all_layers", lambda *a, **k: calls.append(k["cv_schemes"]) or real(*a, **k))
    again = P.probe_blocks(features, targets, ["f0_mean_hz", "pad"], groups, [1, 2], folder, "clean__x",
                           progress=False)
    pd.testing.assert_frame_equal(again, whole)
    assert calls == []
    os.remove(os.path.join(folder, "clean__x__pad__random.meta.json"))
    P.probe_blocks(features, targets, ["pad"], groups, [1, 2], folder, "clean__x", progress=False)
    assert calls == [["random"]]
    # A block of other targets, other speakers or other features is not read.
    for other in (dict(targets=targets.assign(pad=targets["pad"] + 1e-12)), dict(groups=groups[::-1].copy()),
                  dict(features=features + np.float32(1e-3))):
        arguments = dict(features=features, targets=targets, groups=groups)
        arguments.update(other)
        with pytest.raises(RuntimeError, match="computed from other inputs"):
            P.probe_blocks(arguments["features"], arguments["targets"], ["pad"], arguments["groups"], [1, 2],
                           folder, "clean__x", progress=False)
    # Another tag is another set of blocks.
    P.probe_blocks(features, targets.assign(pad=targets["pad"] + 1e-12), ["pad"], groups, [1, 2], folder,
                   "clean__y", progress=False)
    assert calls[-2:] == [["random"], ["speaker"]]


def test_reruns_replace_their_targets_and_the_conditions_become_one_table():
    first = small_r2(("f0_mean_hz", "pad"))
    again = small_r2(("f0_mean_hz",)).assign(r2_mean=0.9)
    merged = P.merge_reruns(first, again, ["f0_mean_hz", "pad"])
    assert len(merged) == len(first) and list(dict.fromkeys(merged["target"])) == ["f0_mean_hz", "pad"]
    assert (merged[merged["target"] == "f0_mean_hz"]["r2_mean"] == 0.9).all()
    pd.testing.assert_frame_equal(merged[merged["target"] == "pad"].reset_index(drop=True),
                                  first[first["target"] == "pad"].reset_index(drop=True))
    pd.testing.assert_frame_equal(P.merge_reruns(first, None, ["f0_mean_hz", "pad"]),
                                  pd.concat([first[first["target"] == t] for t in ("f0_mean_hz", "pad")],
                                            ignore_index=True))
    table = P.assemble({"clean": first, "trim40": merged}, "wavlm-base")
    assert list(table.columns[:3]) == ["encoder", "extraction", "condition"] and len(table) == 2 * len(first)
    assert table["condition"].tolist() == ["clean"] * len(first) + ["trim40"] * len(first)
    assert set(table["extraction"]) == {"unpadded"} and set(table["encoder"]) == {"wavlm-base"}
    assert list(P.assemble({"clean": table[table["condition"] == "clean"]}).columns) == list(table.columns)


def test_r2_tables_are_compared_with_the_decided_tolerance(tmp_path):
    names = list(P.TARGETS["trim40"])
    new = small_r2(names)
    folder = tmp_path / "original" / "tables"
    folder.mkdir(parents=True)
    others = small_r2(("win_lead_40", "win_trail_40", "speech_fraction", "length_residual"))
    old = pd.concat([new.assign(feature_set="x"), others.assign(feature_set="x")], ignore_index=True)
    old.to_csv(folder / "probe_r2_trim40.csv", index=False)
    legacy, copies = probe_legacy(), str(tmp_path / "original")
    check = P.compare_r2(new, "trim40", legacy, copies)
    compared = check[check["note"] == ""]
    assert compared["column"].tolist() == [c for c in PROBE_COLUMNS if c not in PROBE_KEYS]
    assert compared["ok"].all() and compared["identical"].all() and P.largest_difference(check) == 0.0
    by_column = compared.set_index("column")["tolerance"]
    assert P.R2_TOLERANCE == 1e-9 and (by_column[list(P.R2_COLUMNS)] == 1e-9).all()
    assert (by_column[list(P.EXACT_COLUMNS)] == 0.0).all()
    # The four targets that only the original table has are listed, not compared.
    listed = check[check["note"] != ""]
    assert listed["column"].tolist() == ["target win_lead_40", "target win_trail_40", "target speech_fraction",
                                         "target length_residual"]
    assert set(listed["note"]) == {"not ported, not in the manuscript"} and listed["ok"].all()
    assert listed["n_old"].tolist() == [8, 8, 8, 8]
    # The last digit of an R2 is inside the tolerance and is reported; more is not.
    close = P.compare_r2(new.assign(r2_mean=new["r2_mean"] + 2e-12), "trim40", legacy, copies)
    row = close.set_index("column").loc["r2_mean"]
    assert row["ok"] and not row["identical"] and 1e-12 < P.largest_difference(close) < 1e-11
    far = P.compare_r2(new.assign(r2_fold_3=new["r2_fold_3"] + 2e-9), "trim40", legacy, copies)
    assert far[~far["ok"]]["column"].tolist() == ["r2_fold_3"]
    # The median alpha and the counts must be equal.
    for column, value in (("alpha_median", 10.0 + 1e-12), ("n_files", 41), ("n_groups", 5.0)):
        other = P.compare_r2(new.assign(**{column: value}), "trim40", legacy, copies)
        assert other[~other["ok"]]["column"].tolist() == [column], column
    # A row that only one table has fails every column.
    assert not P.compare_r2(new.iloc[1:], "trim40", legacy, copies).query("note == ''")["ok"].any()
    assert P.ORIGINAL_TABLES == {"clean": "probe_r2_clean", "trim40": "probe_r2_trim40"}


def test_the_cells_of_the_probe_table_are_built_from_both_tables():
    from ssl_silence.paper import tables as T

    r2 = small_r2([name for name, _ in T.PROBE_TARGETS], T.PROBE_LAYERS)
    check = P.selected_cells_check(P.assemble({"clean": r2}), r2)
    assert len(check) == len(T.PROBE_TARGETS) * len(T.PROBE_LAYERS) and check["ok"].all()
    assert check["layer"].iloc[:4].tolist() == ["L1", "L4", "L8", "L12"] and check["cell"].iloc[0] == "0.51"
    moved = r2.copy()
    moved.loc[(moved["cv"] == "speaker") & (moved["control"] == "real") & (moved["layer"] == 4)
              & (moved["target"] == "hnr_mean_db"), "r2_mean"] += 0.01
    other = P.selected_cells_check(P.assemble({"clean": moved}), r2)
    assert other[~other["ok"]][["target", "layer"]].values.tolist() == [["Mean HNR", "L4"]]
    # A cell that cannot be built is not a pass.
    assert not P.selected_cells_check(P.assemble({"clean": r2[r2["layer"] != 8]}), r2)["ok"].all()


def test_the_timed_check_scales_to_the_probes_of_the_run():
    x, y, groups = linear_problem(n=60, dim=4, n_speakers=6)
    features = np.stack([x, x], axis=1).astype(np.float32)
    targets = pd.DataFrame({"f0_mean_hz": y})
    timing = P.time_probes(features, targets, "f0_mean_hz", groups, n_probes=240, n_jobs=1)
    assert timing["cv"].tolist() == list(P.CV_SCHEMES) and timing["probes_timed"].tolist() == [1, 1]
    assert (timing["seconds"] > 0).all() and timing["n_probes"].tolist() == [240, 240]
    assert np.allclose(timing["estimated_minutes"], timing["seconds"] * 240 / 60)
