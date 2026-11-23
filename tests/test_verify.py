"""Comparisons with original outputs through a mapping file, with known answers.

The "original" outputs of these tests are invented files with invented
names; the tests against the outputs of the original code are in
test_parity_verify.py.
"""

from __future__ import annotations

import json
import os
import shutil

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

import world
from ssl_silence import provenance, shards
from ssl_silence import training as T
from ssl_silence import verify as V
from synthetic import write_feature_split

OLD_NAMES = {"clean": "plain", "trim40": "cut", "trim20": "cut20"}
FORBIDDEN = {"values": ["/drive/old_root/results", "old-feature-set"],
             "patterns": [r"old_[a-z-]+_L\d\d_(mean|stats)_\w+_s\d+"]}


def mapping(root, **parts) -> dict:
    return {"root": str(root), "forbidden": FORBIDDEN, "conditions": dict(OLD_NAMES), **parts}


# Mapping file ----------------------------------------------------------------------------

def test_mapping_file_is_checked(tmp_path):
    good = mapping(tmp_path)
    path = tmp_path / "map.yaml"
    path.write_text(yaml.safe_dump(good), encoding="utf-8")
    assert V.load_map(str(path)) == good
    assert V.legacy_path(good, "runs/main") == os.path.join(str(tmp_path), "runs", "main")
    with pytest.raises(KeyError, match="forbidden"):
        V.check_map({k: v for k, v in good.items() if k != "forbidden"})
    with pytest.raises(ValueError, match="empty values"):
        V.check_map(mapping(tmp_path, features={"wavlm-base": {"unpadded": {"clean": ""}}}))
    with pytest.raises(ValueError, match="empty values"):
        V.check_map(mapping(tmp_path, runs={"folders": ["a", None]}))
    with pytest.raises(KeyError, match="unknown conditions"):
        V.check_map({**good, "conditions": {"trimmed": "cut"}})
    with pytest.raises(Exception):
        V.check_map({**good, "forbidden": {"values": [], "patterns": ["("]}})
    with pytest.raises(ValueError):
        V.check_map({**good, "forbidden": {"values": [], "names": []}})


# Check tables ------------------------------------------------------------------------------

def test_a_check_table_with_kept_column_names_is_clean(tmp_path):
    """Whole values are compared: the silence columns that this package keeps,
    and words that are part of an original name, are allowed."""
    table = pd.DataFrame({"trimmed_len_40": [1, 2], "pad_trimmed_40": [0, 3],
                          "condition": ["trim40", "clean"],
                          "note": ["old-feature-set is not a whole value here", "cut"],
                          "ok": [True, True]})
    V.assert_clean(table, FORBIDDEN)
    path = V.write_check(table, "kept_names", str(tmp_path), FORBIDDEN, tolerance=0.0)
    assert os.path.basename(path) == "kept_names.csv"
    assert pd.read_csv(path)["trimmed_len_40"].tolist() == [1, 2]
    meta = json.loads((tmp_path / "kept_names.meta.json").read_text())
    assert meta["check"] == "kept_names" and meta["rows"] == 2 and meta["tolerance"] == 0.0


def test_a_check_table_with_an_original_run_id_or_path_is_refused(tmp_path):
    with_run_id = pd.DataFrame({"run_id": ["wavlm-base_unpadded_clean_L04_stats_s42",
                                           "old_wavlm-base_L04_stats_plain_s42"], "ok": True})
    with pytest.raises(ValueError, match="original names"):
        V.assert_clean(with_run_id, FORBIDDEN)
    with_path = pd.DataFrame({"file": ["features/train", "/drive/old_root/results"], "ok": True})
    with pytest.raises(ValueError, match="original names"):
        V.write_check(with_path, "bad", str(tmp_path), FORBIDDEN)
    assert not os.path.exists(tmp_path / "bad.csv")
    with pytest.raises(ValueError, match="column 'old-feature-set'"):
        V.assert_clean(pd.DataFrame({"old-feature-set": [1]}), FORBIDDEN)
    # A longer value that contains a forbidden one is not a match.
    V.assert_clean(pd.DataFrame({"file": ["/drive/old_root/results/x.csv"]}), FORBIDDEN)
    V.assert_clean(with_run_id, {"values": [], "patterns": []})


def test_require_ok_and_summary():
    good = pd.DataFrame({"key": ["a", "b"], "max_abs_diff": [0.0, 2e-5], "ok": [True, True]})
    bad = pd.DataFrame({"key": ["a", "b"], "max_abs_diff": [0.0, 0.3], "ok": [True, False]})
    V.require_ok(good, "good")
    with pytest.raises(V.CheckFailed, match="1 of 2 rows failed"):
        V.require_ok(bad, "bad")
    with pytest.raises(V.CheckFailed, match="nothing was compared"):
        V.require_ok(good.iloc[:0], "empty")
    with pytest.raises(KeyError):
        V.failed_rows(good.drop(columns="ok"))
    summary = V.summarise_checks({"good": good, "bad": bad, "empty": good.iloc[:0]})
    assert summary["check"].tolist() == ["good", "bad", "empty"]
    assert summary["n_failed"].tolist() == [0, 1, 0] and summary["ok"].tolist() == [True, False, False]
    assert summary["max_abs_diff"].tolist()[:2] == [2e-5, 0.3]


# Files ---------------------------------------------------------------------------------------

def test_files_are_compared_by_sha256_at_the_start_and_at_the_end(tmp_path):
    tables, checks = tmp_path / "tables", tmp_path / "checks"
    (tables / "parts").mkdir(parents=True)
    checks.mkdir()
    for file in (tables / "a.csv", tables / "a.meta.json", tables / "parts" / "a_trim40.csv",
                 tables / "own.csv", checks / "table_a.csv", checks / "own_steps.csv"):
        file.write_text(file.name, encoding="utf-8")
    folders = [str(tables), str(checks)]
    before = V.file_hashes(folders, skip=("own",))
    assert [os.path.relpath(file, tmp_path).replace(os.sep, "/") for file in before] == [
        "checks/table_a.csv", "tables/a.csv", "tables/a.meta.json", "tables/parts/a_trim40.csv"]
    assert before[str(tables / "a.csv")] == provenance.sha256_file(tables / "a.csv")
    assert len(V.file_hashes(folders)) == 6 and V.file_hashes([str(tmp_path / "none")]) == {}

    # The files of the run itself are written; nothing else changes.
    (tables / "own.csv").write_text("again", encoding="utf-8")
    (tables / "own_2.csv").write_text("new", encoding="utf-8")
    same = V.unchanged_check(before, folders, skip=("own",))
    assert list(same.columns) == ["file", "sha256_at_start", "sha256_at_end", "status", "ok"]
    assert same["file"].tolist() == list(before) and same["ok"].all()
    assert (same["sha256_at_start"] == same["sha256_at_end"]).all()
    V.require_ok(same, "unchanged")

    # The same text with another line ending, a removed file and a new one: each is a failing row.
    (tables / "a.csv").write_bytes(b"a.csv\r\n")
    (tables / "a.meta.json").unlink()
    (checks / "table_b.csv").write_text("b", encoding="utf-8")
    table = V.unchanged_check(before, folders, skip=("own",)).set_index(
        "file").rename(index=lambda file: os.path.basename(file))
    assert table["status"].to_dict() == {"table_a.csv": "unchanged", "table_b.csv": "new", "a.csv": "changed",
                                         "a.meta.json": "removed", "a_trim40.csv": "unchanged"}
    assert table["ok"].tolist() == [True, False, False, False, True]
    assert table.loc["table_b.csv", "sha256_at_start"] == "" and table.loc["a.meta.json", "sha256_at_end"] == ""
    with pytest.raises(V.CheckFailed, match="3 of 5 rows failed"):
        V.require_ok(table.reset_index(), "unchanged")


def test_copy_is_verified_by_sha256(tmp_path, monkeypatch):
    source = tmp_path / "a.bin"
    source.write_bytes(os.urandom(4096))
    target = str(tmp_path / "copy" / "a.bin")
    info = V.copy_verified(str(source), target)
    assert info == {"bytes": 4096, "sha256": V.sha256_file(str(source))}
    assert open(target, "rb").read() == source.read_bytes()
    stamp = os.path.getmtime(target)
    assert V.copy_verified(str(source), target) == info and os.path.getmtime(target) == stamp
    open(target, "wb").write(b"other")                          # a wrong copy is replaced
    assert V.copy_verified(str(source), target) == info

    def corrupt(src, dst):
        open(dst, "wb").write(b"corrupt")

    monkeypatch.setattr(shutil, "copyfile", corrupt)
    with pytest.raises(OSError, match="differs from its source"):
        V.copy_verified(str(source), str(tmp_path / "copy" / "b.bin"))


# Tensors and shards ------------------------------------------------------------------------------

def test_compare_tensors():
    a = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    same = V.compare_tensors(a, a.clone())
    assert same == {"n": 3, "same_shape": True, "same_dtype": True, "identical": True,
                    "max_abs_diff": 0.0}
    changed = a.clone()
    changed[1, 2] = float(np.nextafter(np.float32(6.0), np.float32(7.0)))   # one bit
    one_bit = V.compare_tensors(a, changed)
    assert not one_bit["identical"] and 0 < one_bit["max_abs_diff"] < 1e-6
    assert not V.compare_tensors(a, a.double())["identical"]            # another type
    assert V.compare_tensors(a, a.double())["max_abs_diff"] == 0.0
    other_shape = V.compare_tensors(a, a[:2])
    assert not other_shape["same_shape"] and np.isnan(other_shape["max_abs_diff"])
    assert V.compare_tensors(["a", "b"], np.array(["a", "b"]))["identical"]
    assert not V.compare_tensors(["a", "b"], ["b", "a"])["identical"]
    nan = torch.tensor([1.0, float("nan")])
    assert V.compare_tensors(nan, nan.clone())["identical"]
    flags = torch.tensor([True, False])
    assert V.compare_tensors(flags, flags.clone())["identical"]
    assert not V.compare_tensors(flags, ~flags)["identical"]


def test_feature_folders_are_compared_key_by_key(tmp_path):
    new_dir, old_dir = str(tmp_path / "new"), str(tmp_path / "old")
    write_feature_split(new_dir, "train", n=300, n_layers=3, dim=4, seed=1)
    shutil.copytree(new_dir, old_dir)
    table = V.compare_feature_dirs(new_dir, old_dir, splits=("train",),
                                   expected_sizes={"train": 300})
    assert list(table["key"]) == ["mean", "std", "n_real", "n_frames", "trim_fallback", "too_short",
                                  "labels", "audio_ids", "system_ids"]
    assert table["ok"].all() and set(table["n_files"]) == {300} and set(table["n_shards"]) == {3}
    assert not V.compare_feature_dirs(new_dir, old_dir, splits=("train",),
                                      expected_sizes={"train": 301})["ok"].any()
    path = os.path.join(old_dir, "train_shard_0001.pt")
    shard = torch.load(path, weights_only=False)
    shard["std"][5, 1, 2] += 1e-3
    shard["audio_ids"][0] = "another"
    torch.save(shard, path)
    table = V.compare_feature_dirs(new_dir, old_dir, splits=("train",),
                                   expected_sizes={"train": 300}).set_index("key")
    assert table["ok"].to_dict() == {"mean": True, "std": False, "n_real": True, "n_frames": True,
                                     "trim_fallback": True, "too_short": True, "labels": True,
                                     "audio_ids": False, "system_ids": True}
    assert table.loc["std", "max_abs_diff"] == pytest.approx(1e-3, rel=1e-3)
    with pytest.raises(V.CheckFailed):
        V.require_ok(table.reset_index(), "features")
    os.remove(path)
    with pytest.raises(ValueError, match="shards against"):
        V.compare_feature_dirs(new_dir, old_dir, splits=("train",), expected_sizes={"train": 300})
    with pytest.raises(FileNotFoundError):
        V.compare_feature_dirs(new_dir, old_dir, splits=("dev",))


def test_one_layer_of_a_padded_shard_against_the_original_tensor():
    rng = np.random.default_rng(0)
    old = {"plain_mean": torch.from_numpy(rng.normal(size=(6, 12, 5)).astype(np.float32)),
           "frames": torch.arange(6)}
    new = {"all_frames_mean": old["plain_mean"][:, 3:4, :].clone(), "n_frames": torch.arange(6),
           "layers": [4]}
    keys = {"all_frames_mean": "plain_mean", "n_frames": "frames"}
    rows = V.compare_shards(new, old, keys, old_layer=4)
    assert [r["key"] for r in rows] == ["all_frames_mean", "n_frames"] and all(r["identical"] for r in rows)
    wrong = dict(new, all_frames_mean=old["plain_mean"][:, 4:5, :].clone())
    assert not V.compare_shards(wrong, old, keys, old_layer=4)[0]["identical"]
    with pytest.raises(ValueError, match="stores layers"):
        V.compare_shards(dict(new, layers=[3]), old, keys, old_layer=4)
    with pytest.raises(KeyError):
        V.compare_shards(new, old, {"real_frames_mean": "plain_mean"}, old_layer=4)


def test_reextraction_tolerance_is_a_closed_bound():
    stored = {"mean": np.zeros((4, 2, 3)), "n_real": np.array([10, 20, 30, 40]),
              "trim_fallback": np.array([False, True, False, False])}
    at_bound = dict(stored, mean=np.full((4, 2, 3), 1e-4))
    table = V.compare_reextraction(at_bound, stored, ["mean"], ["n_real", "trim_fallback"])
    assert table["ok"].all() and table.set_index("key").loc["mean", "max_abs_diff"] == 1e-4
    assert table["exactly_zero"].tolist() == [False, True, True]
    assert table["tolerance"].tolist() == [1e-4, 0.0, 0.0]
    above = dict(stored, mean=np.full((4, 2, 3), np.nextafter(1e-4, 1.0)))
    assert V.compare_reextraction(above, stored, ["mean"], [])["ok"].tolist() == [False]
    one_length = dict(stored, n_real=np.array([10, 20, 30, 41]))
    table = V.compare_reextraction(one_length, stored, ["mean"], ["n_real"]).set_index("key")
    assert table["ok"].to_dict() == {"mean": True, "n_real": False}
    assert V.FEATURE_TOLERANCE == V.SCORE_TOLERANCE == 1e-4 and V.TABLE_TOLERANCE == 1e-9


# Scores ---------------------------------------------------------------------------------------------

def score_set(scores, labels=(0, 1, 1), ids=("a", "b", "c")):
    return T.ScoreSet(np.asarray(scores, dtype=np.float32), np.asarray(labels, dtype=np.int64),
                      np.asarray(ids))


def test_score_sets_side_by_side():
    base = score_set([0.5, 0.25, 0.75])
    same = V.compare_score_sets(base, score_set([0.5, 0.25, 0.75]))
    assert same == {"n": 3, "ids_equal": True, "labels_equal": True, "max_abs_diff": 0.0,
                    "exactly_zero": True, "within_tolerance": True}
    close = V.compare_score_sets(base, score_set([0.5 + 2.0 ** -14, 0.25, 0.75]))
    assert close["max_abs_diff"] == 2.0 ** -14 and close["within_tolerance"] and not close["exactly_zero"]
    far = V.compare_score_sets(base, score_set([0.5 + 2.0 ** -13, 0.25, 0.75]))
    assert far["max_abs_diff"] == 2.0 ** -13 > 1e-4 and not far["within_tolerance"]
    # The bound itself passes.
    assert V.compare_score_sets(base, score_set([0.5 + 2.0 ** -13, 0.25, 0.75]),
                                tolerance=2.0 ** -13)["within_tolerance"]
    reordered = V.compare_score_sets(base, score_set([0.5, 0.75, 0.25], ids=("a", "c", "b")))
    assert not reordered["ids_equal"] and np.isnan(reordered["max_abs_diff"])
    assert not reordered["within_tolerance"]
    relabelled = V.compare_score_sets(base, score_set([0.5, 0.25, 0.75], labels=(0, 0, 1)))
    assert not relabelled["labels_equal"] and not relabelled["within_tolerance"]
    assert not V.compare_score_sets(base, score_set([0.5, 0.25], (0, 1), ("a", "b")))["ids_equal"]


RUNS_SPEC = {"run_id": "old_{encoder}_L{layer:02d}_{pooling}_{condition}{cross}_s{seed}",
             "cross": "_on-{test_condition}", "folders": ["runs/main", "runs/strict"],
             "matched_sets": {"dev": "dev", "eval": "eval", "itw": "itw"},
             "cross_sets": {"dev": "dev_other", "eval": "eval", "itw": "itw"}}


def write_invented_originals(root) -> dict:
    """The scores of the synthetic runs in an invented two-file layout: one file
    per run scored in its own condition and one per other test condition."""
    rows = {"runs/main": [], "runs/strict": []}
    for condition in world.CONDITIONS:
        for layer in world.LAYERS:
            for pooling in world.POOLINGS:
                for seed in world.SEEDS:
                    for test_condition in world.test_conditions(condition, layer):
                        matched = test_condition == condition
                        cross = "" if matched else f"_on-{OLD_NAMES[test_condition]}"
                        run_id = (f"old_{world.ENCODER}_L{layer:02d}_{pooling}_"
                                  f"{OLD_NAMES[condition]}{cross}_s{seed}")
                        names = RUNS_SPEC["matched_sets" if matched else "cross_sets"]
                        sets = {names[split]: T.ScoreSet(
                            world.scores_of(condition, test_condition, pooling, layer, seed, split),
                            world.labels_of(split), world.ids_of(split)) for split in world.SPLITS}
                        folder = "runs/strict" if "trim20" in (condition, test_condition) else "runs/main"
                        T.save_scores(os.path.join(root, folder, "scores", f"{run_id}.npz"), sets, {})
                        rows[folder].append({"run_id": run_id, "best_epoch": 1})
    for folder, table in rows.items():
        pd.DataFrame(table).to_csv(os.path.join(root, folder, "runs.csv"), index=False)
    return mapping(root, runs={world.ENCODER: {"unpadded": RUNS_SPEC}})


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    base = tmp_path_factory.mktemp("verify")
    runs_dir = world.write_runs(str(base / "new"))
    return runs_dir, write_invented_originals(str(base / "old")), str(base / "old")


def test_one_model_is_compared_with_the_two_original_files(runs):
    runs_dir, legacy, _ = runs
    run = T.read_runs(runs_dir).set_index("run_id", drop=False).loc[
        "wavlm-base_unpadded_clean_L03_stats_s42"].to_dict()
    assert V.legacy_run(legacy, run, "clean") == ("old_wavlm-base_L03_stats_plain_s42", True)
    assert V.legacy_run(legacy, run, "trim40") == ("old_wavlm-base_L03_stats_plain_on-cut_s42", False)
    rows = pd.DataFrame(V.compare_run(run, runs_dir, legacy))
    assert sorted(rows["score_set"]) == sorted([
        "dev@clean", "eval@clean", "itw@clean", "dev@trim40", "eval@trim40", "itw@trim40",
        "dev@trim20", "eval@trim20", "itw@trim20"])
    assert rows["ok"].all() and rows["exactly_zero"].all() and rows["best_epoch_equal"].all()
    assert rows.set_index("score_set").loc["itw@trim20", "n"] == world.SPLITS["itw"]


def test_all_runs_and_what_a_difference_looks_like(runs, tmp_path):
    runs_dir, legacy, old_root = runs
    table = V.compare_runs(runs_dir, legacy)
    n_sets = sum(3 * len(world.test_conditions(c, layer)) for c in world.CONDITIONS
                 for layer in world.LAYERS) * len(world.POOLINGS) * len(world.SEEDS)
    assert len(table) == n_sets and table["ok"].all() and (table["max_abs_diff"] == 0).all()
    V.require_ok(table, "scores")
    V.write_check(table, "scores", str(tmp_path), legacy["forbidden"])       # holds no original name

    one = "wavlm-base_unpadded_trim40_L02_mean_s43"
    assert len(V.compare_runs(runs_dir, legacy, run_ids=[one])) == 6
    with pytest.raises(KeyError):
        V.compare_runs(runs_dir, legacy, run_ids=["wavlm-base_unpadded_trim40_L09_mean_s43"])

    copy = str(tmp_path / "old")
    shutil.copytree(old_root, copy)
    changed = dict(legacy, root=copy)
    path = os.path.join(copy, "runs", "main", "scores", "old_wavlm-base_L02_mean_cut_on-plain_s43.npz")
    sets, _ = T.load_scores(path)
    sets["eval"].scores[7] += np.float32(0.01)
    T.save_scores(path, sets, {})
    table = V.compare_runs(runs_dir, changed, run_ids=[one]).set_index("score_set")
    assert table["ok"].to_dict() == {"dev@trim40": True, "eval@trim40": True, "itw@trim40": True,
                                     "dev@clean": True, "eval@clean": False, "itw@clean": True}
    assert table.loc["eval@clean", "max_abs_diff"] == pytest.approx(0.01, rel=1e-4)

    runs_csv = os.path.join(copy, "runs", "main", "runs.csv")
    old_runs = pd.read_csv(runs_csv)
    old_runs.loc[old_runs["run_id"] == "old_wavlm-base_L02_mean_cut_s43", "best_epoch"] = 6
    old_runs.to_csv(runs_csv, index=False)
    table = V.compare_runs(runs_dir, changed, run_ids=[one]).set_index("score_set")
    assert not table.loc["dev@trim40", "best_epoch_equal"] and not table.loc["dev@trim40", "ok"]
    assert table.loc["dev@trim40", "exactly_zero"] and table.loc["dev@clean", "best_epoch_equal"]

    os.remove(path)
    with pytest.raises(FileNotFoundError, match="no original score file"):
        V.compare_runs(runs_dir, changed, run_ids=[one])


# Tables ------------------------------------------------------------------------------------------------

def original_effects() -> pd.DataFrame:
    """A table in an invented original layout: one row per effect label and its qualifier."""
    return pd.DataFrame({
        "set": ["eval", "eval", "eval", "itw"], "pool": ["stats"] * 4, "L": [3, 3, 3, 3],
        "effect": ["test effect", "interaction", "test effect", "test effect"],
        "fixed": ["train=plain", np.nan, "train=cut", "train=plain"],
        "observed": [6.5, -5.0, 0.5, -0.25], "low": [6.0, -6.0, 0.1, -1.0],
        "sig": [True, True, True, False], "kind": ["a", "a", "a", "b"],
        "files": [100, 100, 100, 200]})


EFFECTS_SPEC = {
    "where": {"kind": ["a", "b"]},
    "combine": {"effect": {"columns": ["effect", "fixed"],
                           "values": {"test effect|train=plain": "delta_test",
                                      "interaction|": "interaction"}}},
    "columns": {"test_set": "set", "pooling": "pool", "layer": "L", "estimate": "observed",
                "spk_ci_low": "low", "spk_excludes_zero": "sig", "n_files": "files"},
    "constants": {"condition": "trim40"},
    "keys": ["condition", "test_set", "pooling", "layer", "effect"],
}


def test_an_original_table_is_renamed_through_the_specification():
    renamed = V.rename_legacy(original_effects(), EFFECTS_SPEC)
    assert list(renamed.columns) == ["effect", "test_set", "pooling", "layer", "estimate",
                                     "spk_ci_low", "spk_excludes_zero", "n_files", "condition"]
    assert renamed["effect"].tolist() == ["delta_test", "interaction", "delta_test"]   # one label unmapped
    assert renamed["estimate"].tolist() == [6.5, -5.0, -0.25] and set(renamed["condition"]) == {"trim40"}
    only_a = V.rename_legacy(original_effects(), {**EFFECTS_SPEC, "where": {"kind": "a"}})
    assert len(only_a) == 2
    labelled = V.rename_legacy(original_effects(), {"columns": {"test_set": "set"},
                                                   "values": {"test_set": {"eval": "evaluation"}}})
    assert labelled["test_set"].tolist() == ["evaluation"] * 3 + ["itw"]
    # A column whose original name was wrong is mapped like any other.
    swapped = V.rename_legacy(pd.DataFrame({"n_bonafide": [5], "n_spoof": [7]}),
                              {"columns": {"n_spoof": "n_bonafide"}})
    assert swapped.to_dict("records") == [{"n_spoof": 5}]
    with pytest.raises(KeyError):
        V.rename_legacy(original_effects(), {"columns": {"estimate": "value"}})


def new_effects() -> pd.DataFrame:
    return pd.DataFrame({"condition": "trim40", "test_set": ["eval", "eval", "itw"],
                         "pooling": "stats", "layer": [3, 3, 3],
                         "effect": ["delta_test", "interaction", "delta_test"],
                         "estimate": [6.5, -5.0, -0.25], "spk_ci_low": [6.0, -6.0, -1.0],
                         "spk_excludes_zero": [True, True, False], "n_files": [100, 100, 200],
                         "extra": [1, 2, 3]})


def test_tables_are_compared_row_by_row_on_their_keys():
    old = V.rename_legacy(original_effects(), EFFECTS_SPEC)
    keys = EFFECTS_SPEC["keys"]
    table = V.compare_tables(new_effects().iloc[::-1], old, keys)        # the row order is free
    assert table["column"].tolist() == ["estimate", "spk_ci_low", "spk_excludes_zero", "n_files"]
    assert table["ok"].all() and (table["max_abs_diff"] == 0).all()
    assert table[["n_new", "n_old", "n_matched", "n_only_new", "n_only_old"]].iloc[0].tolist() == [
        3, 3, 3, 0, 0]

    at_bound = new_effects().assign(estimate=[6.5, -5.0, -0.25 + 2.0 ** -30])
    row = V.compare_tables(at_bound, old, keys, tolerance=2.0 ** -30).set_index("column").loc["estimate"]
    assert row["max_abs_diff"] == 2.0 ** -30 and row["ok"]
    row = V.compare_tables(at_bound, old, keys).set_index("column").loc["estimate"]
    assert row["n_different"] == 0 and row["ok"]                         # 9.3e-10 is within 1e-9
    off = new_effects().assign(estimate=[6.5, -5.0, -0.25 + 2e-9])
    row = V.compare_tables(off, old, keys).set_index("column").loc["estimate"]
    assert row["n_different"] == 1 and not row["ok"] and row["max_abs_diff"] == pytest.approx(2e-9)

    flag = new_effects().assign(spk_excludes_zero=[True, False, False])
    result = V.compare_tables(flag, old, keys).set_index("column")
    assert not result.loc["spk_excludes_zero", "ok"] and result.loc["estimate", "ok"]

    missing = V.compare_tables(new_effects().iloc[:2], old, keys)
    assert not missing["ok"].any() and set(missing["n_only_old"]) == {1}
    extra = V.compare_tables(pd.concat([new_effects(), new_effects().assign(layer=4)]), old, keys)
    assert not extra["ok"].any() and set(extra["n_only_new"]) == {3}

    nan_both = V.compare_tables(new_effects().assign(spk_ci_low=np.nan),
                                old.assign(spk_ci_low=np.nan), keys).set_index("column")
    assert nan_both.loc["spk_ci_low", "ok"]
    nan_one = V.compare_tables(new_effects().assign(spk_ci_low=[6.0, np.nan, -1.0]), old,
                               keys).set_index("column")
    assert not nan_one.loc["spk_ci_low", "ok"] and nan_one.loc["spk_ci_low", "max_abs_diff"] == np.inf

    text = V.compare_tables(new_effects().assign(note=["a", "b", "c"]),
                            old.assign(note=["a", "x", "c"]), keys, columns=["note"]).iloc[0]
    assert text["n_different"] == 1 and np.isnan(text["max_abs_diff"]) and not text["ok"]
    with pytest.raises(ValueError, match="repeated keys"):
        V.compare_tables(pd.concat([new_effects(), new_effects()]), old, keys)
    with pytest.raises(KeyError):
        V.compare_tables(new_effects().drop(columns="n_files"), old, keys)


def test_a_table_is_compared_with_the_original_file_of_the_mapping(tmp_path):
    original_effects().to_csv(tmp_path / "effects_original.csv", index=False)
    legacy = mapping(tmp_path, tables={"effects_2x2": {"file": "effects_original.csv",
                                                       **EFFECTS_SPEC}})
    V.check_map(legacy)
    table = V.compare_with_legacy(new_effects(), "effects_2x2", legacy)
    assert table["table"].tolist() == ["effects_2x2"] * 4 and table["ok"].all()
    assert set(table["tolerance"]) == {1e-9}
    V.write_check(table, "effects_2x2", str(tmp_path / "checks"), legacy["forbidden"])
    loose = dict(legacy, tables={"effects_2x2": {**legacy["tables"]["effects_2x2"], "tolerance": 0.1}})
    off = new_effects().assign(estimate=[6.55, -5.0, -0.25])
    assert V.compare_with_legacy(off, "effects_2x2", loose)["ok"].all()
    assert not V.compare_with_legacy(off, "effects_2x2", legacy)["ok"].all()
    with pytest.raises(KeyError):
        V.compare_with_legacy(new_effects(), "rank_agreement", legacy)
