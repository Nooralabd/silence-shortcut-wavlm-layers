"""The classifier, the training function, run records and the run loop."""

from __future__ import annotations

import collections
import os

import numpy as np
import pandas as pd
import pytest
import torch

from synthetic import SPLIT_SIZES, gaussian_features, write_feature_condition
from ssl_silence import config as cfg
from ssl_silence import shards, training as T
from ssl_silence.classifier import Classifier, count_parameters

SPEC = T.RunSpec("wavlm-base", "unpadded", "clean", 4, "stats", 42)


@pytest.fixture(scope="module")
def data():
    return {"train": gaussian_features(420, 16, 1), "dev": gaussian_features(200, 16, 2),
            "eval": gaussian_features(300, 16, 3), "other": gaussian_features(260, 16, 4, shift=0.6)}


def run(data, spec=SPEC, **kwargs):
    train_x, train_y, _ = data["train"]
    dev_x, dev_y, dev_ids = data["dev"]
    sets = {"eval@clean": data["eval"], "eval@trim40": data["other"]}
    return T.train_and_score(train_x, train_y, dev_x, dev_y, sets, spec, dev_ids=dev_ids,
                             device="cpu", **kwargs)


# Classifier ------------------------------------------------------------------------

def test_parameter_counts():
    assert count_parameters(Classifier(768)) == 98_690
    assert count_parameters(Classifier(1536)) == 196_994


def test_classifier_architecture():
    model = Classifier(16)
    kinds = [type(m).__name__ for m in model.net]
    assert kinds == ["Linear", "ReLU", "Dropout", "Linear"]
    assert model.net[0].out_features == 128 and model.net[2].p == 0.3
    assert model(torch.zeros(5, 16)).shape == (5, 2)


def test_class_weights():
    labels = torch.tensor([0] * 2_580 + [1] * 22_800)
    weights = T.class_weights(labels, "cpu")
    assert weights.tolist() == pytest.approx([25_380 / (2 * 2_580), 25_380 / (2 * 22_800)])
    assert weights.tolist() == pytest.approx([4.92, 0.56], abs=0.005)


# Fit -------------------------------------------------------------------------------

def test_constants_of_the_training_recipe():
    assert (cfg.EPOCHS, cfg.BATCH_SIZE, cfg.LEARNING_RATE, cfg.WEIGHT_DECAY) == (50, 128, 1e-4, 1e-5)
    assert (cfg.DEV_CHECK_EVERY, cfg.PATIENCE_CHECKS) == (5, 10)


def fit_with_dev_eers(data, monkeypatch, values):
    """Run ``fit`` with the development-set EER of check k replaced by values[k]."""
    calls = []

    def fake_eer(labels, scores):
        calls.append(len(calls))
        return values[len(calls) - 1]

    monkeypatch.setattr(T, "eer_roc_corners", fake_eer)
    train_x, train_y, _ = data["train"]
    dev_x, dev_y, _ = data["dev"]
    return T.fit(train_x, train_y, dev_x, dev_y, 42, "cpu"), len(calls)


def test_dev_is_checked_after_epoch_1_and_every_5_epochs(data, monkeypatch):
    # Checks at epochs 1, 5, 10, ..., 50: eleven checks.
    improving, n_checks = fit_with_dev_eers(data, monkeypatch, [50.0 - k for k in range(11)])
    assert n_checks == 11 and improving.best_epoch == 50 and improving.epochs_run == 50
    assert improving.best_dev_eer == 40.0
    fourth, _ = fit_with_dev_eers(data, monkeypatch, [9, 8, 7, 1, 7, 8, 9, 9, 9, 9, 9])
    assert fourth.best_epoch == 15 and fourth.best_dev_eer == 1


def test_a_tie_keeps_the_earlier_checkpoint(data, monkeypatch):
    tied, _ = fit_with_dev_eers(data, monkeypatch, [9, 3, 7, 7, 3, 7, 7, 7, 7, 7, 3])
    assert tied.best_epoch == 5


def test_training_never_stops_before_the_last_epoch(data, monkeypatch):
    constant, n_checks = fit_with_dev_eers(data, monkeypatch, [5.0] * 11)
    assert n_checks == 11 and constant.epochs_run == 50 and constant.best_epoch == 1


def test_the_selected_checkpoint_is_the_one_that_scores(data):
    train_x, train_y, _ = data["train"]
    dev_x, dev_y, _ = data["dev"]
    fitted = T.fit(train_x, train_y, dev_x, dev_y, 42, "cpu")
    assert not fitted.model.training
    scores = T.score(fitted.model, dev_x, "cpu")
    from ssl_silence.metrics import eer_roc_corners
    assert eer_roc_corners(dev_y.numpy(), scores) == fitted.best_dev_eer
    assert scores.dtype == np.float32 and scores.min() >= 0 and scores.max() <= 1
    with torch.no_grad():
        expected = torch.softmax(fitted.model(dev_x), 1)[:, 1].numpy()
    assert np.array_equal(scores, expected)


def test_same_seed_is_bit_identical_and_seeds_differ(data):
    first, again = run(data, save=False), run(data, save=False)
    other = run(data, T.RunSpec("wavlm-base", "unpadded", "clean", 4, "stats", 43), save=False)
    for name in first.scores:
        assert np.array_equal(first.scores[name].scores, again.scores[name].scores)
    assert not np.array_equal(first.scores["eval@clean"].scores, other.scores["eval@clean"].scores)
    assert first.best_epoch == again.best_epoch and first.best_dev_eer == again.best_dev_eer


# Run description ---------------------------------------------------------------------

def test_run_ids_and_selection_set():
    assert SPEC.run_id == "wavlm-base_unpadded_clean_L04_stats_s42"
    assert SPEC.selection_set == "dev@clean"
    padded = T.RunSpec("wavlm-base", "padded", "trim40", 4, "mean", 46, "real-frames")
    assert padded.run_id == "wavlm-base_padded_real-frames_trim40_L04_mean_s46"
    assert padded.selection_set == "dev@trim40"
    assert T.RunSpec("hubert-base", "unpadded", "trim15", 12, "mean", 44).run_id == \
        "hubert-base_unpadded_trim15_L12_mean_s44"
    for bad in (dict(encoder="other"), dict(condition="trimmed"), dict(extraction="masked"),
                dict(pooling="max"), dict(frames="all-frames")):
        with pytest.raises((KeyError, ValueError)):
            T.RunSpec(**{**dict(encoder="wavlm-base", extraction="unpadded", condition="clean",
                                layer=4, pooling="stats", seed=42), **bad})
    with pytest.raises(ValueError):
        T.RunSpec("wavlm-base", "padded", "clean", 4, "stats", 42)


def test_planned_runs_of_the_design():
    wavlm = T.plan_runs("wavlm-base", "unpadded")
    assert len(wavlm) == 480 and len({p.spec.run_id for p in wavlm}) == 480
    counts = collections.Counter((p.spec.condition, p.test_conditions) for p in wavlm)
    assert counts == {("clean", ("clean", "trim40")): 90,
                      ("clean", ("clean", "trim40", "trim20", "trim15")): 30,
                      ("trim40", ("trim40", "clean")): 120,
                      ("trim20", ("trim20",)): 90, ("trim20", ("trim20", "clean")): 30,
                      ("trim15", ("trim15",)): 90, ("trim15", ("trim15", "clean")): 30}
    cross_layers = {p.spec.layer for p in wavlm if p.test_conditions == ("trim20", "clean")}
    assert cross_layers == {3, 4, 5}
    assert collections.Counter(p.spec.seed for p in wavlm) == {s: 96 for s in (42, 43, 44, 45, 46)}

    padded = T.plan_runs("wavlm-base", "padded")
    assert sorted({p.spec.run_id.rsplit("_s", 1)[0] for p in padded}) == [
        "wavlm-base_padded_all-frames_clean_L04_stats", "wavlm-base_padded_real-frames_trim40_L04_mean"]
    assert len(padded) == 10

    hubert = T.plan_runs("hubert-base", "unpadded")
    assert len(hubert) == 480
    assert collections.Counter((p.spec.condition, p.test_conditions) for p in hubert) == {
        ("clean", ("clean", "trim40", "trim20", "trim15")): 120, ("trim40", ("trim40", "clean")): 120,
        ("trim20", ("trim20", "clean")): 120, ("trim15", ("trim15", "clean")): 120}


def test_score_sets_of_a_planned_run():
    run_ = T.PlannedRun(SPEC, ("clean", "trim40"))
    assert run_.score_set_names == ("eval@clean", "itw@clean", "dev@trim40", "eval@trim40",
                                    "itw@trim40")
    assert T.PlannedRun(SPEC, ("clean",)).score_set_names == ("eval@clean", "itw@clean")


# Score files and runs.csv --------------------------------------------------------------

def test_scores_are_saved_with_labels_ids_and_meta(data, tmp_path):
    result = run(data, runs_dir=str(tmp_path))
    assert result.score_path == os.path.join(str(tmp_path), "scores", SPEC.run_id + ".npz")
    sets, meta = T.load_scores(result.score_path)
    assert list(sets) == ["dev@clean", "eval@clean", "eval@trim40"]  # no bare "dev"
    for name, key in (("dev@clean", "dev"), ("eval@clean", "eval"), ("eval@trim40", "other")):
        _, y, ids = data[key]
        assert sets[name].scores.dtype == np.float32 and sets[name].labels.dtype == np.int64
        assert np.array_equal(sets[name].labels, y.numpy()) and sets[name].ids.tolist() == ids
        assert np.array_equal(sets[name].scores, result.scores[name].scores)
    assert meta["selection_set"] == "dev@clean" and meta["condition"] == "clean"
    assert meta["run_id"] == SPEC.run_id and meta["score_sets"] == "dev@clean+eval@clean+eval@trim40"

    runs = T.read_runs(str(tmp_path))
    assert list(runs.columns) == T.RUNS_CSV_FIELDS and len(runs) == 1
    row = runs.iloc[0]
    assert row["run_id"] == SPEC.run_id and row["selection_set"] == "dev@clean"
    assert row["test_conditions"] == "clean+trim40" and row["trim_top_db"] == ""
    assert (row["n_train"], row["in_dim"], row["epochs_run"]) == ("420", "16", "50")
    assert float(row["best_dev_eer_roc_corners"]) == result.best_dev_eer
    assert row["best_epoch"] == str(result.best_epoch) and row["score_file"] == f"scores/{SPEC.run_id}.npz"
    assert row["device"] == "cpu" and row["gpu_name"] == "cpu"
    assert T.finished_run_ids(str(tmp_path)) == {SPEC.run_id}


def test_existing_scores_are_not_replaced_without_overwrite(data, tmp_path):
    run(data, runs_dir=str(tmp_path))
    with pytest.raises(FileExistsError):
        run(data, runs_dir=str(tmp_path))
    run(data, runs_dir=str(tmp_path), overwrite=True)
    assert len(T.read_runs(str(tmp_path))) == 1  # the last row of a run id wins


def test_score_set_names_are_checked(data):
    train_x, train_y, _ = data["train"]
    dev_x, dev_y, _ = data["dev"]
    for bad in ("dev", "eval", "test@clean", "eval@trimmed", "dev@clean"):
        with pytest.raises((KeyError, ValueError)):
            T.train_and_score(train_x, train_y, dev_x, dev_y, {bad: data["eval"]}, SPEC,
                              device="cpu", save=False)
    with pytest.raises(ValueError):
        T.train_and_score(train_x, train_y, dev_x, dev_y, {"eval@clean": data["eval"]}, SPEC,
                          dev_ids=["a"], device="cpu", save=False)


# Layer cache and the run loop ---------------------------------------------------------

@pytest.fixture(scope="module")
def root(tmp_path_factory):
    base = tmp_path_factory.mktemp("root")
    for k, condition in enumerate(("clean", "trim40")):
        write_feature_condition(cfg.feature_dir("wavlm-base", "unpadded", condition, str(base)), k)
    return str(base)


def test_layer_cache_equals_the_shards(root, tmp_path):
    feature_dir = cfg.feature_dir("wavlm-base", "unpadded", "clean", root)
    cache = str(tmp_path / "cache")
    index = shards.build_layer_cache(feature_dir, "dev", cache)
    stamp = os.path.getmtime(index)
    assert shards.build_layer_cache(feature_dir, "dev", cache) == index
    assert os.path.getmtime(index) == stamp
    for layer in (1, 3, 5):
        for pooling in ("mean", "stats"):
            cached = shards.load_layer_from_cache(cache, "dev", layer, pooling)
            direct = shards.load_layer(feature_dir, "dev", layer, pooling)
            assert torch.equal(cached[0], direct[0]) and torch.equal(cached[1], direct[1])
            assert cached[2] == direct[2]
    with pytest.raises(ValueError):
        shards.load_layer_from_cache(cache, "dev", 6, "mean")
    with pytest.raises(ValueError):
        shards.load_layer_from_cache(cache, "dev", 1, "mean", frames="all-frames")


def planned(layers=(2, 4), seeds=(42, 43)):
    out = []
    for condition, other in (("clean", "trim40"), ("trim40", "clean")):
        for layer in layers:
            for pooling in ("mean", "stats"):
                for seed in seeds:
                    out.append(T.PlannedRun(T.RunSpec("wavlm-base", "unpadded", condition, layer,
                                                      pooling, seed), (condition, other)))
    return out


def test_run_loop_is_resumable_and_uses_the_training_function(root, tmp_path):
    runs_dir, cache = str(tmp_path / "runs"), str(tmp_path / "cache")
    plan = planned(layers=(2,), seeds=(42,))
    first = T.run_experiments(plan, feature_root=root, cache_root=cache, runs_dir=runs_dir,
                              device="cpu", max_new_runs=1, progress=False)
    assert first["status"].tolist() == ["trained", "pending", "pending", "pending"]
    done = T.run_experiments(plan, feature_root=root, cache_root=cache, runs_dir=runs_dir,
                             device="cpu", progress=False)
    assert done["status"].tolist() == ["skipped", "trained", "trained", "trained"]
    assert T.finished_run_ids(runs_dir) == {p.spec.run_id for p in plan}

    spec = plan[1].spec  # clean, layer 2, stats
    sets, meta = T.load_scores(os.path.join(runs_dir, "scores", spec.run_id + ".npz"))
    assert list(sets) == ["dev@clean", "eval@clean", "itw@clean", "dev@trim40", "eval@trim40",
                          "itw@trim40"]
    assert meta["selection_set"] == "dev@clean"
    assert {name: len(s.scores) for name, s in sets.items()} == {
        f"{split}@{c}": SPLIT_SIZES[split] for c in ("clean", "trim40")
        for split in ("dev", "eval", "itw")}
    assert sets["eval@trim40"].ids[0] == "eval_0"

    # The same run through the training function with features read directly.
    load = lambda condition, split: shards.load_layer(
        cfg.feature_dir("wavlm-base", "unpadded", condition, root), split, 2, "stats")
    train_x, train_y, _ = load("clean", "train")
    dev_x, dev_y, dev_ids = load("clean", "dev")
    direct = T.train_and_score(
        train_x, train_y, dev_x, dev_y,
        {"eval@clean": load("clean", "eval"), "eval@trim40": load("trim40", "eval")}, spec,
        dev_ids=dev_ids, device="cpu", save=False)
    for name in ("dev@clean", "eval@clean", "eval@trim40"):
        assert np.array_equal(direct.scores[name].scores, sets[name].scores)

    # Mean pooling is the first half of the statistics vector.
    mean_spec = plan[0].spec
    mean_sets, _ = T.load_scores(os.path.join(runs_dir, "scores", mean_spec.run_id + ".npz"))
    load_mean = lambda condition, split: shards.load_layer(
        cfg.feature_dir("wavlm-base", "unpadded", condition, root), split, 2, "mean")
    tx, ty, _ = load_mean("clean", "train")
    dx, dy, _ = load_mean("clean", "dev")
    direct = T.train_and_score(tx, ty, dx, dy, {"itw@trim40": load_mean("trim40", "itw")},
                               mean_spec, device="cpu", save=False)
    assert np.array_equal(direct.scores["itw@trim40"].scores, mean_sets["itw@trim40"].scores)


def test_an_interrupted_run_is_trained_again(root, tmp_path):
    runs_dir, cache = str(tmp_path / "runs"), str(tmp_path / "cache")
    plan = planned(layers=(4,), seeds=(42,))[:1]
    T.run_experiments(plan, feature_root=root, cache_root=cache, runs_dir=runs_dir, device="cpu",
                      progress=False)
    pd.DataFrame(columns=T.RUNS_CSV_FIELDS).to_csv(os.path.join(runs_dir, "runs.csv"), index=False)
    assert T.finished_run_ids(runs_dir) == set()  # a score file without a row is not finished
    again = T.run_experiments(plan, feature_root=root, cache_root=cache, runs_dir=runs_dir,
                              device="cpu", progress=False)
    assert again["status"].tolist() == ["trained"] and len(T.read_runs(runs_dir)) == 1


def test_run_loop_takes_one_encoder_and_extraction():
    mixed = [T.PlannedRun(SPEC, ("clean",)),
             T.PlannedRun(T.RunSpec("hubert-base", "unpadded", "clean", 4, "stats", 42), ("clean",))]
    with pytest.raises(ValueError):
        T.run_experiments(mixed, feature_root="/x", cache_root="/y", runs_dir="/z", device="cpu")
    with pytest.raises(ValueError):
        T.run_experiments([mixed[0], mixed[0]], feature_root="/x", cache_root="/y", runs_dir="/z",
                          device="cpu")


# Pilot and report of a grid ----------------------------------------------------------------

def test_the_pilot_is_the_two_regression_runs():
    assert (cfg.PILOT_LAYER, cfg.PILOT_POOLING, cfg.PILOT_SEED, cfg.PILOT_CONDITIONS) == \
        (4, "stats", 42, ("clean", "trim40"))
    for encoder in ("wavlm-base", "hubert-base"):
        pilot = T.pilot_runs(T.plan_runs(encoder, "unpadded"))
        assert [p.spec.run_id for p in pilot] == [f"{encoder}_unpadded_clean_L04_stats_s42",
                                                  f"{encoder}_unpadded_trim40_L04_stats_s42"]
    # WavLM Base: the untrimmed pilot model is scored in all four conditions.
    pilot = T.pilot_runs(T.plan_runs("wavlm-base", "unpadded"))
    assert pilot[0].test_conditions == ("clean", "trim40", "trim20", "trim15")
    assert pilot[1].test_conditions == ("trim40", "clean")
    with pytest.raises(ValueError, match="0 pilot runs"):
        T.pilot_runs(T.plan_runs("wavlm-base", "unpadded", seeds=(43,)))
    with pytest.raises(ValueError, match="pilot runs"):
        T.pilot_runs(T.plan_runs("wavlm-base", "padded"))       # layer 4 with both statistics, untrimmed only


def test_grid_report_counts_finished_runs_and_estimates_the_time(root, tmp_path):
    runs_dir, cache = str(tmp_path / "runs"), str(tmp_path / "cache")
    plan = planned(layers=(2,), seeds=(42, 43))                 # two conditions, two poolings, two seeds
    empty = T.grid_report(plan, runs_dir).set_index("condition")
    assert empty.index.tolist() == ["clean", "trim40", "all"]
    assert empty["planned"].tolist() == [4, 4, 8] == empty["to_train"].tolist()
    assert empty["finished"].tolist() == [0, 0, 0] and empty["hours_to_train"].isna().all()
    status = T.run_experiments(plan, feature_root=root, cache_root=cache, runs_dir=runs_dir,
                               device="cpu", max_new_runs=3, progress=False)
    report = T.grid_report(plan, runs_dir, [90.0, float("nan"), 30.0]).set_index("condition")
    by_condition = {c: sum(p.spec.condition == c and s == "trained"
                           for p, s in zip(plan, status["status"])) for c in ("clean", "trim40")}
    assert report["finished"].tolist() == [by_condition["clean"], by_condition["trim40"], 3]
    assert report["to_train"].tolist() == [4 - by_condition["clean"], 4 - by_condition["trim40"], 5]
    assert set(report["seconds_per_run"]) == {60.0}
    assert report.loc["all", "hours_to_train"] == pytest.approx(5 * 60 / 3600)


# The checkpoint revision of the features in the run records -----------------------------------

OLD_FIELDS = [name for name in T.RUNS_CSV_FIELDS if name != "encoder_revision"]


def test_the_row_and_the_score_file_record_the_encoder_revision(data, tmp_path):
    result = run(data, runs_dir=str(tmp_path), encoder_revision="abc123")
    row = T.read_runs(str(tmp_path)).iloc[0]
    assert row["encoder_revision"] == "abc123" and row["encoder"] == "wavlm-base"
    assert T.load_scores(result.score_path)[1]["encoder_revision"] == "abc123"
    other = T.RunSpec("wavlm-base", "unpadded", "clean", 4, "stats", 43)
    run(data, spec=other, runs_dir=str(tmp_path))                  # not given: empty
    assert T.read_runs(str(tmp_path))["encoder_revision"].tolist() == ["abc123", ""]
    assert T.RUNS_CSV_FIELDS[:3] == ["run_id", "encoder", "encoder_revision"]


def test_a_runs_file_written_before_the_revision_column_keeps_its_columns(data, tmp_path):
    runs_dir = str(tmp_path)
    path = os.path.join(runs_dir, "runs.csv")
    old_row = {name: "x" for name in OLD_FIELDS}
    old_row["run_id"] = "an_old_run"
    pd.DataFrame([old_row])[OLD_FIELDS].to_csv(path, index=False)
    before = open(path, encoding="utf-8").read()
    result = run(data, runs_dir=runs_dir, encoder_revision="abc123")
    text = open(path, encoding="utf-8").read()
    assert text.startswith(before) and text.splitlines()[0].split(",") == OLD_FIELDS
    assert len(text.splitlines()) == 3 and "abc123" not in text
    runs = T.read_runs(runs_dir)
    assert list(runs.columns) == T.RUNS_CSV_FIELDS and runs["encoder_revision"].tolist() == ["", ""]
    assert runs["run_id"].tolist() == ["an_old_run", SPEC.run_id]
    assert runs.iloc[1]["condition"] == "clean" and runs.iloc[1]["epochs_run"] == "50"
    assert T.load_scores(result.score_path)[1]["encoder_revision"] == "abc123"   # the score file has it
    assert T.finished_run_ids(runs_dir) == {SPEC.run_id}


REVISION = "0123456789abcdef0123456789abcdef01234567"
COMMIT = "a" * 40


def set_shard_meta(folder: str, **meta) -> None:
    """Add keys to the meta of every shard of a folder."""
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        payload = torch.load(path, weights_only=False)
        payload["meta"] = {**payload["meta"], **meta}
        torch.save(payload, path)


@pytest.fixture()
def hubert(tmp_path, monkeypatch):
    """Features of a small stand-in for HuBERT Base in two conditions with the same files and
    labels, a registry entry that fits them, and a commit without changes."""
    base = str(tmp_path / "root")
    for condition in ("clean", "trim40"):
        folder = cfg.feature_dir("hubert-base", "unpadded", condition, base)
        write_feature_condition(folder, 7)
        set_shard_meta(folder, encoder="hubert-base", revision=REVISION)
    entry = dict(cfg.encoder("hubert-base"), hidden_size=8, n_layers=5, revision=REVISION)
    real = cfg.encoder
    monkeypatch.setattr(cfg, "encoder", lambda name: entry if name == "hubert-base" else real(name))
    monkeypatch.setattr(T.provenance, "git_commit", lambda: COMMIT)
    plan = [T.PlannedRun(T.RunSpec("hubert-base", "unpadded", condition, 2, pooling, 42), (condition, other))
            for condition, other in (("clean", "trim40"), ("trim40", "clean")) for pooling in ("mean", "stats")]
    lengths = shards.shard_lengths(cfg.feature_dir("hubert-base", "unpadded", "clean", base))
    lists = {split: part[["audio_id", "label"]].reset_index(drop=True)
             for split, part in lengths.groupby("split", sort=False)}
    runs_dir = str(tmp_path / "runs")
    T.run_experiments(plan, feature_root=base, cache_root=str(tmp_path / "cache"), runs_dir=runs_dir,
                      device="cpu", progress=False)
    return {"root": base, "plan": plan, "lists": lists, "runs_dir": runs_dir, "tmp": tmp_path}


def test_the_run_loop_records_the_revision_of_the_shards_it_read(hubert, tmp_path):
    runs = T.read_runs(hubert["runs_dir"])
    assert set(runs["encoder_revision"]) == {REVISION} and set(runs["encoder"]) == {"hubert-base"}
    assert set(runs["git_commit"]) == {COMMIT} and len(runs) == 4
    # Shards of two revisions are not one set of features.
    set_shard_meta(cfg.feature_dir("hubert-base", "unpadded", "trim40", hubert["root"]), revision="another")
    with pytest.raises(ValueError, match="more than one checkpoint revision"):
        T.run_experiments(hubert["plan"][:1], feature_root=hubert["root"], cache_root=str(tmp_path / "c2"),
                          runs_dir=str(tmp_path / "r2"), device="cpu", progress=False)
    assert not os.path.exists(os.path.join(str(tmp_path / "r2"), "runs.csv"))


def test_shards_without_a_revision_give_an_empty_revision(root, tmp_path):
    runs_dir = str(tmp_path / "runs")
    T.run_experiments(planned(layers=(2,), seeds=(42,))[:1], feature_root=root, cache_root=str(tmp_path / "c"),
                      runs_dir=runs_dir, device="cpu", progress=False)
    assert T.read_runs(runs_dir)["encoder_revision"].tolist() == [""]


# Checks of finished runs without original scores -----------------------------------------------

def test_finished_runs_pass_both_checks(hubert):
    plan, runs_dir = hubert["plan"], hubert["runs_dir"]
    sets = T.score_set_check(plan, runs_dir, hubert["lists"])
    assert len(sets) == 4 * 6 and sets["ok"].all() and sets["planned"].all() and sets["present"].all()
    first = sets[sets["run_id"] == plan[0].spec.run_id]
    assert first["score_set"].tolist() == ["dev@clean", "eval@clean", "itw@clean", "dev@trim40", "eval@trim40",
                                           "itw@trim40"]
    assert first["n_scores"].tolist() == first["n_files"].tolist() == \
        [SPLIT_SIZES[name.split("@")[0]] for name in first["score_set"]]
    # No metric in the table.
    assert not [column for column in sets.columns if "eer" in column or "dcf" in column]

    records = T.run_record_check(plan, runs_dir, n_train=SPLIT_SIZES["train"])
    assert len(records) == 4 and records["ok"].all() and records["finished"].all()
    assert set(records["encoder"]) == {"hubert-base"} and set(records["encoder_revision"]) == {REVISION}
    assert set(records["git_commit"]) == {COMMIT} and set(records["epochs_run"]) == {"50"}
    assert records["in_dim"].tolist() == ["8", "16", "8", "16"]
    assert not [column for column in records.columns if "eer" in column]


def rewrite_scores(runs_dir, run_id, change):
    path = os.path.join(runs_dir, "scores", run_id + ".npz")
    sets, meta = T.load_scores(path)
    change(sets, meta)
    T.save_scores(path, sets, meta)


@pytest.mark.parametrize("problem, column", [
    ("order", "same_ids_in_order"), ("label", "same_labels"), ("nan", "valid_scores"),
    ("above_one", "valid_scores"), ("missing_set", "present"), ("extra_set", "planned"),
    ("short", "same_ids_in_order")])
def test_a_score_set_that_is_not_the_file_list_fails(hubert, problem, column):
    plan, runs_dir = hubert["plan"], hubert["runs_dir"]
    target, name = plan[1].spec.run_id, "eval@trim40"

    def change(sets, meta):
        s = sets[name]
        if problem == "order":
            sets[name] = T.ScoreSet(s.scores[::-1].copy(), s.labels[::-1].copy(), s.ids[::-1].copy())
        elif problem == "label":
            s.labels[3] = 1 - s.labels[3]
        elif problem == "nan":
            s.scores[5] = np.nan
        elif problem == "above_one":
            s.scores[5] = 1.5
        elif problem == "missing_set":
            del sets[name]
        elif problem == "extra_set":
            sets["eval@trim20"] = T.ScoreSet(s.scores, s.labels, s.ids)
        elif problem == "short":
            sets[name] = T.ScoreSet(s.scores[:-1], s.labels[:-1], s.ids[:-1])

    rewrite_scores(runs_dir, target, change)
    table = T.score_set_check(plan, runs_dir, hubert["lists"])
    failed = table[~table["ok"]]
    assert len(failed) == 1 and failed.iloc[0]["run_id"] == target
    assert failed.iloc[0]["score_set"] == ("eval@trim20" if problem == "extra_set" else name)
    assert not failed.iloc[0][column]
    assert len(table) == 4 * 6 + (problem == "extra_set")


def test_a_run_without_a_score_file_fails_every_planned_set_and_its_record(hubert):
    plan, runs_dir = hubert["plan"], hubert["runs_dir"]
    path = os.path.join(runs_dir, "scores", plan[2].spec.run_id + ".npz")
    os.replace(path, path + ".away")
    table = T.score_set_check(plan, runs_dir, hubert["lists"])
    failed = table[~table["ok"]]
    assert len(failed) == 6 and set(failed["run_id"]) == {plan[2].spec.run_id} and not failed["present"].any()
    records = T.run_record_check(plan, runs_dir, n_train=SPLIT_SIZES["train"])
    assert records["ok"].tolist() == [True, True, False, True]
    assert records["finished"].tolist() == [True, True, False, True]
    # Nothing was trained at all: every row fails, nothing raises.
    empty = T.run_record_check(plan, str(hubert["tmp"] / "nothing"), n_train=SPLIT_SIZES["train"])
    assert not empty["ok"].any() and not empty["finished"].any()
    assert not T.score_set_check(plan, str(hubert["tmp"] / "nothing"), hubert["lists"])["ok"].any()


def rewrite_row(runs_dir, run_id, **values):
    path = os.path.join(runs_dir, "runs.csv")
    runs = pd.read_csv(path, dtype=str, keep_default_na=False)
    for key, value in values.items():
        runs.loc[runs["run_id"] == run_id, key] = value
    runs.to_csv(path, index=False)


@pytest.mark.parametrize("values, column", [
    ({"encoder_revision": "another"}, "revision_of_the_registry"),
    ({"encoder_revision": ""}, "revision_of_the_registry"),
    ({"git_commit": COMMIT + "-dirty"}, "clean_commit"),
    ({"git_commit": "unknown"}, "clean_commit"),
    ({"encoder": "wavlm-base"}, "names_the_run"),
    ({"seed": "43"}, "names_the_run"),
    ({"score_sets": "dev@clean+eval@clean+itw@clean"}, "names_the_run"),
    ({"epochs_run": "45"}, "trained_as_planned"),
    ({"best_epoch": "7"}, "trained_as_planned"),
    ({"n_train": "5"}, "trained_as_planned"),
    ({"in_dim": "16"}, "trained_as_planned"),
    ({"best_dev_eer_roc_corners": "nan"}, "trained_as_planned"),
    ({"gpu_name": "another GPU"}, "meta_equals_row")])
def test_a_run_record_that_is_not_as_planned_fails(hubert, values, column):
    plan, runs_dir = hubert["plan"], hubert["runs_dir"]
    rewrite_row(runs_dir, plan[0].spec.run_id, **values)        # clean, layer 2, mean
    records = T.run_record_check(plan, runs_dir, n_train=SPLIT_SIZES["train"])
    assert records["ok"].tolist() == [False, True, True, True]
    assert not records.iloc[0][column]


def test_the_revision_of_the_registry_is_what_the_check_compares_with(hubert, monkeypatch):
    entry = dict(cfg.encoder("hubert-base"), revision="the registry was changed")
    monkeypatch.setattr(cfg, "encoder", lambda name: entry)
    records = T.run_record_check(hubert["plan"], hubert["runs_dir"], n_train=SPLIT_SIZES["train"])
    assert not records["ok"].any() and not records["revision_of_the_registry"].any()
    assert records["names_the_run"].all() and records["meta_equals_row"].all()
