"""classifier and training against the original training function, on a CPU.
Skipped without the original code.

The original code trained two models for the two test conditions of a
configuration: a grid run (scored in its training condition) and a
cross-condition run (same seed, scored in the other condition). Here one
model is scored in both. Mapping of the score sets, for training condition c
and other condition o:

    this repository    original grid run    original cross-condition run
    dev@c              dev                  dev
    eval@c, itw@c      eval, itw            eval_tr, itw_tr
    dev@o                                   dev_tc
    eval@o, itw@o                           eval, itw
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from original_names import ORIGINAL_VALUES
from synthetic import gaussian_features, write_feature_condition
from ssl_silence import config as cfg
from ssl_silence import training as T
from ssl_silence.classifier import Classifier

# Conditions of this repository -> condition names of the original code.
CONDITIONS = {"clean": "clean", "trim40": "trimmed"}
LAYERS, POOLINGS, SEEDS = (2, 4), ("mean", "stats"), (42, 43)


def record(parity_record, name, differences):
    worst = max(differences)
    parity_record("training", name, len(differences), worst)
    assert worst == 0.0, f"{name}: max abs diff {worst}"


def identical(new, old) -> float:
    new, old = np.asarray(new), np.asarray(old)
    if new.shape != old.shape or new.dtype != old.dtype:
        return np.inf
    if new.dtype.kind in "OUS":
        return 0.0 if (new == old).all() else np.inf
    return float(np.max(np.abs(new.astype(np.float64) - old.astype(np.float64)))) if new.size else 0.0


def test_classifier_and_class_weights(original, parity_record):
    old_models, old_train = original("models"), original("train")
    differences = []
    for in_dim in (16, 768, 1536):
        for seed in (0, 42):
            torch.manual_seed(seed)
            new = Classifier(in_dim)
            torch.manual_seed(seed)
            old = old_models.Classifier(in_dim)
            assert list(new.state_dict()) == list(old.state_dict())
            differences += [identical(a, b) for a, b in zip(new.state_dict().values(),
                                                            old.state_dict().values())]
            x = torch.randn(9, in_dim, generator=torch.Generator().manual_seed(seed))
            new.eval(), old.eval()
            with torch.no_grad():
                differences.append(identical(new(x), old(x)))
    record(parity_record, "Classifier (initial weights and outputs)", differences)
    differences = []
    for n_bonafide, n_spoof in ((2_580, 22_800), (1, 1), (37, 400)):
        labels = torch.tensor([0] * n_bonafide + [1] * n_spoof)
        differences.append(identical(T.class_weights(labels, "cpu"), old_train.class_weights(labels, "cpu")))
    record(parity_record, "class_weights", differences)


def test_one_model_scored_in_both_conditions(original, parity_record):
    """train_and_score on tensors: the grid run and the cross-condition run of the original."""
    O = original("train")
    scores, selection = [], []
    for dim, pooling in ((16, "mean"), (32, "stats")):
        train_x, train_y, _ = gaussian_features(420, dim, 10 + dim)
        own = {split: gaussian_features(n, dim, seed + dim)
               for split, n, seed in (("dev", 200, 11), ("eval", 300, 12), ("itw", 250, 13))}
        other = {split: gaussian_features(n, dim, seed + dim, shift=0.5)
                 for split, n, seed in (("dev", 200, 21), ("eval", 300, 22), ("itw", 250, 23))}
        for seed in (42, 43, 44):
            spec = T.RunSpec("wavlm-base", "unpadded", "clean", 4, pooling, seed)
            new = T.train_and_score(
                train_x, train_y, own["dev"][0], own["dev"][1],
                {"eval@clean": own["eval"], "itw@clean": own["itw"], "dev@trim40": other["dev"],
                 "eval@trim40": other["eval"], "itw@trim40": other["itw"]},
                spec, dev_ids=own["dev"][2], device="cpu", save=False)
            grid = O.train_and_score(
                train_x, train_y, own["dev"][0], own["dev"][1],
                {"eval": own["eval"], "itw": own["itw"]},
                O.RunSpec(layer=4, pooling=pooling, seed=seed), dev_ids=own["dev"][2],
                device="cpu", save=False)
            cross = O.train_and_score(
                train_x, train_y, own["dev"][0], own["dev"][1],
                {"eval": other["eval"], "itw": other["itw"], "dev_tc": other["dev"],
                 "eval_tr": own["eval"], "itw_tr": own["itw"]},
                O.RunSpec(layer=4, pooling=pooling, seed=seed, test_condition="trimmed"),
                dev_ids=own["dev"][2], device="cpu", save=False)
            pairs = [("dev@clean", grid, "dev"), ("eval@clean", grid, "eval"),
                     ("itw@clean", grid, "itw"), ("dev@clean", cross, "dev"),
                     ("eval@clean", cross, "eval_tr"), ("itw@clean", cross, "itw_tr"),
                     ("dev@trim40", cross, "dev_tc"), ("eval@trim40", cross, "eval"),
                     ("itw@trim40", cross, "itw")]
            for name, old, old_name in pairs:
                for part in ("scores", "labels", "ids"):
                    scores.append(identical(getattr(new.scores[name], part),
                                            getattr(old.scores[old_name], part)))
            for old in (grid, cross):
                selection += [abs(new.best_dev_eer - old.best_dev_eer),
                              float(abs(new.best_epoch - old.best_epoch)),
                              float(abs(new.epochs_run - old.epochs_run))]
    record(parity_record, "train_and_score (scores, labels, ids)", scores)
    record(parity_record, "train_and_score (best dev EER, best epoch, epochs)", selection)


@pytest.fixture(scope="module")
def feature_roots(tmp_path_factory):
    """The same synthetic shards in the folder layouts of both codes."""
    base = tmp_path_factory.mktemp("features")
    new_root, old_root = str(base / "new"), str(base / "old")
    for k, (condition, old_condition) in enumerate(CONDITIONS.items()):
        write_feature_condition(cfg.feature_dir("wavlm-base", "unpadded", condition, new_root), k)
        write_feature_condition(os.path.join(old_root, "wavlm-base", old_condition), k)
    return new_root, old_root, base


def test_from_shards_to_score_files(original, parity_record, feature_roots):
    """The run loops: one run per configuration here, a grid run and a cross-condition run there."""
    O = original("train")
    new_root, old_root, base = feature_roots
    feature_set = ORIGINAL_VALUES["unpadded_feature_set"]
    new_runs_dir, old_results = str(base / "runs"), str(base / "old_results")

    planned, old_specs = [], []
    for condition, other in (("clean", "trim40"), ("trim40", "clean")):
        for layer in LAYERS:
            for pooling in POOLINGS:
                for seed in SEEDS:
                    planned.append(T.PlannedRun(
                        T.RunSpec("wavlm-base", "unpadded", condition, layer, pooling, seed),
                        (condition, other)))
                    common = dict(encoder="wavlm-base", layer=layer, pooling=pooling, mask="none",
                                  condition=CONDITIONS[condition], seed=seed, feature_set=feature_set)
                    old_specs += [O.RunSpec(**common),
                                  O.RunSpec(**common, test_condition=CONDITIONS[other])]
    T.run_experiments(planned, feature_root=new_root, cache_root=str(base / "cache"),
                      runs_dir=new_runs_dir, device="cpu", progress=False)
    O.run_grid(old_specs, feature_root=old_root, cache_root=str(base / "old_cache"),
               results_dir=old_results, device="cpu", progress=False)

    new_runs = T.read_runs(new_runs_dir).set_index("run_id")
    old_runs = O.read_runs(old_results).set_index("run_id")
    assert len(new_runs) == len(planned) == 16 and len(old_runs) == len(old_specs) == 32
    scores, selection = [], []
    for k, run in enumerate(planned):
        spec, (condition, other) = run.spec, run.test_conditions
        new, meta = T.load_scores(os.path.join(new_runs_dir, "scores", spec.run_id + ".npz"))
        assert meta["selection_set"] == f"dev@{condition}"
        grid_spec, cross_spec = old_specs[2 * k], old_specs[2 * k + 1]
        grid = O.load_scores(os.path.join(old_results, "scores", grid_spec.run_id + ".npz"))
        cross = O.load_scores(os.path.join(old_results, "scores", cross_spec.run_id + ".npz"))
        pairs = [(f"dev@{condition}", grid, "dev"), (f"eval@{condition}", grid, "eval"),
                 (f"itw@{condition}", grid, "itw"), (f"dev@{condition}", cross, "dev"),
                 (f"eval@{condition}", cross, "eval_tr"), (f"itw@{condition}", cross, "itw_tr"),
                 (f"dev@{other}", cross, "dev_tc"), (f"eval@{other}", cross, "eval"),
                 (f"itw@{other}", cross, "itw")]
        assert set(new) == {name for name, _, _ in pairs}
        for name, old, old_name in pairs:
            for part in ("scores", "labels", "ids"):
                scores.append(identical(getattr(new[name], part), getattr(old[old_name], part)))
        for old_spec in (grid_spec, cross_spec):
            old_row, new_row = old_runs.loc[old_spec.run_id], new_runs.loc[spec.run_id]
            selection += [
                abs(float(new_row["best_dev_eer_roc_corners"])
                    - float(old_row[ORIGINAL_VALUES["best_dev_eer_column"]])),
                float(abs(int(new_row["best_epoch"]) - int(old_row["best_epoch"]))),
                float(abs(int(new_row["n_train"]) - int(old_row["n_train"]))),
                float(abs(int(new_row["in_dim"]) - int(old_row["in_dim"])))]
    record(parity_record, "run_experiments (score files of 16 runs against 32 original runs)", scores)
    record(parity_record, "run_experiments (best dev EER, best epoch, sizes in runs.csv)", selection)
