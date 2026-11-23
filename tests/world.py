"""A small synthetic set of runs with saved scores, in the layout of this
repository and, for the parity tests, in the layout of the original code.

Design (as the real one, smaller): encoder wavlm-base, unpadded; training
conditions clean, trim40 and trim20; every model scores its own condition;
clean and trim40 score each other for every layer; clean and trim20 score
each other for CROSS_LAYERS only.
"""

from __future__ import annotations

import os

import numpy as np

from ssl_silence import training as T

ENCODER = "wavlm-base"
SEEDS = (42, 43, 44)
LAYERS = (1, 2, 3, 4)
CROSS_LAYERS = (3, 4)
POOLINGS = ("mean", "stats")
CONDITIONS = ("clean", "trim40", "trim20")
SPLITS = {"dev": 150, "eval": 240, "itw": 200}
ATTACKS = ("A07", "A08", "A09")
# Conditions of this repository -> condition names of the original code.
OLD_CONDITIONS = {"clean": "clean", "trim40": "trimmed", "trim20": "trimmed20"}


def labels_of(split: str) -> np.ndarray:
    rng = np.random.default_rng({"dev": 1, "eval": 2, "itw": 3}[split])
    n = SPLITS[split]
    p_spoof = 0.4 if split == "itw" else 0.75
    y = (rng.random(n) < p_spoof).astype(np.int64)
    y[:4] = [0, 1, 0, 1]
    return y


def ids_of(split: str) -> np.ndarray:
    return np.array([f"{split}_{k:04d}" for k in range(SPLITS[split])])


def speakers_of(split: str) -> dict[str, str]:
    """audio id -> speaker; some speakers have files of both labels."""
    rng = np.random.default_rng({"dev": 11, "eval": 12, "itw": 13}[split])
    names = [f"spk{k:02d}" for k in range(14)]
    return {a: names[int(rng.integers(0, len(names)))] for a in ids_of(split)}


def attacks_of_eval() -> dict[str, str]:
    """audio id -> system id of the evaluation set ("-" for bonafide)."""
    y = labels_of("eval")
    return {a: ("-" if y[k] == 0 else ATTACKS[k % len(ATTACKS)])
            for k, a in enumerate(ids_of("eval"))}


def durations_of_itw() -> dict[str, float]:
    """audio id -> original duration in seconds, on both sides of 4 s."""
    rng = np.random.default_rng(21)
    return {a: float(d) for a, d in zip(ids_of("itw"), rng.uniform(0.6, 9.0, SPLITS["itw"]))}


def test_conditions(condition: str, layer: int) -> tuple[str, ...]:
    """Conditions in which the model of (condition, layer) is scored, its own first."""
    if condition == "clean":
        return ("clean", "trim40") + (("trim20",) if layer in CROSS_LAYERS else ())
    if condition == "trim40":
        return ("trim40", "clean")
    return ("trim20",) + (("clean",) if layer in CROSS_LAYERS else ())


def scores_of(condition: str, test_condition: str, pooling: str, layer: int, seed: int,
              split: str, encoder: str = ENCODER) -> np.ndarray:
    """Deterministic scores (float32): better with depth, worse across conditions,
    with ties (rounded) in one pooling. The second encoder has other scores
    and another order of the layers."""
    key = [CONDITIONS.index(condition), CONDITIONS.index(test_condition),
           POOLINGS.index(pooling), layer, seed, list(SPLITS).index(split)]
    key += [] if encoder == ENCODER else [1]
    rng = np.random.default_rng(key)
    y = labels_of(split)
    separation = 0.6 + 0.35 * layer - (0.5 if condition != test_condition else 0.0) \
        - (0.3 if split == "itw" else 0.0)
    if encoder != ENCODER:
        separation += 0.5 * (layer % 2) - 0.2
    logits = rng.normal(0.0, 1.0, y.size) + separation * y
    scores = 1.0 / (1.0 + np.exp(-logits))
    if pooling == "mean":
        scores = np.round(scores, 2)
    return scores.astype(np.float32)


def planned_runs(encoder: str = ENCODER) -> list[T.PlannedRun]:
    return [T.PlannedRun(T.RunSpec(encoder, "unpadded", condition, layer, pooling, seed),
                         test_conditions(condition, layer))
            for condition in CONDITIONS for layer in LAYERS for pooling in POOLINGS
            for seed in SEEDS]


def write_runs(runs_dir: str, encoder: str = ENCODER, reverse_files: bool = False) -> str:
    """Write the score files and runs.csv of the world in the layout of this
    repository. ``reverse_files`` stores every score set in the reverse file order."""
    order = slice(None, None, -1) if reverse_files else slice(None)
    for run in planned_runs(encoder):
        spec = run.spec
        sets = {}
        for name in (spec.selection_set,) + run.score_set_names:
            split, _, test_condition = name.partition("@")
            sets[name] = T.ScoreSet(scores_of(spec.condition, test_condition, spec.pooling,
                                              spec.layer, spec.seed, split, encoder)[order],
                                    labels_of(split)[order], ids_of(split)[order])
        T.record_run(spec, sets, runs_dir=runs_dir, overwrite=True,
                     fit_info={"n_train": 0, "in_dim": 0, "best_dev_eer_roc_corners": "nan",
                               "best_epoch": 1, "epochs_run": 50, "device": "cpu",
                               "gpu_name": "cpu"})
    return runs_dir


def write_asv_scores(folder: str) -> str:
    """Write synthetic official-format ASV score files for dev and eval."""
    from ssl_silence import config as cfg

    os.makedirs(folder, exist_ok=True)
    rng = np.random.default_rng(31)
    for name in cfg.ASV_SCORE_FILENAMES.values():
        with open(os.path.join(folder, name), "w") as fh:
            for key, mean, n in (("target", 3.0, 300), ("nontarget", 0.0, 200), ("spoof", 2.0, 400)):
                for value in rng.normal(mean, 1.0, n):
                    fh.write(f"x {key} {float(value)!r}\n")
    return folder


def write_original_runs(train_module, feature_set: str, main_dir: str, threshold_dir: str) -> None:
    """Write the same scores in the layout of the original code.

    Grid runs (scored in their training condition) and cross-condition runs
    (same scores for the training condition, plus the other condition). The
    clean and 40 dB runs go to ``main_dir``; everything that involves the
    20 dB condition goes to ``threshold_dir``.
    """
    O = train_module

    def score_set(condition, test_condition, pooling, layer, seed, split):
        return O.ScoreSet(scores_of(condition, test_condition, pooling, layer, seed, split),
                          labels_of(split), ids_of(split))

    for condition in CONDITIONS:
        for layer in LAYERS:
            for pooling in POOLINGS:
                for seed in SEEDS:
                    args = (pooling, layer, seed)
                    common = dict(encoder=ENCODER, layer=layer, pooling=pooling, mask="none",
                                  condition=OLD_CONDITIONS[condition], seed=seed,
                                  feature_set=feature_set)
                    own = {split: score_set(condition, condition, *args, split) for split in SPLITS}
                    target = threshold_dir if condition == "trim20" else main_dir
                    O.record_run(O.RunSpec(**common), own, results_dir=target, overwrite=True,
                                 extra={"best_epoch": 1})
                    for other in test_conditions(condition, layer)[1:]:
                        sets = {"dev": own["dev"],
                                "eval": score_set(condition, other, *args, "eval"),
                                "itw": score_set(condition, other, *args, "itw"),
                                "dev_tc": score_set(condition, other, *args, "dev"),
                                "eval_tr": own["eval"], "itw_tr": own["itw"]}
                        target = threshold_dir if "trim20" in (condition, other) else main_dir
                        O.record_run(O.RunSpec(**common, test_condition=OLD_CONDITIONS[other]),
                                     sets, results_dir=target, overwrite=True,
                                     extra={"best_epoch": 1})
