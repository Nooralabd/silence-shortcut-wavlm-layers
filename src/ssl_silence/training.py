"""The single training function, run records and the resumable run loop.

``fit`` is the only training loop of the repository. One model is trained per
(encoder, extraction, training condition, layer, pooling, seed); its
checkpoint is selected on the development set of the training condition, and
it scores the development set, the evaluation set and In-the-Wild in every
test condition of the design.

A score set is named ``<split>@<condition>``. A run writes

    <runs_dir>/scores/<run_id>.npz   ``<set>_scores`` (float32, P(spoof)),
                                     ``<set>_labels`` (int64), ``<set>_ids``
                                     per score set, and ``meta`` (JSON)
    <runs_dir>/runs.csv              one row per run; ``encoder_revision`` is
                                     the checkpoint revision recorded in the
                                     feature shards that the run read

No metric is computed here except the development-set EER that selects the
checkpoint (``metrics.eer_roc_corners``); it is never reported.

Early stopping: a check without improvement adds DEV_CHECK_EVERY to a counter
and training stops when the counter reaches PATIENCE_CHECKS *
DEV_CHECK_EVERY. With checks at epoch 1 and every 5 epochs this cannot happen
before epoch 50, so every model trains for the 50 epochs.
"""

from __future__ import annotations

import csv
import json
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from . import config as cfg
from . import provenance, shards
from .classifier import Classifier
from .metrics import eer_roc_corners

RUNS_CSV_FIELDS = [
    "run_id", "encoder", "encoder_revision", "extraction", "frames", "condition", "trim_top_db", "layer", "pooling",
    "seed", "selection_set", "test_conditions", "score_sets", "n_train", "in_dim",
    "best_dev_eer_roc_corners", "best_epoch", "epochs_run", "device", "gpu_name",
    "torch_version", "git_commit", "timestamp", "score_file",
]
SCORED_SPLITS = ("dev", "eval", "itw")


def set_name(split: str, condition: str) -> str:
    """Name of a score set: ``<split>@<condition>``."""
    return f"{split}@{condition}"


# Seeding and loss weights ----------------------------------------------------------

def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch and make cuDNN deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def class_weights(labels: torch.Tensor, device: str | torch.device) -> torch.Tensor:
    """Cross-entropy weights N / (2 N_c), shape (2,)."""
    n_bonafide = (labels == 0).sum().float()
    n_spoof = (labels == 1).sum().float()
    n = n_bonafide + n_spoof
    return torch.tensor([n / (2 * n_bonafide), n / (2 * n_spoof)]).to(device)


# Run description -------------------------------------------------------------------

@dataclass(frozen=True)
class RunSpec:
    """What identifies a run. ``layer`` is 1-based; ``frames`` applies to the
    padded extraction only ("all-frames" or "real-frames")."""

    encoder: str
    extraction: str
    condition: str
    layer: int
    pooling: str
    seed: int
    frames: str | None = None

    def __post_init__(self) -> None:
        cfg.encoder(self.encoder)
        cfg.check_condition(self.condition)
        if self.extraction not in cfg.EXTRACTIONS:
            raise ValueError(f"extraction must be one of {cfg.EXTRACTIONS}")
        if self.pooling not in cfg.POOLINGS:
            raise ValueError(f"pooling must be one of {cfg.POOLINGS}")
        if (self.extraction == "padded") != (self.frames is not None) or \
                (self.frames is not None and self.frames not in shards.FRAME_SETS):
            raise ValueError("frames is 'all-frames' or 'real-frames' for the padded "
                             "extraction and None for the unpadded one")

    @property
    def run_id(self) -> str:
        frames = "" if self.frames is None else f"_{self.frames}"
        return (f"{self.encoder}_{self.extraction}{frames}_{self.condition}_"
                f"L{self.layer:02d}_{self.pooling}_s{self.seed}")

    @property
    def selection_set(self) -> str:
        """The score set that selects the checkpoint: the dev set of the training condition."""
        return set_name("dev", self.condition)


@dataclass
class ScoreSet:
    """Scores of one set: scores (N,) float32 = P(spoof); labels (N,) int64; ids (N,) str."""

    scores: np.ndarray
    labels: np.ndarray
    ids: np.ndarray


@dataclass
class FitResult:
    model: nn.Module
    best_dev_eer: float   # percent, ROC-corner EER (selection only)
    best_epoch: int       # 1-based epoch of the selected checkpoint
    epochs_run: int


@dataclass
class RunResult:
    run_id: str
    best_dev_eer: float
    best_epoch: int
    epochs_run: int
    scores: dict[str, ScoreSet] = field(default_factory=dict)
    score_path: str | None = None


# The training function ---------------------------------------------------------------

def fit(train_x: torch.Tensor, train_y: torch.Tensor, dev_x: torch.Tensor, dev_y: torch.Tensor,
        seed: int, device: str | torch.device) -> FitResult:
    """Train the classifier and return the checkpoint with the lowest development-set EER.

    train_x, dev_x: (N, D) float features; train_y, dev_y: (N,) long, 0 =
    bonafide, 1 = spoof. The EER is checked after epoch 1 and every
    DEV_CHECK_EVERY epochs; on a tie the earlier checkpoint is kept.
    """
    weights = class_weights(train_y, device)

    seed_everything(seed)
    generator = torch.Generator().manual_seed(seed)

    loader = DataLoader(TensorDataset(train_x.to(device), train_y.to(device)),
                        batch_size=cfg.BATCH_SIZE, shuffle=True, generator=generator)

    model = Classifier(train_x.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE,
                                 weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    best_eer, best_state, waited = 100.0, None, 0
    best_epoch, epochs_run = 0, 0
    dev_x_device = dev_x.to(device)
    dev_y_np = dev_y.numpy()

    for epoch in range(cfg.EPOCHS):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss_fn(model(xb), yb).backward()
            optimizer.step()
        scheduler.step()
        epochs_run = epoch + 1

        if epoch == 0 or (epoch + 1) % cfg.DEV_CHECK_EVERY == 0:
            model.eval()
            with torch.no_grad():
                dev_scores = torch.softmax(model(dev_x_device), 1)[:, 1].cpu().numpy()
            dev_eer = eer_roc_corners(dev_y_np, dev_scores)
            if dev_eer < best_eer:
                best_eer = dev_eer
                best_epoch = epoch + 1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                waited = 0
            else:
                waited += cfg.DEV_CHECK_EVERY
                if waited >= cfg.PATIENCE_CHECKS * cfg.DEV_CHECK_EVERY:
                    break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    return FitResult(model, float(best_eer), best_epoch, epochs_run)


def score(model: nn.Module, x: torch.Tensor, device: str | torch.device) -> np.ndarray:
    """P(spoof) of every row of x (N, D), as a float32 array (N,)."""
    with torch.no_grad():
        return torch.softmax(model(x.to(device)), 1)[:, 1].cpu().numpy()


def _ids_array(ids: Sequence[str] | None, n: int) -> np.ndarray:
    if ids is None:
        return np.array([], dtype=str)
    if len(ids) != n:
        raise ValueError(f"{len(ids)} ids for {n} trials")
    return np.asarray([str(i) for i in ids], dtype=str)


def train_and_score(train_x: torch.Tensor, train_y: torch.Tensor, dev_x: torch.Tensor,
                    dev_y: torch.Tensor,
                    score_sets: Mapping[str, tuple[torch.Tensor, torch.Tensor, Sequence[str] | None]],
                    spec: RunSpec, *, dev_ids: Sequence[str] | None = None,
                    runs_dir: str | None = None, device: str | torch.device | None = None,
                    save: bool = True, overwrite: bool = False,
                    encoder_revision: str = "") -> RunResult:
    """Train one model and score the selection set and every other score set.

    dev_x, dev_y, dev_ids are the development set of the training condition;
    it selects the checkpoint and is stored as ``spec.selection_set``.
    ``score_sets`` maps ``<split>@<condition>`` to (features, labels, audio
    ids or None); none of them takes part in the selection.
    ``encoder_revision`` is recorded in the row of the run and has no other
    effect.
    """
    selection = spec.selection_set
    if selection in score_sets:
        raise ValueError(f"{selection} is the selection set; pass it as dev_x, dev_y")
    for name in score_sets:
        split, _, condition = name.partition("@")
        if split not in SCORED_SPLITS or not condition:
            raise ValueError(f"score set {name!r} is not <split>@<condition>")
        cfg.check_condition(condition)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    runs_dir = runs_dir or cfg.runs_dir(spec.encoder, spec.extraction)
    score_path = os.path.join(runs_dir, "scores", f"{spec.run_id}.npz")
    if save and os.path.exists(score_path) and not overwrite:
        raise FileExistsError(f"{score_path} already exists; pass overwrite=True to replace it")

    fitted = fit(train_x, train_y, dev_x, dev_y, spec.seed, device)

    dev_y_np = dev_y.numpy().astype(np.int64)
    sets = {selection: ScoreSet(score(fitted.model, dev_x, device), dev_y_np,
                                _ids_array(dev_ids, len(dev_y_np)))}
    for name, (x, y, ids) in score_sets.items():
        y_np = y.numpy().astype(np.int64)
        sets[name] = ScoreSet(score(fitted.model, x, device), y_np, _ids_array(ids, len(y_np)))

    result = RunResult(spec.run_id, fitted.best_dev_eer, fitted.best_epoch, fitted.epochs_run, sets)
    if save:
        result.score_path = record_run(
            spec, sets, runs_dir=runs_dir, overwrite=overwrite,
            fit_info={"encoder_revision": encoder_revision,
                      "n_train": int(train_x.shape[0]), "in_dim": int(train_x.shape[1]),
                      "best_dev_eer_roc_corners": repr(fitted.best_dev_eer),
                      "best_epoch": fitted.best_epoch, "epochs_run": fitted.epochs_run,
                      "device": str(device), "gpu_name": provenance.gpu_name(device)})
    return result


# Score files and runs.csv --------------------------------------------------------------

def save_scores(path: str, sets: Mapping[str, ScoreSet], meta: Mapping[str, object]) -> None:
    """Write the score sets and the meta dict (JSON) to an .npz file, atomically."""
    arrays = {"meta": np.asarray(json.dumps(dict(meta), sort_keys=True))}
    for name, s in sets.items():
        arrays[f"{name}_scores"] = np.asarray(s.scores, dtype=np.float32)
        arrays[f"{name}_labels"] = np.asarray(s.labels, dtype=np.int64)
        arrays[f"{name}_ids"] = np.asarray(s.ids, dtype=str)
    cfg.makedirs(os.path.dirname(path))
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def load_scores(path: str) -> tuple[dict[str, ScoreSet], dict]:
    """Inverse of ``save_scores``: (score sets, meta)."""
    with np.load(path, allow_pickle=False) as data:
        names = [k[:-len("_scores")] for k in data.files if k.endswith("_scores")]
        sets = {n: ScoreSet(data[f"{n}_scores"], data[f"{n}_labels"], data[f"{n}_ids"])
                for n in names}
        meta = json.loads(str(data["meta"])) if "meta" in data.files else {}
    return sets, meta


def record_run(spec: RunSpec, sets: Mapping[str, ScoreSet], *, runs_dir: str,
               fit_info: Mapping[str, object], overwrite: bool = False) -> str:
    """Save the scores of a run and append its row to runs.csv. Returns the score path.

    A runs.csv that was written with other columns keeps its columns: the
    row is appended with those (the score file always holds every field).
    """
    cfg.require_drive(runs_dir)
    if spec.selection_set not in sets:
        raise KeyError(f"the selection set {spec.selection_set} is missing")
    score_path = os.path.join(runs_dir, "scores", f"{spec.run_id}.npz")
    if os.path.exists(score_path) and not overwrite:
        raise FileExistsError(f"{score_path} already exists; pass overwrite=True to replace it")
    conditions = list(dict.fromkeys(name.partition("@")[2] for name in sets))
    row = {"run_id": spec.run_id, "encoder": spec.encoder, "extraction": spec.extraction,
           "frames": spec.frames or "", "condition": spec.condition,
           "trim_top_db": cfg.top_db_text(spec.condition),
           "layer": spec.layer, "pooling": spec.pooling, "seed": spec.seed,
           "selection_set": spec.selection_set, "test_conditions": "+".join(conditions),
           "score_sets": "+".join(sets), "torch_version": torch.__version__,
           "git_commit": provenance.git_commit(),
           "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "score_file": "/".join(("scores", f"{spec.run_id}.npz"))}
    row.update(fit_info)
    row.setdefault("encoder_revision", "")
    save_scores(score_path, sets, {k: row[k] for k in RUNS_CSV_FIELDS if k != "score_file"})
    path = os.path.join(runs_dir, "runs.csv")
    new = not os.path.exists(path)
    fields = RUNS_CSV_FIELDS
    if not new:
        with open(path, newline="", encoding="utf-8") as fh:
            fields = next(csv.reader(fh), None) or RUNS_CSV_FIELDS
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow(row)
    return score_path


def read_runs(runs_dir: str) -> pd.DataFrame:
    """runs.csv as strings; the last row of a run id wins. A column that the
    file does not have (``encoder_revision`` in a file written before it
    existed) is empty."""
    cfg.require_drive(runs_dir)
    runs = pd.read_csv(os.path.join(runs_dir, "runs.csv"), dtype=str, keep_default_na=False)
    runs = runs.drop_duplicates("run_id", keep="last").reset_index(drop=True)
    return runs.reindex(columns=RUNS_CSV_FIELDS, fill_value="")


def finished_run_ids(runs_dir: str) -> set[str]:
    """Run ids that have both a runs.csv row and a score file."""
    if not os.path.exists(os.path.join(runs_dir, "runs.csv")):
        return set()
    score_dir = os.path.join(runs_dir, "scores")
    files = set(os.listdir(score_dir)) if os.path.isdir(score_dir) else set()
    return {r for r in read_runs(runs_dir)["run_id"] if f"{r}.npz" in files}


# Planned runs ------------------------------------------------------------------------

@dataclass(frozen=True)
class PlannedRun:
    """A run and the conditions in which its model is scored."""

    spec: RunSpec
    test_conditions: tuple[str, ...]

    @property
    def score_set_names(self) -> tuple[str, ...]:
        """Score sets besides the selection set, in a fixed order."""
        names = []
        for condition in self.test_conditions:
            splits = ("eval", "itw") if condition == self.spec.condition else ("dev", "eval", "itw")
            names += [set_name(split, condition) for split in splits]
        return tuple(names)


def plan_runs(encoder: str, extraction: str,
              seeds: Sequence[int] = cfg.SEEDS, section: str = "experiments") -> list[PlannedRun]:
    """The runs of one encoder and extraction from the experiment configuration.

    Groups that share a training condition are merged: a model scores the
    union of the test conditions listed for its layer and pooling.
    ``section``: "experiments" (the runs of the analysis) or "supplementary"
    (the runs of the supplementary condition, which have their own folder).
    """
    groups = cfg.experiments()[section][encoder][extraction]
    n_layers = cfg.encoder(encoder)["n_layers"]
    merged: dict[tuple, list[str]] = {}
    for group in groups:
        layers = range(1, n_layers + 1) if group["layers"] == "all" else group["layers"]
        for layer in layers:
            for pooling in group.get("poolings", cfg.POOLINGS):
                key = (group["train"], int(layer), pooling, group.get("frames"))
                merged.setdefault(key, [])
                merged[key] += [c for c in group["test"] if c not in merged[key]]
    planned = []
    for (condition, layer, pooling, frames), conditions in merged.items():
        ordered = [condition] + [c for c in conditions if c != condition]
        if condition not in conditions:
            raise ValueError(f"{condition}/L{layer}/{pooling}: the training condition is not scored")
        for seed in seeds:
            planned.append(PlannedRun(RunSpec(encoder, extraction, condition, layer, pooling,
                                              int(seed), frames), tuple(ordered)))
    if len({p.spec.run_id for p in planned}) != len(planned):
        raise ValueError("duplicate run ids")
    return planned


def _duration(seconds: float) -> str:
    return f"{seconds / 3600:.1f} h" if seconds >= 3600 else f"{seconds / 60:.0f} min"


def run_experiments(planned: Sequence[PlannedRun], *, feature_root: str | None = None,
                    cache_root: str | None = None, runs_dir: str | None = None,
                    device: str | torch.device | None = None, max_new_runs: int | None = None,
                    progress: bool = True) -> pd.DataFrame:
    """Train every planned run that is not finished yet.

    All runs share one encoder and extraction. Features are read from
    ``<feature_root>/features/<encoder>/<extraction>/<condition>/`` through a
    local per-layer cache under ``cache_root``. A run is finished when its
    score file and its runs.csv row both exist; a score file without a row is
    an interrupted run and is trained again. The row of a run records the
    checkpoint revision of the feature shards it read (empty when the shards
    record none); shards of more than one revision raise.

    Returns one row per planned run: run_id, status ("skipped", "trained" or
    "pending"), seconds, best_dev_eer_roc_corners, best_epoch.
    """
    keys = {(p.spec.encoder, p.spec.extraction) for p in planned}
    if len(keys) != 1:
        raise ValueError(f"one encoder and extraction per call, got {sorted(keys)}")
    encoder, extraction = keys.pop()
    if len({p.spec.run_id for p in planned}) != len(planned):
        raise ValueError("duplicate run ids")
    runs_dir = runs_dir or cfg.runs_dir(encoder, extraction, feature_root)
    cache_root = cache_root or cfg.LOCAL_CACHE
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    done = finished_run_ids(runs_dir)
    unfinished = [p for p in planned if p.spec.run_id not in done]
    todo = unfinished if max_new_runs is None else unfinished[:max_new_runs]
    status = {p.spec.run_id: {"run_id": p.spec.run_id,
                              "status": "skipped" if p.spec.run_id in done else "pending",
                              "seconds": float("nan"), "best_dev_eer_roc_corners": float("nan"),
                              "best_epoch": 0} for p in planned}
    if progress:
        print(f"{len(planned)} runs | finished {len(planned) - len(unfinished)} | "
              f"to train now {len(todo)}")

    cached: set[tuple[str, str]] = set()
    recorded: dict[tuple[str, str], list[str]] = {}   # checkpoint revisions in the shards of a set
    loaded_key, data = None, {}
    times: list[float] = []

    def cache_dir(condition: str) -> str:
        return os.path.join(cache_root, encoder, extraction, condition)

    for k, run in enumerate(todo):
        spec = run.spec
        needed = {(spec.condition, "train"), (spec.condition, "dev")}
        needed |= {tuple(reversed(name.split("@"))) for name in run.score_set_names}
        for condition, split in sorted(needed):
            if (condition, split) not in cached:
                shards.build_layer_cache(cfg.feature_dir(encoder, extraction, condition, feature_root),
                                         split, cache_dir(condition), extraction)
                cached.add((condition, split))
                recorded[(condition, split)] = shards.cache_meta(cache_dir(condition), split)["revision"]
        key = (spec.condition, spec.layer, spec.frames, run.test_conditions)
        if key != loaded_key:
            data = {(condition, split): shards.load_layer_from_cache(
                cache_dir(condition), split, spec.layer, "stats", extraction, spec.frames)
                for condition, split in sorted(needed)}
            loaded_key = key
        revisions = sorted({revision for key in needed for revision in recorded[key]})
        if len(revisions) > 1:
            raise ValueError(f"{spec.run_id}: the feature shards record more than one "
                             f"checkpoint revision: {revisions}")

        def pick(condition: str, split: str):
            x, y, ids = data[(condition, split)]
            if spec.pooling == "mean":  # stats = [mean, std]
                x = x[:, :x.shape[1] // 2].contiguous()
            return x, y, ids

        start = time.time()
        train_x, train_y, _ = pick(spec.condition, "train")
        dev_x, dev_y, dev_ids = pick(spec.condition, "dev")
        score_sets = {name: pick(name.split("@")[1], name.split("@")[0])
                      for name in run.score_set_names}
        result = train_and_score(train_x, train_y, dev_x, dev_y, score_sets, spec,
                                 dev_ids=dev_ids, runs_dir=runs_dir, device=device, overwrite=True,
                                 encoder_revision=revisions[0] if revisions else "")
        times.append(time.time() - start)
        status[spec.run_id].update(status="trained", seconds=times[-1],
                                   best_dev_eer_roc_corners=result.best_dev_eer,
                                   best_epoch=result.best_epoch)
        if progress:
            left = (len(todo) - k - 1) * float(np.mean(times))
            print(f"[{k + 1}/{len(todo)}] {spec.run_id} | best dev EER (ROC corners) "
                  f"{result.best_dev_eer:.3f} % at epoch {result.best_epoch} | "
                  f"{times[-1]:.0f} s | about {_duration(left)} left")
    return pd.DataFrame(list(status.values()))


# Pilot and report of a grid ----------------------------------------------------------------

def pilot_runs(planned: Sequence[PlannedRun]) -> list[PlannedRun]:
    """The pilot of a grid: the planned runs of ``config.PILOT_LAYER``,
    ``PILOT_POOLING`` and ``PILOT_SEED`` in ``PILOT_CONDITIONS``, one per
    condition and in that order. Raises unless every condition has exactly one."""
    pilot = []
    for condition in cfg.PILOT_CONDITIONS:
        found = [p for p in planned
                 if (p.spec.condition, p.spec.layer, p.spec.pooling, p.spec.seed)
                 == (condition, cfg.PILOT_LAYER, cfg.PILOT_POOLING, cfg.PILOT_SEED)]
        if len(found) != 1:
            raise ValueError(f"{len(found)} pilot runs for the condition {condition!r}, expected 1")
        pilot += found
    return pilot


def grid_report(planned: Sequence[PlannedRun], runs_dir: str,
                seconds: Sequence[float] = ()) -> pd.DataFrame:
    """Planned, finished and remaining runs per training condition, with a time estimate.

    ``seconds`` are measured durations of trained runs on this runtime (NaN
    values are ignored); their mean is ``seconds_per_run``, and
    ``hours_to_train`` is that mean times the runs still to train. Both are
    NaN without a measurement. The last row ("all") is the whole grid.
    """
    done = finished_run_ids(runs_dir)
    measured = [float(s) for s in seconds if s == s]
    per_run = float(np.mean(measured)) if measured else float("nan")
    rows = []
    conditions = list(dict.fromkeys(p.spec.condition for p in planned))
    for condition in conditions + ["all"]:
        runs = [p for p in planned if condition in ("all", p.spec.condition)]
        finished = sum(p.spec.run_id in done for p in runs)
        rows.append({"condition": condition, "planned": len(runs), "finished": finished,
                     "to_train": len(runs) - finished, "seconds_per_run": per_run,
                     "hours_to_train": (len(runs) - finished) * per_run / 3600})
    return pd.DataFrame(rows)


# Checks of finished runs without original scores ---------------------------------------------

def score_set_check(planned: Sequence[PlannedRun], runs_dir: str,
                    file_lists: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Every score set of the planned runs against the file lists.

    ``file_lists`` maps a split to its file list (``audio_id`` and ``label``
    in protocol order). One row per run and planned score set: whether the
    set is in the score file, the number of scores and of files, whether the
    ids are those of the file list in the same order, whether the labels are
    equal, and whether every score is a finite probability. A score set that
    was not planned is a failing row. No metric is computed.
    """
    reference = {split: (np.asarray([str(a) for a in items["audio_id"]], dtype=str),
                         items["label"].to_numpy().astype(np.int64))
                 for split, items in file_lists.items()}
    rows = []
    for run in planned:
        spec = run.spec
        path = os.path.join(runs_dir, "scores", f"{spec.run_id}.npz")
        sets = load_scores(path)[0] if os.path.exists(path) else {}
        expected = (spec.selection_set,) + run.score_set_names
        for name in expected + tuple(n for n in sets if n not in expected):
            ids, labels = reference[name.partition("@")[0]]
            row = {"run_id": spec.run_id, "score_set": name, "planned": name in expected,
                   "present": name in sets, "n_scores": 0, "n_files": len(ids),
                   "same_ids_in_order": False, "same_labels": False, "valid_scores": False}
            if name in sets:
                found = sets[name]
                row["n_scores"] = len(found.scores)
                row["same_ids_in_order"] = bool(np.array_equal(found.ids, ids))
                row["same_labels"] = bool(np.array_equal(found.labels, labels))
                row["valid_scores"] = bool(
                    len(found.scores) == len(found.labels) == len(found.ids)
                    and np.isfinite(found.scores).all()
                    and (found.scores >= 0).all() and (found.scores <= 1).all())
            row["ok"] = bool(row["planned"] and row["present"] and row["same_ids_in_order"]
                             and row["same_labels"] and row["valid_scores"] and len(ids) > 0)
            rows.append(row)
    return pd.DataFrame(rows)


RUN_RECORD_COLUMNS = ["run_id", "finished", "encoder", "encoder_revision", "git_commit",
                      "epochs_run", "best_epoch", "n_train", "in_dim"]


def run_record_check(planned: Sequence[PlannedRun], runs_dir: str, n_train: int) -> pd.DataFrame:
    """The runs.csv row and the score-file meta of every planned run.

    One row per planned run. ``ok`` needs all of: the run is finished (row
    and score file); the row names the run (encoder, extraction, frames,
    condition, layer, pooling, seed, selection set and score sets as
    planned); ``encoder_revision`` is the revision of the encoder registry;
    ``git_commit`` is a commit without uncommitted changes; the model was
    trained for ``config.EPOCHS`` epochs and its checkpoint is from an epoch
    at which the development set is checked; ``n_train`` training files and
    the input size of the pooling; a finite development-set EER; and the
    meta of the score file equals the row.
    """
    have_csv = os.path.exists(os.path.join(runs_dir, "runs.csv"))
    runs = read_runs(runs_dir).set_index("run_id", drop=False) if have_csv else None
    finished = finished_run_ids(runs_dir)
    checked_epochs = {1} | set(range(cfg.DEV_CHECK_EVERY, cfg.EPOCHS + 1, cfg.DEV_CHECK_EVERY))
    rows = []
    for run in planned:
        spec = run.spec
        entry = cfg.encoder(spec.encoder)
        row = {key: "" for key in RUN_RECORD_COLUMNS}
        row.update(run_id=spec.run_id, finished=spec.run_id in finished, names_the_run=False,
                   revision_of_the_registry=False, clean_commit=False, trained_as_planned=False,
                   meta_equals_row=False)
        if row["finished"]:
            record = runs.loc[spec.run_id]
            row.update({key: record[key] for key in RUN_RECORD_COLUMNS[2:]})
            wanted = {"encoder": spec.encoder, "extraction": spec.extraction,
                      "frames": spec.frames or "", "condition": spec.condition,
                      "trim_top_db": cfg.top_db_text(spec.condition),
                      "layer": str(spec.layer), "pooling": spec.pooling, "seed": str(spec.seed),
                      "selection_set": spec.selection_set,
                      "test_conditions": "+".join(run.test_conditions),
                      "score_sets": "+".join((spec.selection_set,) + run.score_set_names),
                      "score_file": f"scores/{spec.run_id}.npz"}
            row["names_the_run"] = all(record[key] == value for key, value in wanted.items())
            row["revision_of_the_registry"] = record["encoder_revision"] == str(entry["revision"])
            commit = record["git_commit"]
            row["clean_commit"] = len(commit) == 40 and set(commit) <= set("0123456789abcdef")
            in_dim = entry["hidden_size"] * (2 if spec.pooling == "stats" else 1)
            try:
                dev_eer = float(record["best_dev_eer_roc_corners"])
                row["trained_as_planned"] = bool(
                    int(record["epochs_run"]) == cfg.EPOCHS
                    and int(record["best_epoch"]) in checked_epochs
                    and int(record["n_train"]) == n_train and int(record["in_dim"]) == in_dim
                    and 0.0 <= dev_eer <= 100.0)
            except ValueError:
                row["trained_as_planned"] = False
            meta = load_scores(os.path.join(runs_dir, "scores", f"{spec.run_id}.npz"))[1]
            row["meta_equals_row"] = all(str(meta.get(key)) == record[key]
                                         for key in RUNS_CSV_FIELDS if key != "score_file")
        row["ok"] = bool(row["finished"] and row["names_the_run"]
                         and row["revision_of_the_registry"] and row["clean_commit"]
                         and row["trained_as_planned"] and row["meta_equals_row"])
        rows.append(row)
    return pd.DataFrame(rows)
