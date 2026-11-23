"""Class-stratified bootstrap of the seed-mean EER, at the file and the speaker level.

Every quantity is the official EER per seed, averaged over the seeds, in the
point estimate and inside every resample. Resamples are multiplicities
(counts) over the files of a test set in a fixed reference order, so that
every cell, layer and encoder is evaluated on the same resamples (paired).

    file level     within each class, the files are drawn with replacement
    speaker level  within each class, the speakers are drawn with
                   replacement; a drawn speaker contributes all its files of
                   that class

The classes are always drawn in the order bonafide, spoof.
"""

from __future__ import annotations

import os
from typing import Callable, Mapping, Sequence

import numpy as np

from . import config as cfg
from . import metrics as M
from .training import RunSpec, load_scores, set_name

# The file order of every resample: the scores of this configuration, first seed.
REFERENCE = {"condition": "clean", "pooling": "stats", "layer": 4}
POOLING_CODES = {"stats": 0, "mean": 1}


# EER of resamples ------------------------------------------------------------------

def resampled_eer(labels: np.ndarray, scores: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Official EER (%) of every resample: counts (n_boot, N) -> (n_boot,).

    The arithmetic of ``metrics.eer`` on the resampled trial list (stable
    sort of the negated scores, bonafide first among ties); the sort is done
    once and each resample repeats the sorted labels.
    """
    labels = np.asarray(labels)
    is_spoof = labels == cfg.LABEL_SPOOF
    order = np.lexsort((is_spoof, -np.asarray(scores, dtype=np.float64)))
    target_sorted = ~is_spoof[order]
    out = np.empty(counts.shape[0])
    for b in range(counts.shape[0]):
        lab = np.repeat(target_sorted, counts[b][order])
        n = lab.size
        tar_sums = np.cumsum(lab)
        n_tar = int(tar_sums[-1])
        n_non = n - n_tar
        non_sums = n_non - (np.arange(1, n + 1) - tar_sums)
        frr = np.concatenate((np.atleast_1d(0), tar_sums / n_tar))
        far = np.concatenate((np.atleast_1d(1), non_sums / n_non))
        i = np.argmin(np.abs(frr - far))
        out[b] = np.mean((frr[i], far[i])) * 100
    return out


def seed_mean_eer(matrix: np.ndarray, labels: np.ndarray) -> float:
    """Mean over seeds of the official EER (%). matrix (n_seeds, N)."""
    return float(np.mean([M.eer(labels, matrix[s]) for s in range(matrix.shape[0])]))


def seed_mean_resampled_eer(matrix: np.ndarray, labels: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Mean over seeds of the EER of every resample: (n_boot,)."""
    per_seed = np.stack([resampled_eer(labels, matrix[s], counts) for s in range(matrix.shape[0])])
    return np.mean(per_seed, axis=0)


# Draws -----------------------------------------------------------------------------

def _counts_from_index(index: np.ndarray, n: int) -> np.ndarray:
    return np.bincount(index, minlength=n).astype(np.int16)


def class_files(labels: np.ndarray) -> list[np.ndarray]:
    """File indices of the bonafide and of the spoof class."""
    return [np.flatnonzero(labels == c) for c in (cfg.LABEL_BONAFIDE, cfg.LABEL_SPOOF)]


def class_groups(labels: np.ndarray, groups: np.ndarray) -> list[list[np.ndarray]]:
    """Per class, the file indices of every speaker, speakers in sorted order."""
    groups = np.asarray(groups).astype(str)
    return [[np.flatnonzero((groups == u) & (labels == c)) for u in np.unique(groups[labels == c])]
            for c in (cfg.LABEL_BONAFIDE, cfg.LABEL_SPOOF)]


def draw_file_counts(rng: np.random.Generator, labels: np.ndarray,
                     cls_files: Sequence[np.ndarray] | None = None) -> np.ndarray:
    """One class-stratified file resample as multiplicities (N,) int16."""
    cls_files = class_files(labels) if cls_files is None else cls_files
    index = np.concatenate([f[rng.integers(0, len(f), len(f))] for f in cls_files])
    return _counts_from_index(index, labels.size)


def draw_speaker_counts(rng: np.random.Generator, labels: np.ndarray,
                        cls_groups: Sequence[Sequence[np.ndarray]]) -> np.ndarray:
    """One class-stratified speaker resample as multiplicities (N,) int16."""
    index = np.concatenate([np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))])
                            for g in cls_groups])
    return _counts_from_index(index, labels.size)


def replicate_seeds(test_sets: Sequence[str] = cfg.TEST_SETS, n_boot: int = cfg.N_BOOT,
                    seed: int = cfg.BOOTSTRAP_SEED) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Replicate seeds per test set: (file-level seeds, speaker-level seeds).

    One root generator is consumed for every test set in order: ``n_boot``
    file-level seeds, then ``n_boot`` speaker-level seeds. Replicate b is
    drawn by ``default_rng(seed_b)``.
    """
    rng = np.random.default_rng(seed)
    return {t: (rng.integers(0, 2 ** 31, n_boot), rng.integers(0, 2 ** 31, n_boot))
            for t in test_sets}


def seeded_counts(labels: np.ndarray, seeds: np.ndarray,
                  draw: Callable[[np.random.Generator], np.ndarray]) -> np.ndarray:
    """(n_boot, N) multiplicities; replicate b is ``draw(default_rng(seeds[b]))``."""
    counts = np.zeros((len(seeds), labels.size), dtype=np.int16)
    for b, replicate_seed in enumerate(seeds):
        counts[b] = draw(np.random.default_rng(replicate_seed))
    return counts


def sequential_counts(rng: np.random.Generator, n_boot: int, labels: np.ndarray,
                      draw: Callable[[np.random.Generator], np.ndarray]) -> np.ndarray:
    """(n_boot, N) multiplicities drawn one after the other from one generator."""
    counts = np.zeros((n_boot, labels.size), dtype=np.int16)
    for b in range(n_boot):
        counts[b] = draw(rng)
    return counts


def speakers_per_label(labels: np.ndarray, groups: np.ndarray) -> dict[str, int]:
    """Number of speakers with bonafide files and with spoof files."""
    labels, groups = np.asarray(labels), np.asarray(groups).astype(str)
    return {"n_speakers_bonafide": int(np.unique(groups[labels == 0]).size),
            "n_speakers_spoof": int(np.unique(groups[labels == 1]).size)}


def percentile_interval(values: np.ndarray) -> tuple[float, float, float]:
    """(2.5th percentile, 97.5th percentile, sample standard deviation) of bootstrap values."""
    return (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)),
            float(np.std(values, ddof=1)))


def interval_columns(prefix: str, boots: np.ndarray | None) -> dict[str, object]:
    """``<prefix>_ci_low``, ``_ci_high``, ``_boot_se`` and ``_excludes_zero`` of bootstrap values."""
    if boots is None:
        return {f"{prefix}_ci_low": np.nan, f"{prefix}_ci_high": np.nan,
                f"{prefix}_boot_se": np.nan, f"{prefix}_excludes_zero": None}
    low, high, se = percentile_interval(boots)
    return {f"{prefix}_ci_low": low, f"{prefix}_ci_high": high, f"{prefix}_boot_se": se,
            f"{prefix}_excludes_zero": bool(low > 0 or high < 0)}


# Scores ----------------------------------------------------------------------------

class ScoreStore:
    """Score files of one encoder and extraction; a cell is read from the run
    trained in its training condition, score set ``<split>@<test condition>``.

    ``base_runs_dir``: a second runs folder from which the cell trained and
    tested in the base condition (``REFERENCE["condition"]``) is read; every
    other cell is read from ``runs_dir``."""

    def __init__(self, runs_dir: str, encoder: str = "wavlm-base", extraction: str = "unpadded",
                 seeds: Sequence[int] = cfg.SEEDS, frames: str | None = None,
                 base_runs_dir: str | None = None):
        self.runs_dir, self.encoder, self.extraction = runs_dir, encoder, extraction
        self.seeds, self.frames = tuple(int(s) for s in seeds), frames
        self.base_runs_dir = base_runs_dir
        self._cache: dict[tuple[str, str], dict] = {}

    def run_id(self, condition: str, pooling: str, layer: int, seed: int) -> str:
        return RunSpec(self.encoder, self.extraction, condition, int(layer), pooling, int(seed),
                       self.frames).run_id

    def sets(self, run_id: str, folder: str | None = None) -> dict:
        key = (self.runs_dir if folder is None else folder, run_id)
        if key not in self._cache:
            if len(self._cache) >= 64:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = load_scores(os.path.join(key[0], "scores", f"{run_id}.npz"))[0]
        return self._cache[key]

    def folder_of(self, condition: str, test_condition: str) -> str:
        """The runs folder a cell is read from."""
        base = REFERENCE["condition"]
        if self.base_runs_dir is not None and condition == test_condition == base:
            return self.base_runs_dir
        return self.runs_dir

    def score_set(self, condition: str, test_condition: str, pooling: str, layer: int, seed: int,
                  split: str):
        """Scores of one run in one test condition."""
        name = set_name(split, test_condition)
        sets = self.sets(self.run_id(condition, pooling, layer, seed),
                         self.folder_of(condition, test_condition))
        if name not in sets:
            raise KeyError(f"{self.run_id(condition, pooling, layer, seed)} has no score set {name}")
        return sets[name]

    def cell(self, condition: str, test_condition: str, pooling: str, layer: int, split: str
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Scores of every seed of one cell: (matrix (n_seeds, N) float64, labels, ids)."""
        sets = [self.score_set(condition, test_condition, pooling, layer, s, split)
                for s in self.seeds]
        ids, labels = sets[0].ids, sets[0].labels
        for other in sets[1:]:
            if not (np.array_equal(other.ids, ids) and np.array_equal(other.labels, labels)):
                raise ValueError(f"seeds of {condition}->{test_condition} {pooling}/L{layer} hold "
                                 f"different {split} files")
        return np.stack([x.scores.astype(np.float64) for x in sets]), labels, ids


class Resamples:
    """File- and speaker-level resamples per test set, in the reference file order.

    The replicate seeds are those of ``replicate_seeds`` for config.TEST_SETS
    in that order, whichever test sets are used.
    """

    def __init__(self, store: ScoreStore, speaker_maps: Mapping[str, Mapping[str, str]] | None = None,
                 n_boot: int = cfg.N_BOOT, seed: int = cfg.BOOTSTRAP_SEED,
                 reference: Mapping = REFERENCE):
        self.store, self.n_boot, self.seed = store, n_boot, seed
        self.speaker_maps = speaker_maps or {}
        self.reference = dict(reference)
        self.replicates = replicate_seeds(cfg.TEST_SETS, n_boot, seed)
        self._entries: dict[str, dict] = {}
        self._cells: dict[tuple, dict] = {}

    def reference_files(self, test_set: str) -> tuple[np.ndarray, np.ndarray]:
        """(labels, ids) of a test set in the reference order."""
        ref = self.reference
        reference = self.store.score_set(ref["condition"], ref["condition"], ref["pooling"],
                                         ref["layer"], self.store.seeds[0], test_set)
        return reference.labels, reference.ids

    def groups(self, test_set: str) -> np.ndarray | None:
        """Speaker of every file in the reference order, or None without a speaker map."""
        if test_set not in self.speaker_maps:
            return None
        _, ids = self.reference_files(test_set)
        speakers = self.speaker_maps[test_set]
        try:
            return np.array([speakers[str(i)] for i in ids])
        except KeyError as error:
            raise KeyError(f"file {error} of {test_set} has no speaker in the map") from None

    def get(self, test_set: str) -> dict:
        """labels, ids, file counts, speaker counts (or None) and speaker numbers of a test set."""
        if test_set not in self._entries:
            labels, ids = self.reference_files(test_set)
            file_seeds, speaker_seeds = self.replicates[test_set]
            cls_files = class_files(labels)
            entry = {"labels": labels, "ids": ids, "spk": None, "n_spk": {}, "groups": None,
                     "file": seeded_counts(labels, file_seeds,
                                           lambda r: draw_file_counts(r, labels, cls_files))}
            groups = self.groups(test_set)
            if groups is not None:
                cls_groups = class_groups(labels, groups)
                entry["groups"] = groups
                entry["spk"] = seeded_counts(labels, speaker_seeds,
                                             lambda r: draw_speaker_counts(r, labels, cls_groups))
                entry["n_spk"] = speakers_per_label(labels, groups)
            self._entries[test_set] = entry
        return self._entries[test_set]

    def matrix(self, condition: str, test_condition: str, pooling: str, layer: int,
               test_set: str) -> np.ndarray:
        """(n_seeds, N) scores of a cell in the reference file order."""
        matrix, labels, ids = self.store.cell(condition, test_condition, pooling, layer, test_set)
        ref_labels, ref_ids = self.reference_files(test_set)
        if np.array_equal(ids, ref_ids):
            if not np.array_equal(labels, ref_labels):
                raise ValueError("labels differ from the reference")
            return matrix
        position = {str(i): k for k, i in enumerate(ids)}
        try:
            order = np.array([position[str(i)] for i in ref_ids])
        except KeyError as error:
            raise KeyError(f"file {error} is missing from {condition}->{test_condition}") from None
        if not np.array_equal(labels[order], ref_labels):
            raise ValueError("labels differ from the reference")
        return matrix[:, order]


    def cell_eer(self, condition: str, test_condition: str, pooling: str, layer: int,
                 test_set: str) -> dict:
        """Seed-mean EER of one cell: ``observed``, and ``file`` and ``spk`` (None
        without speakers), its values in every file- and speaker-level resample.

        Kept after the first call, so that every table that needs the cell
        (the effects of each threshold, the difference between two encoders)
        uses the same values without computing them again. The arrays are
        not to be changed in place.
        """
        key = (condition, test_condition, pooling, int(layer), test_set)
        if key not in self._cells:
            entry = self.get(test_set)
            labels = entry["labels"]
            matrix = self.matrix(condition, test_condition, pooling, int(layer), test_set)
            self._cells[key] = {
                "observed": seed_mean_eer(matrix, labels),
                "file": seed_mean_resampled_eer(matrix, labels, entry["file"]),
                "spk": (None if entry["spk"] is None
                        else seed_mean_resampled_eer(matrix, labels, entry["spk"]))}
        return self._cells[key]


def require_paired(a: Resamples, b: Resamples, test_set: str) -> None:
    """Raise unless two sets of resamples (two encoders) are the same resamples
    of the same files of ``test_set``: number of resamples, seed, file order,
    labels, speakers and multiplicities."""
    if (a.n_boot, a.seed) != (b.n_boot, b.seed):
        raise ValueError("the two encoders use a different number of resamples or seed")
    ea, eb = a.get(test_set), b.get(test_set)
    if not np.array_equal(ea["ids"], eb["ids"]):
        raise ValueError(f"the two encoders hold different {test_set} files or file orders")
    if not np.array_equal(ea["labels"], eb["labels"]):
        raise ValueError(f"the two encoders hold different {test_set} labels")
    if (ea["groups"] is None) != (eb["groups"] is None) or (
            ea["groups"] is not None and not np.array_equal(ea["groups"], eb["groups"])):
        raise ValueError(f"the two encoders use different {test_set} speakers")
    for level in ("file", "spk"):
        if ea[level] is not None and not np.array_equal(ea[level], eb[level]):
            raise ValueError(f"the {level}-level resamples of {test_set} differ")
