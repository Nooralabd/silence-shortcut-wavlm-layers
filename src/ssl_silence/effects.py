"""The 2x2 train/test design: cells, effects and their bootstrap intervals.

E(a, b) is the seed-mean EER of the model trained in condition a and tested
in condition b; C is the base condition (clean) and T a trimmed condition.

    cc = E(C, C)   ct = E(C, T)   tc = E(T, C)   tt = E(T, T)

    delta_test    = ct - cc      trimming the test audio of the clean model
    delta_train   = tt - ct      also trimming the training audio
    delta_total   = tt - cc      trimming both
    delta_train_c = tc - cc      trimming the training audio only
    delta_test_t  = tt - tc      trimming the test audio of the trimmed model
    interaction   = tt - tc - ct + cc

All intervals are paired: every cell is evaluated on the same resamples.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from . import config as cfg
from . import metrics as M
from .bootstrap import (POOLING_CODES, Resamples, ScoreStore, class_files, class_groups,
                        draw_file_counts, draw_speaker_counts, interval_columns, require_paired,
                        seed_mean_eer, seed_mean_resampled_eer, sequential_counts)

BASE = "clean"
CELLS = ("cc", "ct", "tc", "tt")
# Weights of the cells, in the order in which the terms are added.
EFFECT_WEIGHTS = {
    "delta_test": {"ct": 1, "cc": -1},
    "delta_train": {"tt": 1, "ct": -1},
    "delta_total": {"tt": 1, "cc": -1},
    "delta_train_c": {"tc": 1, "cc": -1},
    "delta_test_t": {"tt": 1, "tc": -1},
    "interaction": {"tt": 1, "tc": -1, "ct": -1, "cc": 1},
}
ATTACK_LAYERS = (3, 4, 5)


def cell_conditions(condition: str, base: str = BASE) -> dict[str, tuple[str, str]]:
    """Cell name -> (training condition, test condition)."""
    if condition == base:
        raise ValueError("the trimmed condition must differ from the base condition")
    return {"cc": (base, base), "ct": (base, condition), "tc": (condition, base),
            "tt": (condition, condition)}


def weighted(values: Mapping[str, object], weights: Mapping[str, int]):
    """Sum of weight * value over the cells, in the order of ``weights``."""
    return sum(w * values[k] for k, w in weights.items())


def cells_table(summary: pd.DataFrame, condition: str, base: str = BASE) -> pd.DataFrame:
    """The four cells per pooling and layer from ``selection.seed_summary``.

    Columns: condition, pooling, layer, cell, train_condition, test_condition,
    n_seeds, eval_eer_mean, eval_eer_sd, itw_eer_mean, itw_eer_sd. Only
    layers with all four cells are returned.
    """
    names = {pair: cell for cell, pair in cell_conditions(condition, base).items()}
    part = summary[[(tr, te) in names for tr, te in zip(summary["condition"],
                                                         summary["test_condition"])]].copy()
    part["cell"] = [names[(tr, te)] for tr, te in zip(part["condition"], part["test_condition"])]
    complete = part.groupby(["pooling", "layer"])["cell"].transform("nunique") == len(CELLS)
    part = part[complete].rename(columns={"condition": "train_condition"})
    part.insert(0, "condition", condition)
    columns = ["condition", "pooling", "layer", "cell", "train_condition", "test_condition",
               "n_seeds", "eval_eer_mean", "eval_eer_sd", "itw_eer_mean", "itw_eer_sd"]
    return part[columns].sort_values(["pooling", "layer", "cell"]).reset_index(drop=True)


def cell_statistics(resamples: Resamples, condition: str, pooling: str, layer: int,
                    test_set: str, base: str = BASE, cells: Sequence[str] = CELLS
                    ) -> dict[str, dict]:
    """Seed-mean EER of ``cells`` (default: all four): observed values and the
    bootstrap values at the file level and at the speaker level (None without
    speakers)."""
    entry = resamples.get(test_set)
    found = {cell: resamples.cell_eer(train, test, pooling, layer, test_set)
             for cell, (train, test) in cell_conditions(condition, base).items()
             if cell in cells}
    return {"observed": {k: v["observed"] for k, v in found.items()},
            "file": {k: v["file"] for k, v in found.items()},
            "spk": (None if entry["spk"] is None else {k: v["spk"] for k, v in found.items()}),
            "n_files": int(entry["labels"].size)}


def effects_ci(resamples: Resamples, condition: str, layers: Sequence[int],
               poolings: Sequence[str] = cfg.POOLINGS, test_sets: Sequence[str] = cfg.TEST_SETS,
               base: str = BASE) -> pd.DataFrame:
    """The six effects per test set, pooling and layer, with paired percentile
    intervals at the file level and at the speaker level.

    Columns: encoder, condition, test_set, pooling, layer, effect, estimate,
    file_ci_low, file_ci_high, file_boot_se, file_excludes_zero, spk_ci_low,
    spk_ci_high, spk_boot_se, spk_excludes_zero, eer_cc, eer_ct, eer_tc,
    eer_tt, n_files, n_boot.
    """
    rows = []
    for test_set in test_sets:
        for pooling in poolings:
            for layer in layers:
                stats = cell_statistics(resamples, condition, pooling, int(layer), test_set, base)
                for effect in cfg.EFFECTS:
                    weights = EFFECT_WEIGHTS[effect]
                    spk = None if stats["spk"] is None else weighted(stats["spk"], weights)
                    rows.append({"encoder": resamples.store.encoder, "condition": condition,
                                 "test_set": test_set, "pooling": pooling, "layer": int(layer),
                                 "effect": effect,
                                 "estimate": weighted(stats["observed"], weights),
                                 **interval_columns("file", weighted(stats["file"], weights)),
                                 **interval_columns("spk", spk),
                                 **{f"eer_{k}": stats["observed"][k] for k in CELLS},
                                 "n_files": stats["n_files"], "n_boot": resamples.n_boot})
    return pd.DataFrame(rows)


MATCHED_CELLS = ("cc", "tt")


def total_effect_ci(resamples: Resamples, condition: str, layers: Sequence[int],
                    poolings: Sequence[str] = cfg.POOLINGS,
                    test_sets: Sequence[str] = cfg.TEST_SETS, base: str = BASE) -> pd.DataFrame:
    """``delta_total`` from the two matched cells only, per test set, pooling and layer.

    For the layers of a condition whose cross-condition cells were not
    scored. The resamples and the arithmetic are those of ``effects_ci``, so
    a row equals the ``delta_total`` row of ``effects_ci`` where both exist.

    Columns: encoder, condition, test_set, pooling, layer, effect, estimate,
    the file- and speaker-level interval columns, eer_cc, eer_tt, n_files,
    n_boot.
    """
    weights = EFFECT_WEIGHTS["delta_total"]
    assert set(weights) == set(MATCHED_CELLS)
    rows = []
    for test_set in test_sets:
        for pooling in poolings:
            for layer in layers:
                stats = cell_statistics(resamples, condition, pooling, int(layer), test_set, base,
                                        cells=MATCHED_CELLS)
                spk = None if stats["spk"] is None else weighted(stats["spk"], weights)
                rows.append({"encoder": resamples.store.encoder, "condition": condition,
                             "test_set": test_set, "pooling": pooling, "layer": int(layer),
                             "effect": "delta_total",
                             "estimate": weighted(stats["observed"], weights),
                             **interval_columns("file", weighted(stats["file"], weights)),
                             **interval_columns("spk", spk),
                             **{f"eer_{k}": stats["observed"][k] for k in MATCHED_CELLS},
                             "n_files": stats["n_files"], "n_boot": resamples.n_boot})
    return pd.DataFrame(rows)


def selected_configs_eer_ci(resamples: Resamples, configs: pd.DataFrame,
                            test_sets: Sequence[str] = cfg.TEST_SETS) -> pd.DataFrame:
    """Seed-mean EER of each configuration in its own condition, with intervals.

    ``configs`` has the columns config, selection, condition, pooling, layer
    (``selection.selected_configs``). Columns: encoder, config, selection,
    condition, pooling, layer, test_set, n_files, eer, file_ci_low,
    file_ci_high, file_boot_se, spk_ci_low, spk_ci_high, spk_boot_se,
    n_speakers_bonafide, n_speakers_spoof, n_boot.
    """
    rows = []
    for test_set in test_sets:
        entry = resamples.get(test_set)
        labels = entry["labels"]
        for _, c in configs.iterrows():
            matrix = resamples.matrix(c["condition"], c["condition"], c["pooling"],
                                      int(c["layer"]), test_set)
            file_columns = interval_columns("file", seed_mean_resampled_eer(matrix, labels, entry["file"]))
            spk_columns = interval_columns("spk", None if entry["spk"] is None else
                                           seed_mean_resampled_eer(matrix, labels, entry["spk"]))
            for columns in (file_columns, spk_columns):
                columns.pop(next(k for k in columns if k.endswith("_excludes_zero")))
            rows.append({"encoder": resamples.store.encoder, "config": c["config"],
                         "selection": c["selection"], "condition": c["condition"],
                         "pooling": c["pooling"], "layer": int(c["layer"]), "test_set": test_set,
                         "n_files": int(labels.size), "eer": seed_mean_eer(matrix, labels),
                         **file_columns, **spk_columns,
                         "n_speakers_bonafide": entry["n_spk"].get("n_speakers_bonafide", np.nan),
                         "n_speakers_spoof": entry["n_spk"].get("n_speakers_spoof", np.nan),
                         "n_boot": resamples.n_boot})
    return pd.DataFrame(rows)


def attack_number(attack: str) -> int:
    """Number of an attack label "A<nn>"."""
    if not (len(attack) >= 2 and attack[0] == "A" and attack[1:].isdigit()):
        raise ValueError(f"attack labels must be 'A<nn>', got {attack!r}")
    return int(attack[1:])


def per_attack_effect_ci(resamples: Resamples, condition: str, attack_map: Mapping[str, str],
                         layers: Sequence[int] = ATTACK_LAYERS,
                         poolings: Sequence[str] = cfg.POOLINGS, test_set: str = "eval",
                         base: str = BASE) -> pd.DataFrame:
    """Trimmed minus clean (matched cells) per attack, with paired intervals.

    The EER of an attack uses its spoof files and all bonafide files. The
    resamples are drawn within that subset by generators seeded
    ``[seed, layer, pooling code, attack number, level]`` (level 0: files, 1:
    speakers), ``n_boot`` draws one after the other.

    Columns: encoder, condition, pooling, layer, attack, n_bonafide, n_spoof,
    n_speakers_bonafide, n_speakers_spoof, eer_clean, eer_trimmed, estimate
    and the file- and speaker-level interval columns, n_boot.
    """
    entry = resamples.get(test_set)
    labels, ids = entry["labels"], entry["ids"]
    attack = np.array([attack_map[str(i)] for i in ids])
    if set(attack[labels == 0]) != {"-"}:
        raise ValueError("the attack map and the labels disagree on bonafide files")
    groups = resamples.groups(test_set)
    rows = []
    for pooling in poolings:
        for layer in layers:
            layer = int(layer)
            base_matrix = resamples.matrix(base, base, pooling, layer, test_set)
            trim_matrix = resamples.matrix(condition, condition, pooling, layer, test_set)
            for a in sorted(set(attack[labels == 1])):
                keep = np.flatnonzero((labels == 0) | (attack == a))
                y = labels[keep]
                ma, mb = base_matrix[:, keep], trim_matrix[:, keep]
                cls_files = class_files(y)
                cls_groups = class_groups(y, groups[keep]) if groups is not None else None
                stream = [resamples.seed, layer, POOLING_CODES[pooling], attack_number(a)]
                file_counts = sequential_counts(np.random.default_rng(stream + [0]),
                                                resamples.n_boot, y,
                                                lambda r: draw_file_counts(r, y, cls_files))
                eer_base, eer_trim = seed_mean_eer(ma, y), seed_mean_eer(mb, y)
                file_boots = (seed_mean_resampled_eer(mb, y, file_counts)
                              - seed_mean_resampled_eer(ma, y, file_counts))
                spk_boots = None
                if cls_groups is not None:
                    spk_counts = sequential_counts(np.random.default_rng(stream + [1]),
                                                   resamples.n_boot, y,
                                                   lambda r: draw_speaker_counts(r, y, cls_groups))
                    spk_boots = (seed_mean_resampled_eer(mb, y, spk_counts)
                                 - seed_mean_resampled_eer(ma, y, spk_counts))
                rows.append({"encoder": resamples.store.encoder, "condition": condition,
                             "pooling": pooling, "layer": layer, "attack": a,
                             "n_bonafide": int((y == 0).sum()), "n_spoof": int((y == 1).sum()),
                             "n_speakers_bonafide": len(cls_groups[0]) if cls_groups else np.nan,
                             "n_speakers_spoof": len(cls_groups[1]) if cls_groups else np.nan,
                             "eer_clean": eer_base, "eer_trimmed": eer_trim,
                             "estimate": eer_trim - eer_base,
                             **interval_columns("file", file_boots),
                             **interval_columns("spk", spk_boots), "n_boot": resamples.n_boot})
    return pd.DataFrame(rows)


def duration_subsets(durations: np.ndarray, window_s: float = 4.0) -> dict[str, np.ndarray]:
    """Boolean masks of the duration subsets: all, at most ``window_s``, longer."""
    return {"all": durations > 0, f"<={window_s:g}s": durations <= window_s,
            f">{window_s:g}s": durations > window_s}


def itw_by_duration(resamples: Resamples, condition: str, duration_map: Mapping[str, float],
                    layers: Sequence[int] = ATTACK_LAYERS, poolings: Sequence[str] = cfg.POOLINGS,
                    window_s: float = 4.0, test_set: str = "itw", base: str = BASE) -> pd.DataFrame:
    """Cells and effects on In-the-Wild split by original duration.

    ``duration_map`` maps audio id to seconds. For every subset, pooling and
    layer: the EER of the four cells per seed and the six effects as per-seed
    differences, each with the mean and the sample standard deviation over
    seeds.

    Columns: encoder, condition, pooling, layer, subset, quantity (eer_cc,
    eer_ct, eer_tc, eer_tt or an effect), n_files, n_bonafide, n_spoof, mean, sd.
    """
    entry = resamples.get(test_set)
    labels, ids = entry["labels"], entry["ids"]
    durations = np.array([duration_map[str(i)] for i in ids])
    rows = []
    for pooling in poolings:
        for layer in layers:
            matrices = {cell: resamples.matrix(train, test, pooling, int(layer), test_set)
                        for cell, (train, test) in cell_conditions(condition, base).items()}
            for subset, mask in duration_subsets(durations, window_s).items():
                per_seed = {cell: np.array([M.eer(labels[mask], m[s][mask])
                                            for s in range(m.shape[0])])
                            for cell, m in matrices.items()}
                quantities = {f"eer_{cell}": per_seed[cell] for cell in CELLS}
                quantities.update({effect: weighted(per_seed, EFFECT_WEIGHTS[effect])
                                   for effect in cfg.EFFECTS})
                for quantity, values in quantities.items():
                    rows.append({"encoder": resamples.store.encoder, "condition": condition,
                                 "pooling": pooling, "layer": int(layer), "subset": subset,
                                 "quantity": quantity, "n_files": int(mask.sum()),
                                 "n_bonafide": int((labels[mask] == 0).sum()),
                                 "n_spoof": int((labels[mask] == 1).sum()),
                                 "mean": float(np.mean(values)),
                                 "sd": float(np.std(values, ddof=1))})
    return pd.DataFrame(rows)


def require_both_classes(labels: np.ndarray, counts: np.ndarray, what: str) -> None:
    """Raise if a resample (row of ``counts``) has no bonafide or no spoof file."""
    for label, name in ((cfg.LABEL_BONAFIDE, "bonafide"), (cfg.LABEL_SPOOF, "spoof")):
        empty = np.flatnonzero(counts[:, labels == label].sum(axis=1) == 0)
        if empty.size:
            raise ValueError(f"{what}: {empty.size} resamples have no {name} file "
                             f"(first: resample {int(empty[0])})")


def itw_by_duration_ci(resamples: Resamples, condition: str, duration_map: Mapping[str, float],
                       layers: Sequence[int] = ATTACK_LAYERS,
                       poolings: Sequence[str] = cfg.POOLINGS, window_s: float = 4.0,
                       test_set: str = "itw", base: str = BASE
                       ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The six effects on In-the-Wild per duration subset, with paired intervals,
    and the difference of every effect between the two subsets.

    The resamples are those of the whole test set (the same as in
    ``effects_ci``, file and speaker level). The EER of a subset is computed
    from the resampled files of that subset, so the number of files per class
    in a subset varies between resamples and a speaker with files in both
    subsets is drawn for both together. No generator of its own. A resample in
    which a subset has no file of one class raises ValueError.

    Returns (by_subset, difference). by_subset, one row per pooling, layer,
    subset and effect: encoder, condition, test_set, pooling, layer, subset,
    effect, estimate, the file- and speaker-level interval columns, n_files,
    n_bonafide, n_spoof, n_boot. The rows of the subset "all" equal the rows
    of ``effects_ci``. difference (longer minus at most ``window_s``), one
    row per pooling, layer and effect: encoder, condition, test_set, pooling,
    layer, effect, subset_long, subset_short, estimate_long, estimate_short,
    estimate, the interval columns, n_boot.
    """
    entry = resamples.get(test_set)
    labels, ids = entry["labels"], entry["ids"]
    durations = np.array([duration_map[str(i)] for i in ids])
    subsets = duration_subsets(durations, window_s)
    _, short, long = subsets
    restricted = {}
    for subset, mask in subsets.items():
        y = labels[mask]
        counts = {level: None if entry[level] is None else entry[level][:, mask]
                  for level in ("file", "spk")}
        for level, c in counts.items():
            if c is not None:
                require_both_classes(y, c, f"{test_set} {subset}, {level} level")
        restricted[subset] = (mask, y, counts)
    fixed = {"encoder": resamples.store.encoder, "condition": condition, "test_set": test_set}
    by_subset, difference = [], []
    for pooling in poolings:
        for layer in layers:
            layer = int(layer)
            matrices = {cell: resamples.matrix(train, test, pooling, layer, test_set)
                        for cell, (train, test) in cell_conditions(condition, base).items()}
            values = {}
            for subset, (mask, y, counts) in restricted.items():
                observed = {cell: seed_mean_eer(m[:, mask], y) for cell, m in matrices.items()}
                boots = {level: None if c is None else
                         {cell: seed_mean_resampled_eer(m[:, mask], y, c)
                          for cell, m in matrices.items()} for level, c in counts.items()}
                for effect in cfg.EFFECTS:
                    weights = EFFECT_WEIGHTS[effect]
                    estimate = weighted(observed, weights)
                    effect_boots = {level: None if b is None else weighted(b, weights)
                                    for level, b in boots.items()}
                    values[(subset, effect)] = (estimate, effect_boots)
                    by_subset.append({**fixed, "pooling": pooling, "layer": layer,
                                      "subset": subset, "effect": effect, "estimate": estimate,
                                      **interval_columns("file", effect_boots["file"]),
                                      **interval_columns("spk", effect_boots["spk"]),
                                      "n_files": int(mask.sum()),
                                      "n_bonafide": int((y == cfg.LABEL_BONAFIDE).sum()),
                                      "n_spoof": int((y == cfg.LABEL_SPOOF).sum()),
                                      "n_boot": resamples.n_boot})
            for effect in cfg.EFFECTS:
                (e_long, b_long), (e_short, b_short) = values[(long, effect)], values[(short, effect)]
                boots = {level: None if b_long[level] is None else b_long[level] - b_short[level]
                         for level in ("file", "spk")}
                difference.append({**fixed, "pooling": pooling, "layer": layer, "effect": effect,
                                   "subset_long": long, "subset_short": short,
                                   "estimate_long": e_long, "estimate_short": e_short,
                                   "estimate": e_long - e_short,
                                   **interval_columns("file", boots["file"]),
                                   **interval_columns("spk", boots["spk"]),
                                   "n_boot": resamples.n_boot})
    return pd.DataFrame(by_subset), pd.DataFrame(difference)


def encoder_differences(resamples_a: Resamples, resamples_b: Resamples, condition: str,
                        layers: Sequence[int], poolings: Sequence[str] = cfg.POOLINGS,
                        test_sets: Sequence[str] = cfg.TEST_SETS, base: str = BASE,
                        cells: Sequence[str] = CELLS) -> pd.DataFrame:
    """Encoder a minus encoder b for the four cells and the six effects, paired.

    Both encoders are evaluated on the same files and the same resamples
    (``bootstrap.require_paired``), so the difference is computed inside every
    resample. ``cells``: the cells that both encoders have for these layers;
    only these cells and the effects that need no other cell are returned
    (MATCHED_CELLS: eer_cc, eer_tt and delta_total).

    Columns: encoder_a, encoder_b, condition, test_set, pooling, layer,
    quantity (eer_cc, eer_ct, eer_tc, eer_tt or an effect), estimate_a,
    estimate_b, estimate, the file- and speaker-level interval columns,
    n_files, n_boot.
    """
    weights = {f"eer_{cell}": {cell: 1} for cell in CELLS if cell in cells}
    weights.update({effect: EFFECT_WEIGHTS[effect] for effect in cfg.EFFECTS
                    if set(EFFECT_WEIGHTS[effect]) <= set(cells)})
    rows = []
    for test_set in test_sets:
        require_paired(resamples_a, resamples_b, test_set)
        for pooling in poolings:
            for layer in layers:
                a = cell_statistics(resamples_a, condition, pooling, int(layer), test_set, base,
                                    cells)
                b = cell_statistics(resamples_b, condition, pooling, int(layer), test_set, base,
                                    cells)
                for quantity, w in weights.items():
                    estimate_a, estimate_b = weighted(a["observed"], w), weighted(b["observed"], w)
                    spk = (None if a["spk"] is None else
                           weighted(a["spk"], w) - weighted(b["spk"], w))
                    rows.append({"encoder_a": resamples_a.store.encoder,
                                 "encoder_b": resamples_b.store.encoder, "condition": condition,
                                 "test_set": test_set, "pooling": pooling, "layer": int(layer),
                                 "quantity": quantity, "estimate_a": estimate_a,
                                 "estimate_b": estimate_b, "estimate": estimate_a - estimate_b,
                                 **interval_columns("file", weighted(a["file"], w)
                                                    - weighted(b["file"], w)),
                                 **interval_columns("spk", spk),
                                 "n_files": a["n_files"], "n_boot": resamples_a.n_boot})
    return pd.DataFrame(rows)
