"""Metrics per run from the saved scores, seed summaries and dev-only selection.

Every reported metric comes from the score files. Layers are selected on the
development set of the training condition only.
"""

from __future__ import annotations

import os
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from . import config as cfg
from . import metrics as M
from .training import load_scores, read_runs

RUN_COLUMNS = ["run_id", "encoder", "extraction", "frames", "condition", "layer", "pooling",
               "seed", "selection_set"]
METRIC_COLUMNS = RUN_COLUMNS + ["score_set", "split", "test_condition", "metric", "value",
                                "n_bonafide", "n_spoof"]
CELL_KEYS = ["encoder", "extraction", "frames", "condition", "test_condition", "pooling", "layer"]
# (split, metric) -> column prefix of the seed summary
TARGETS = {("dev", "eer"): "dev_eer", ("eval", "eer"): "eval_eer",
           ("eval", "tandem_min_tdcf"): "eval_tdcf", ("itw", "eer"): "itw_eer"}
# Reference configurations (condition, pooling, layer), reported for continuity
# besides the dev-selected ones.
REFERENCE_CONFIGS = {"wavlm-base": (("trim40", "mean", 4), ("clean", "stats", 4))}
# The two configurations of the padded comparison: (condition, frames, pooling, layer).
PADDED_CONFIGS = (("trim40", "real-frames", "mean", 4), ("clean", "all-frames", "stats", 4))


def metrics_for_scores(labels: np.ndarray, scores: np.ndarray,
                       asv: M.AsvErrorRates | None = None) -> dict[str, float]:
    """Metrics of one score vector: the official EER and, with ASV error rates,
    the minimum tandem t-DCF."""
    out = {"eer": M.eer(labels, scores)}
    if asv is not None:
        out["tandem_min_tdcf"] = M.tandem_min_tdcf(labels, scores, asv)
    return out


def metrics_per_run(runs_dir: str, asv: Mapping[str, M.AsvErrorRates] | None = None
                    ) -> pd.DataFrame:
    """One row per run x score set x metric, computed from the score files of runs.csv.

    ``asv`` maps a split ("dev", "eval") to the error rates of the official
    ASV system; the tandem t-DCF is computed for the score sets of those
    splits. Columns: METRIC_COLUMNS.
    """
    asv = asv or {}
    rows = []
    for _, run in read_runs(runs_dir).iterrows():
        sets, _ = load_scores(os.path.join(runs_dir, run["score_file"]))
        for name, s in sets.items():
            split, _, test_condition = name.partition("@")
            for metric, value in metrics_for_scores(s.labels, s.scores, asv.get(split)).items():
                row = {c: run[c] for c in RUN_COLUMNS}
                row.update(score_set=name, split=split, test_condition=test_condition,
                           metric=metric, value=value, n_bonafide=int((s.labels == 0).sum()),
                           n_spoof=int((s.labels == 1).sum()))
                rows.append(row)
    return pd.DataFrame(rows, columns=METRIC_COLUMNS)


def load_asv(asv_dir: str) -> dict[str, M.AsvErrorRates]:
    """ASV error rates of the splits whose official score file is in ``asv_dir``."""
    out = {}
    for split, name in cfg.ASV_SCORE_FILENAMES.items():
        path = os.path.join(asv_dir, name)
        if os.path.exists(path):
            out[split] = M.load_asv_error_rates(path)
    return out


def seed_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Mean and sample standard deviation over seeds for every train/test cell.

    One row per (encoder, extraction, frames, condition, test_condition,
    pooling, layer) with n_seeds and, for dev_eer, eval_eer, eval_tdcf and
    itw_eer, ``<name>_mean`` and ``<name>_sd`` (ddof = 1; NaN for one seed).
    ``condition`` is the training condition.
    """
    sel = table.copy()
    sel["target"] = [TARGETS.get(k) for k in zip(sel["split"], sel["metric"])]
    sel = sel[sel["target"].notna()]
    sel["layer"] = sel["layer"].astype(int)
    sel["value"] = sel["value"].astype(float)
    per_run = sel.pivot_table(index=CELL_KEYS + ["seed"], columns="target", values="value",
                              aggfunc="first")
    per_run = per_run.reindex(columns=list(TARGETS.values()))
    grouped = per_run.groupby(level=CELL_KEYS)
    out = pd.DataFrame({"n_seeds": grouped.size()})
    for name in TARGETS.values():
        out[f"{name}_mean"] = grouped[name].mean()
        out[f"{name}_sd"] = grouped[name].std(ddof=1)
    return out.reset_index().sort_values(CELL_KEYS).reset_index(drop=True)


def matched_cells(summary: pd.DataFrame) -> pd.DataFrame:
    """Rows of ``seed_summary`` in which the test condition is the training condition."""
    return summary[summary["test_condition"] == summary["condition"]].reset_index(drop=True)


def select_on_dev(summary: pd.DataFrame) -> pd.DataFrame:
    """Layer selection on the development set only.

    For every (encoder, extraction, frames, condition, pooling) the layer
    with the lowest mean development-set EER over seeds in the matched cell.
    A tie is equality of the stored means; the lower layer is selected.
    ``best_in_condition`` marks the best configuration of each condition.
    """
    keys = ["encoder", "extraction", "frames", "condition", "pooling"]
    ordered = matched_cells(summary).sort_values(["dev_eer_mean", "layer"], kind="mergesort")
    best = ordered.groupby(keys, sort=False).head(1).copy()
    winners = best.groupby(["encoder", "extraction", "frames", "condition"], sort=False).head(1).index
    best["best_in_condition"] = best.index.isin(winners)
    return best.sort_values(keys).reset_index(drop=True)


def selected_configs(summary: pd.DataFrame, encoder: str,
                     conditions: Sequence[str] = ("clean", "trim40")) -> pd.DataFrame:
    """The configurations reported for one encoder, with their selection label.

    dev-selected: the layer of ``select_on_dev`` per condition and pooling of
    the unpadded extraction. reference: REFERENCE_CONFIGS of the encoder (none
    for an encoder without an entry).

    Returns condition, pooling, layer, selection, config.
    """
    part = summary[(summary["encoder"] == encoder) & (summary["extraction"] == "unpadded")
                   & summary["condition"].isin(conditions)]
    rows = {key: {"condition": key[0], "pooling": key[1], "layer": int(key[2]),
                  "selection": "reference"}
            for key in REFERENCE_CONFIGS.get(encoder, ()) if key[0] in conditions}
    for _, r in select_on_dev(part).iterrows():
        key = (r["condition"], r["pooling"], int(r["layer"]))
        if key in rows:
            rows[key]["selection"] = "reference + dev-selected"
        else:
            rows[key] = {"condition": key[0], "pooling": key[1], "layer": key[2],
                         "selection": "dev-selected"}
    out = pd.DataFrame(list(rows.values()), columns=["condition", "pooling", "layer", "selection"])
    out["config"] = [f"{r.condition}/{r.pooling}/L{r.layer}" for r in out.itertuples()]
    return out.sort_values(["condition", "pooling", "layer"]).reset_index(drop=True)


def padded_vs_unpadded(summary: pd.DataFrame, encoder: str = "wavlm-base",
                       configs: Sequence[tuple] = PADDED_CONFIGS) -> pd.DataFrame:
    """Padded against unpadded extraction for the configurations of the comparison.

    For every configuration three rows: the padded cell, the unpadded cell
    (same condition, pooling and layer) and their difference (unpadded minus
    padded; standard deviations empty). Columns: config, extraction,
    condition, frames, pooling, layer, n_seeds and the ``<quantity>_mean`` /
    ``_sd`` columns of the summary. A missing cell gives NaN.
    """
    cells = matched_cells(summary[summary["encoder"] == encoder])
    quantities = [c[:-len("_mean")] for c in cells.columns if c.endswith("_mean")]
    rows = []
    for condition, frames, pooling, layer in configs:
        base = (cells["condition"] == condition) & (cells["pooling"] == pooling) \
            & (cells["layer"].astype(int) == int(layer))
        picked = {"padded": cells[base & (cells["extraction"] == "padded")
                                  & (cells["frames"] == frames)],
                  "unpadded": cells[base & (cells["extraction"] == "unpadded")]}
        got = {}
        for extraction, cell in picked.items():
            row = {"config": f"{condition}/{pooling}/L{int(layer)}", "extraction": extraction,
                   "condition": condition, "frames": frames if extraction == "padded" else "",
                   "pooling": pooling, "layer": int(layer),
                   "n_seeds": int(cell["n_seeds"].iloc[0]) if len(cell) else 0}
            for q in quantities:
                row[f"{q}_mean"] = float(cell[f"{q}_mean"].iloc[0]) if len(cell) else np.nan
                row[f"{q}_sd"] = float(cell[f"{q}_sd"].iloc[0]) if len(cell) else np.nan
            got[extraction] = row
            rows.append(row)
        diff = {"config": got["padded"]["config"], "extraction": "unpadded - padded",
                "condition": condition, "frames": "", "pooling": pooling, "layer": int(layer),
                "n_seeds": ""}
        for q in quantities:
            diff[f"{q}_mean"] = got["unpadded"][f"{q}_mean"] - got["padded"][f"{q}_mean"]
            diff[f"{q}_sd"] = np.nan
        rows.append(diff)
    return pd.DataFrame(rows)
