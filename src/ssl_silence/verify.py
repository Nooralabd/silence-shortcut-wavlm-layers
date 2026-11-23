"""Comparisons of the outputs of this package with the original outputs.

Everything that names an original output (folders, run ids, column names,
labels of conditions and effects) is read from a mapping file that is kept
outside the repository (``<root>/migration/legacy_map.yaml``). The check
tables written here hold the names of this package only.

Mapping file (YAML; folders and files are relative to ``root``)::

    root: <folder of the original outputs>
    forbidden:            # what must never appear in a check table
      values: [...]       # whole cell values and column names, compared exactly
      patterns: [...]     # regular expressions that must not match a whole value
    conditions: {clean: ..., trim40: ..., trim20: ..., trim15: ...}
    features:
      <encoder>:
        unpadded: {<condition>: <folder>}
        padded:
          layer: 4
          keys: {<key of this package>: <original key>}
          folders: {<condition>: <folder>}
    runs:
      <encoder>:
        <extraction>:
          run_id: <format string with encoder, layer, pooling, condition, seed, cross, frames>
          cross: <format string with test_condition; empty for a matched run>
          frames: {<frames of this package>: <original value>}
          folders: [<folder with runs.csv and scores/>, ...]
          matched_sets: {dev: ..., eval: ..., itw: ...}   # sets of a run scored in its own condition
          cross_sets: {dev: ..., eval: ..., itw: ...}     # sets of a run scored in another condition
    tables:
      <name>:
        file: <csv file>
        where: {<original column>: <value or list>}
        combine: {<new column>: {columns: [...], values: {"a|b": <new value>}}}
        columns: {<column of this package>: <original column>}
        values: {<column of this package>: {<original value>: <new value>}}
        constants: {<column of this package>: <value>}
        keys: [<columns of this package that identify a row>]
        tolerance: <number>

A check writes its table first and stops afterwards (``write_check``, then
``require_ok``), so a failed check leaves its evidence.
"""

from __future__ import annotations

import glob
import os
import re
import shutil
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from . import config as cfg
from .provenance import sha256_file, write_table

# Tolerances: a difference equal to the tolerance passes, a larger one fails.
FEATURE_TOLERANCE = 1e-4      # re-extracted features against the stored ones
SCORE_TOLERANCE = 1e-4        # scores of a retrained run against the original run
TABLE_TOLERANCE = 1e-9        # analysis tables (same generator streams)
SECONDS_TOLERANCE = 1e-9      # measured durations
TARGET_TOLERANCE = 1e-6       # acoustic targets
MAP_KEYS = ("root", "forbidden", "conditions")
SEPARATOR = "|"


class CheckFailed(RuntimeError):
    """A check table has failing rows."""


# Mapping file --------------------------------------------------------------------------

def load_map(path: str) -> dict[str, Any]:
    """Read the mapping file and check its outline."""
    import yaml

    with open(path, encoding="utf-8") as fh:
        legacy = yaml.safe_load(fh)
    check_map(legacy)
    return legacy


def check_map(legacy: Mapping[str, Any]) -> None:
    """Raise unless the mapping has its required parts and no empty value."""
    if not isinstance(legacy, Mapping):
        raise ValueError("the mapping file must hold a mapping")
    missing = [key for key in MAP_KEYS if key not in legacy]
    if missing:
        raise KeyError(f"the mapping file lacks {missing}")
    forbidden = legacy["forbidden"]
    if not isinstance(forbidden, Mapping) or set(forbidden) - {"values", "patterns"}:
        raise ValueError("forbidden must hold 'values' and 'patterns' only")
    for pattern in forbidden.get("patterns") or []:
        re.compile(pattern)
    unknown = [c for c in legacy["conditions"] if c not in cfg.conditions()]
    if unknown:
        raise KeyError(f"unknown conditions in the mapping file: {unknown}")
    empty = _empty_leaves(legacy)
    if empty:
        raise ValueError(f"the mapping file has empty values: {empty[:5]}")


def _empty_leaves(node, path: str = "") -> list[str]:
    if isinstance(node, Mapping):
        return [p for key, value in node.items() for p in _empty_leaves(value, f"{path}/{key}")]
    if isinstance(node, (list, tuple)):
        return [p for k, value in enumerate(node) for p in _empty_leaves(value, f"{path}[{k}]")]
    return [path] if node is None or (isinstance(node, str) and not node.strip()) else []


def legacy_path(legacy: Mapping[str, Any], relative: str) -> str:
    """A folder or file of the mapping under its root."""
    return os.path.join(legacy["root"], *str(relative).replace("\\", "/").split("/"))


# Check tables --------------------------------------------------------------------------------

def assert_clean(table: pd.DataFrame, forbidden: Mapping[str, Sequence[str]]) -> None:
    """Raise if a column name or a whole cell value of ``table`` is a forbidden
    value or matches a forbidden pattern completely. Parts of a value are not
    compared, so a column such as trimmed_len_40 is allowed."""
    values = {str(v) for v in forbidden.get("values") or []}
    patterns = [re.compile(p) for p in forbidden.get("patterns") or []]

    def bad(text: str) -> bool:
        return text in values or any(p.fullmatch(text) for p in patterns)

    found = [f"column {c!r}" for c in map(str, table.columns) if bad(c)]
    for column in table.columns:
        if table[column].dtype == object or pd.api.types.is_string_dtype(table[column]):
            cells = {str(v) for v in table[column].dropna().unique()}
            found += [f"{column}: {v!r}" for v in sorted(cells) if bad(v)]
    if found:
        raise ValueError(f"original names in a check table: {found[:5]}"
                         + (f" and {len(found) - 5} more" if len(found) > 5 else ""))


def write_check(table: pd.DataFrame, name: str, checks_dir: str,
                forbidden: Mapping[str, Sequence[str]], **extra) -> str:
    """Write a check table to ``<checks_dir>/<name>.csv`` with its meta file,
    after ``assert_clean``. Returns the path."""
    assert_clean(table, forbidden)
    path = os.path.join(checks_dir, f"{name}.csv")
    write_table(table, path, check=name, **extra)
    return path


def failed_rows(table: pd.DataFrame) -> pd.DataFrame:
    """Rows of a check table whose ``ok`` is not true."""
    if "ok" not in table.columns:
        raise KeyError("a check table needs the column 'ok'")
    return table[~table["ok"].astype(bool)]


def require_ok(table: pd.DataFrame, name: str) -> None:
    """Raise CheckFailed if the check table is empty or has a failing row."""
    if table.empty:
        raise CheckFailed(f"{name}: nothing was compared")
    failed = failed_rows(table)
    if len(failed):
        raise CheckFailed(f"{name}: {len(failed)} of {len(table)} rows failed; first:\n"
                          f"{failed.head(5).to_string(index=False)}")


def summarise_checks(checks: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per check: check, n_rows, n_failed, max_abs_diff (where the table has it), ok."""
    rows = []
    for name, table in checks.items():
        n_failed = len(failed_rows(table)) if len(table) else 0
        largest = (float(pd.to_numeric(table["max_abs_diff"], errors="coerce").max())
                   if "max_abs_diff" in table.columns and len(table) else float("nan"))
        rows.append({"check": name, "n_rows": len(table), "n_failed": n_failed,
                     "max_abs_diff": largest, "ok": bool(len(table) and n_failed == 0)})
    return pd.DataFrame(rows)


# Files -----------------------------------------------------------------------------------------

def file_hashes(folders: Sequence[str], skip: Sequence[str] = (),
                skip_folders: Sequence[str] = ()) -> dict[str, str]:
    """{path: sha256} of every file under ``folders`` (with their subfolders),
    in sorted order. ``skip``: a file whose name starts with one of these is
    left out; ``skip_folders``: folders that are left out with everything
    under them (both: what the run itself writes)."""
    found = {}
    left_out = {os.path.normpath(folder) for folder in skip_folders}
    for folder in folders:
        for parent, children, names in os.walk(folder):
            children[:] = [child for child in children
                           if os.path.normpath(os.path.join(parent, child)) not in left_out]
            if os.path.normpath(parent) in left_out:
                continue
            for name in names:
                if not name.startswith(tuple(skip)):
                    found[os.path.join(parent, name)] = None
    return {file: sha256_file(file) for file in sorted(found)}


def unchanged_check(before: Mapping[str, str], folders: Sequence[str],
                    skip: Sequence[str] = (), skip_folders: Sequence[str] = ()) -> pd.DataFrame:
    """Every file of ``before`` (``file_hashes`` at the start of a run) against
    the same folders now: one row per file with ``sha256_at_start``,
    ``sha256_at_end`` and ``status`` (unchanged, changed, removed, or new for
    a file that did not exist at the start). Only an unchanged file is ok."""
    now = file_hashes(folders, skip, skip_folders)
    rows = []
    for file in sorted(set(before) | set(now)):
        start, end = before.get(file, ""), now.get(file, "")
        status = ("new" if not start else "removed" if not end
                  else "unchanged" if start == end else "changed")
        rows.append({"file": file, "sha256_at_start": start, "sha256_at_end": end,
                     "status": status, "ok": status == "unchanged"})
    return pd.DataFrame(rows, columns=["file", "sha256_at_start", "sha256_at_end", "status", "ok"])


def copy_verified(source: str, target: str) -> dict[str, object]:
    """Copy a file and compare the sha256 of source and copy.

    An existing target with the sha256 of the source is kept. Returns bytes
    and sha256; raises OSError if the copy differs from the source.
    """
    digest = sha256_file(source)
    if not (os.path.exists(target) and sha256_file(target) == digest):
        cfg.makedirs(os.path.dirname(target))
        tmp = target + ".tmp"
        shutil.copyfile(source, tmp)
        os.replace(tmp, target)
        if sha256_file(target) != digest:
            raise OSError(f"the copy {target} differs from its source")
    return {"bytes": int(os.path.getsize(target)), "sha256": digest}


# Tensors and shards -----------------------------------------------------------------------------

def _array(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def compare_tensors(new, old) -> dict[str, object]:
    """Two arrays (or lists of strings) side by side.

    Returns n (first dimension), same_shape, same_dtype, identical (same
    shape, type and values, NaN equal to NaN) and max_abs_diff (NaN when the
    shapes differ or the values are not numbers).
    """
    a, b = _array(new), _array(old)
    same_shape, same_dtype = a.shape == b.shape, a.dtype == b.dtype
    numeric = a.dtype.kind in "biuf" and b.dtype.kind in "biuf"
    largest = float("nan")
    if same_shape and numeric:
        x, y = a.astype(np.float64), b.astype(np.float64)
        both_nan = np.isnan(x) & np.isnan(y)
        largest = float(np.max(np.abs(np.where(both_nan, 0.0, x - y)), initial=0.0))
    if same_shape and numeric:
        equal = bool(np.array_equal(a, b, equal_nan=a.dtype.kind == "f" and b.dtype.kind == "f"))
    else:
        equal = bool(same_shape and np.array_equal(a.astype(str), b.astype(str)))
    return {"n": int(a.shape[0]) if a.ndim else 1, "same_shape": same_shape,
            "same_dtype": same_dtype, "identical": bool(equal and same_dtype),
            "max_abs_diff": largest}


def legacy_shards(folder: str, split: str) -> list[dict[str, Any]]:
    """Every shard file of a split in an original feature folder, in file order, memory-mapped."""
    import torch

    paths = sorted(glob.glob(os.path.join(folder, f"{split}_shard_*.pt")))
    if not paths:
        raise FileNotFoundError(f"no shards of split {split!r} in the original folder")
    return [torch.load(p, map_location="cpu", weights_only=False, mmap=True) for p in paths]


def compare_shards(new: Mapping[str, Any], old: Mapping[str, Any], keys: Mapping[str, str],
                   old_layer: int | None = None) -> list[dict[str, object]]:
    """One shard of this package against an original shard, key by key.

    ``keys`` maps a key of this package to the original key. With
    ``old_layer`` a feature tensor of ``new`` holds that layer only
    ((n, 1, D), ``layers`` = [old_layer]) and is compared with that layer of
    the original tensor ((n, L, D), layers 1 ... L).
    """
    rows = []
    for key, old_key in keys.items():
        if key not in new or old_key not in old:
            raise KeyError(f"key {key!r} is missing from a shard")
        a, b = new[key], old[old_key]
        if old_layer is not None and _array(b).ndim == 3:
            stored = list(new.get("layers") or [])
            if stored != [old_layer]:
                raise ValueError(f"the shard stores layers {stored}, expected [{old_layer}]")
            b = b[:, old_layer - 1:old_layer, :]
        rows.append({"key": key, **compare_tensors(a, b)})
    return rows


def compare_feature_dirs(new_dir: str, old_dir: str, keys: Mapping[str, str] | None = None,
                         splits: Sequence[str] = cfg.SPLITS, extraction: str = "unpadded",
                         old_layer: int | None = None,
                         expected_sizes: Mapping[str, int] | None = None) -> pd.DataFrame:
    """The shards of a feature folder of this package against an original folder.

    ``keys`` defaults to every key of the shard schema under its own name.
    One row per split and key: split, key, n_shards, n_files, n_expected,
    identical (in every shard), max_abs_diff, ok (identical and the number
    of files is the expected one).
    """
    from . import shards

    if keys is None:
        names = (shards.FEATURE_KEYS[extraction] + shards.PER_FILE_KEYS[extraction]
                 + shards.LIST_KEYS)
        keys = {name: name for name in names}
    sizes = dict(cfg.ASVSPOOF_SPLIT_SIZES, itw=cfg.ITW_SIZE) if expected_sizes is None \
        else dict(expected_sizes)
    rows = []
    for split in splits:
        new = shards.split_shards(new_dir, split, extraction)
        old = legacy_shards(old_dir, split)
        if len(new) != len(old):
            raise ValueError(f"{split}: {len(new)} shards against {len(old)} original shards")
        per_key: dict[str, list[dict]] = {key: [] for key in keys}
        for a, b in zip(new, old):
            for row in compare_shards(a, b, keys, old_layer):
                per_key[row["key"]].append(row)
        for key, parts in per_key.items():
            n_files = sum(p["n"] for p in parts)
            identical = all(p["identical"] for p in parts)
            diffs = [p["max_abs_diff"] for p in parts if not np.isnan(p["max_abs_diff"])]
            rows.append({"split": split, "key": key, "n_shards": len(parts), "n_files": n_files,
                         "n_expected": sizes[split], "identical": identical,
                         "max_abs_diff": max(diffs) if diffs else float("nan"),
                         "ok": bool(identical and n_files == sizes[split])})
    return pd.DataFrame(rows)


def compare_reextraction(new: Mapping[str, Any], stored: Mapping[str, Any],
                         float_keys: Sequence[str], exact_keys: Sequence[str],
                         tolerance: float = FEATURE_TOLERANCE) -> pd.DataFrame:
    """Re-extracted features of some files against the stored features of the same files.

    One row per key: key, n, max_abs_diff, exactly_zero, tolerance, ok. A key
    of ``float_keys`` passes up to ``tolerance``; a key of ``exact_keys``
    (lengths, flags, labels, ids) must be identical.
    """
    rows = []
    for key in list(float_keys) + list(exact_keys):
        result = compare_tensors(new[key], stored[key])
        exact = key in exact_keys
        ok = result["identical"] if exact else bool(
            result["same_shape"] and result["max_abs_diff"] <= tolerance)
        rows.append({"key": key, "n": result["n"], "max_abs_diff": result["max_abs_diff"],
                     "exactly_zero": bool(result["identical"]
                                          or (result["same_shape"] and result["max_abs_diff"] == 0.0)),
                     "tolerance": 0.0 if exact else tolerance, "ok": bool(ok)})
    return pd.DataFrame(rows)


# Scores ---------------------------------------------------------------------------------------------

def legacy_run(legacy: Mapping[str, Any], run: Mapping[str, object], test_condition: str
               ) -> tuple[str, bool]:
    """(original run id, matched) of the original run that holds the scores of
    ``run`` (a row of runs.csv of this package) in ``test_condition``."""
    spec = legacy["runs"][run["encoder"]][run["extraction"]]
    names = legacy["conditions"]
    matched = test_condition == run["condition"]
    cross = "" if matched else spec["cross"].format(test_condition=names[test_condition])
    frames = (spec.get("frames") or {}).get(run.get("frames") or "", "")
    run_id = spec["run_id"].format(encoder=run["encoder"], layer=int(run["layer"]),
                                   pooling=run["pooling"], condition=names[run["condition"]],
                                   seed=int(run["seed"]), cross=cross, frames=frames)
    return run_id, matched


def legacy_score_file(legacy: Mapping[str, Any], run: Mapping[str, object], run_id: str
                      ) -> tuple[str, str]:
    """(score file, folder) of an original run: the first folder of the mapping that holds it."""
    spec = legacy["runs"][run["encoder"]][run["extraction"]]
    for folder in spec["folders"]:
        path = os.path.join(legacy_path(legacy, folder), "scores", f"{run_id}.npz")
        if os.path.exists(path):
            return path, legacy_path(legacy, folder)
    raise FileNotFoundError(f"no original score file for the run {run['run_id']}")


def legacy_best_epoch(folder: str, run_id: str) -> int:
    """best_epoch of a run in the runs.csv of an original folder (the last row of the run id)."""
    runs = pd.read_csv(os.path.join(folder, "runs.csv"), dtype=str, keep_default_na=False)
    rows = runs[runs["run_id"] == run_id]
    if rows.empty:
        raise KeyError("the original runs.csv has no row for a run that has a score file")
    return int(rows["best_epoch"].iloc[-1])


def compare_score_sets(new, old, tolerance: float = SCORE_TOLERANCE) -> dict[str, object]:
    """Two score sets (scores, labels, ids): n, ids_equal, labels_equal,
    max_abs_diff (float64; NaN when the ids or labels differ), exactly_zero
    and within_tolerance."""
    ids_equal = new.ids.size == old.ids.size and bool(np.array_equal(new.ids.astype(str),
                                                                    old.ids.astype(str)))
    labels_equal = bool(np.array_equal(new.labels, old.labels))
    comparable = ids_equal and labels_equal and new.scores.size == old.scores.size
    largest = (float(np.max(np.abs(new.scores.astype(np.float64) - old.scores.astype(np.float64)),
                            initial=0.0)) if comparable else float("nan"))
    return {"n": int(new.labels.size), "ids_equal": ids_equal, "labels_equal": labels_equal,
            "max_abs_diff": largest, "exactly_zero": bool(comparable and largest == 0.0),
            "within_tolerance": bool(comparable and largest <= tolerance)}


def compare_run(run: Mapping[str, object], runs_dir: str, legacy: Mapping[str, Any],
                tolerance: float = SCORE_TOLERANCE) -> list[dict[str, object]]:
    """Every score set of one run of this package against the original scores.

    A set ``<split>@<condition>`` in the training condition is read from the
    original run scored in its own condition; a set in another condition
    from the original run scored in that condition. Returns one row per set:
    run_id, score_set, n, ids_equal, labels_equal, max_abs_diff,
    exactly_zero, within_tolerance, best_epoch, best_epoch_equal, ok.
    """
    from .training import load_scores

    spec = legacy["runs"][run["encoder"]][run["extraction"]]
    sets, _ = load_scores(os.path.join(runs_dir, "scores", f"{run['run_id']}.npz"))
    cache: dict[str, tuple[dict, int]] = {}
    rows = []
    for name in sets:
        split, _, test_condition = name.partition("@")
        old_id, matched = legacy_run(legacy, run, test_condition)
        if old_id not in cache:
            path, folder = legacy_score_file(legacy, run, old_id)
            cache[old_id] = (load_scores(path)[0], legacy_best_epoch(folder, old_id))
        old_sets, old_epoch = cache[old_id]
        old_name = spec["matched_sets" if matched else "cross_sets"][split]
        if old_name not in old_sets:
            raise KeyError(f"the original run of {run['run_id']} lacks the set for {name}")
        result = compare_score_sets(sets[name], old_sets[old_name], tolerance)
        epoch_equal = int(run["best_epoch"]) == old_epoch
        rows.append({"run_id": run["run_id"], "score_set": name, **result,
                     "best_epoch": int(run["best_epoch"]), "best_epoch_equal": epoch_equal,
                     "ok": bool(result["within_tolerance"] and epoch_equal)})
    return rows


def compare_runs(runs_dir: str, legacy: Mapping[str, Any], run_ids: Sequence[str] | None = None,
                 tolerance: float = SCORE_TOLERANCE) -> pd.DataFrame:
    """``compare_run`` for the runs of a runs folder (all, or ``run_ids``)."""
    from .training import read_runs

    runs = read_runs(runs_dir)
    if run_ids is not None:
        unknown = sorted(set(run_ids) - set(runs["run_id"]))
        if unknown:
            raise KeyError(f"runs without a row in runs.csv: {unknown[:3]}")
        runs = runs[runs["run_id"].isin(set(run_ids))]
    rows = [row for run in runs.to_dict("records")
            for row in compare_run(run, runs_dir, legacy, tolerance)]
    return pd.DataFrame(rows)


# Tables ------------------------------------------------------------------------------------------------

def rename_legacy(table: pd.DataFrame, spec: Mapping[str, Any]) -> pd.DataFrame:
    """An original table in the column names and labels of this package.

    Steps, in this order: ``where`` keeps rows by original column values;
    ``combine`` builds a column from several original columns (their values
    joined by "|", missing as empty) through a value map, and drops the rows
    without an entry; ``columns`` renames; ``values`` maps labels; ``constants``
    adds columns. Only the columns named by the specification are returned.
    """
    part = table
    for column, wanted in (spec.get("where") or {}).items():
        wanted = wanted if isinstance(wanted, list) else [wanted]
        part = part[part[column].astype(str).isin([str(w) for w in wanted])]
    out = pd.DataFrame(index=part.index)
    for column, rule in (spec.get("combine") or {}).items():
        joined = part[rule["columns"]].fillna("").astype(str).agg(SEPARATOR.join, axis=1)
        out[column] = joined.map(rule["values"])
    for column, old_column in (spec.get("columns") or {}).items():
        if old_column not in part.columns:
            raise KeyError(f"an original table lacks the column mapped to {column!r}")
        out[column] = part[old_column]
    for column in (spec.get("combine") or {}):
        out = out[out[column].notna()]
    for column, mapping in (spec.get("values") or {}).items():
        out[column] = out[column].map(lambda v: mapping.get(v, mapping.get(str(v), v)))
    for column, value in (spec.get("constants") or {}).items():
        out[column] = value
    return out.reset_index(drop=True)


def _numbers(values: pd.Series) -> np.ndarray | None:
    """The values as float64 (booleans as 0 and 1), or None if they are not numbers."""
    if values.dtype == bool:
        return values.to_numpy(dtype=np.float64)
    text = values.astype(str).str.strip().str.lower()
    if text.isin(["true", "false"]).all():
        return (text == "true").to_numpy(dtype=np.float64)
    converted = pd.to_numeric(values, errors="coerce")
    if converted.isna().to_numpy().sum() > values.isna().to_numpy().sum():
        return None
    return converted.to_numpy(dtype=np.float64)


def compare_tables(new: pd.DataFrame, old: pd.DataFrame, keys: Sequence[str],
                   columns: Sequence[str] | None = None,
                   tolerance: float = TABLE_TOLERANCE) -> pd.DataFrame:
    """Two tables with the same column names, row by row on ``keys``.

    ``columns`` defaults to every column of ``old`` that is not a key. One
    row per column: column, n_new, n_old, n_matched, n_only_new, n_only_old,
    n_different, max_abs_diff, tolerance, ok. Numbers (and booleans) are
    compared up to ``tolerance``, NaN equal to NaN; other values must be
    equal. A row that is in one table only makes every column fail.
    """
    keys = list(keys)
    columns = [c for c in old.columns if c not in keys] if columns is None else list(columns)
    for name, table in (("new", new), ("original", old)):
        lacking = [c for c in keys + columns if c not in table.columns]
        if lacking:
            raise KeyError(f"the {name} table lacks the columns {lacking}")
        if table.duplicated(keys).any():
            raise ValueError(f"the {name} table has repeated keys")
    a = new[keys + columns].copy()
    b = old[keys + columns].copy()
    for key in keys:
        a[key], b[key] = a[key].astype(str), b[key].astype(str)
    merged = a.merge(b, on=keys, how="outer", suffixes=("_new", "_old"), indicator=True)
    both = merged[merged["_merge"] == "both"]
    n_only_new = int((merged["_merge"] == "left_only").sum())
    n_only_old = int((merged["_merge"] == "right_only").sum())
    rows = []
    for column in columns:
        x, y = _numbers(both[f"{column}_new"]), _numbers(both[f"{column}_old"])
        if x is not None and y is not None:
            both_nan = np.isnan(x) & np.isnan(y)
            diff = np.abs(np.where(both_nan, 0.0, x - y))
            diff = np.where(np.isnan(diff), np.inf, diff)            # one side missing
            largest = float(np.max(diff, initial=0.0))
            n_different = int((diff > tolerance).sum())
        else:
            left = both[f"{column}_new"].astype(str).to_numpy()
            right = both[f"{column}_old"].astype(str).to_numpy()
            n_different, largest = int((left != right).sum()), float("nan")
        rows.append({"column": column, "n_new": len(new), "n_old": len(old),
                     "n_matched": len(both), "n_only_new": n_only_new, "n_only_old": n_only_old,
                     "n_different": n_different, "max_abs_diff": largest, "tolerance": tolerance,
                     "ok": bool(n_different == 0 and n_only_new == 0 and n_only_old == 0
                                and len(both) > 0)})
    return pd.DataFrame(rows)


def read_legacy(name: str, legacy: Mapping[str, Any], copies: str | None = None) -> pd.DataFrame:
    """The original table ``name`` of the mapping in the layout of this package.

    The file is read with every number exactly as written and renamed
    (``rename_legacy``). ``copies``: the folder under which
    ``migration.copy_original_files`` copied the original tables; the copy
    ``tables/<name>.csv`` is then read instead of the original file.
    """
    spec = legacy["tables"][name]
    path = (legacy_path(legacy, spec["file"]) if copies is None
            else os.path.join(copies, "tables", f"{name}.csv"))
    return rename_legacy(pd.read_csv(path, float_precision="round_trip"), spec)


def compare_with_legacy(new: pd.DataFrame, name: str, legacy: Mapping[str, Any],
                        copies: str | None = None) -> pd.DataFrame:
    """A table of this package against the original table ``name`` of the mapping.

    Reads the original file with every number exactly as written (the default
    parser of pandas can differ in the last digit), renames it
    (``rename_legacy``) and compares the
    columns of the specification that are not keys, with its tolerance
    (default TABLE_TOLERANCE). The check table gets a leading column ``table``.

    ``copies``: the folder under which ``migration.copy_original_files``
    copied the original tables; the copy ``tables/<name>.csv`` is then read
    instead of the original file.
    """
    spec = legacy["tables"][name]
    old = read_legacy(name, legacy, copies)
    result = compare_tables(new, old, spec["keys"], tolerance=float(spec.get("tolerance",
                                                                             TABLE_TOLERANCE)))
    result.insert(0, "table", name)
    return result
