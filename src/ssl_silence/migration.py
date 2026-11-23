"""Copying the original features and tables to the new root, with manifests.

Nothing under the original folders is written, moved or deleted: files are
read there and copied to the new root (``config.ROOT``). Every copy is
verified by sha256 (``verify.copy_verified``), is kept when it already
exists with the right content, and is listed in a manifest that names the
new path only.

Unpadded feature shards are copied file by file. From the padded shards only
one layer is needed (the layer of the two configurations of the padded
comparison); it is written into new shards with the key names of this
package and compared tensor by tensor (``verify.compare_feature_dirs``).
"""

from __future__ import annotations

import glob
import os
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from . import config as cfg
from . import shards
from .provenance import sha256_file
from .verify import copy_verified, legacy_path

MANIFEST_COLUMNS = ["file", "bytes", "sha256"]


def relative(path: str, root: str) -> str:
    """``path`` relative to ``root``, with forward slashes."""
    return os.path.relpath(path, root).replace(os.sep, "/")


def copy_files(pairs: Sequence[tuple[str, str]], root: str, progress: bool = False) -> pd.DataFrame:
    """Copy (source, target) pairs; every target lies under ``root``.

    Returns the manifest: file (relative to ``root``), bytes, sha256.
    """
    rows = []
    for k, (source, target) in enumerate(pairs, start=1):
        info = copy_verified(source, target)
        rows.append({"file": relative(target, root), **info})
        if progress:
            print(f"[copy] {k}/{len(pairs)} {relative(target, root)}")
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def shard_files(folder: str, split: str) -> list[str]:
    """The shard files of a split in a feature folder, in order."""
    return sorted(glob.glob(os.path.join(folder, f"{split}_shard_*.pt")))


def copy_feature_dir(source_dir: str, target_dir: str, root: str,
                     splits: Sequence[str] = cfg.SPLITS, progress: bool = True) -> pd.DataFrame:
    """Copy the shard files of a feature folder, file by file. Returns the manifest."""
    pairs = []
    for split in splits:
        files = shard_files(source_dir, split)
        if not files:
            raise FileNotFoundError(f"no shards of split {split!r} in the source folder")
        pairs += [(path, os.path.join(target_dir, os.path.basename(path))) for path in files]
    return copy_files(pairs, root, progress)


def padded_payload(old: Mapping[str, Any], keys: Mapping[str, str], layer: int, encoder: str,
                   condition: str, split: str) -> dict[str, Any]:
    """A padded shard of this package that holds one layer of an original shard.

    ``keys`` maps the keys of this package to the original keys. A feature
    tensor (n, L, D) keeps layer ``layer`` only, as (n, 1, D). ``too_short``
    is not stored in the original shards: it is ``n_real`` below the minimum
    input length. The original provenance is kept under ``source``.
    """
    import torch

    payload: dict[str, Any] = {}
    for key, old_key in keys.items():
        value = old[old_key]
        if key in shards.FEATURE_KEYS["padded"]:
            value = value[:, layer - 1:layer, :].clone()
        elif isinstance(value, torch.Tensor):
            value = value.clone()
        else:
            value = list(value)
        payload[key] = value
    if "too_short" not in payload:
        payload["too_short"] = payload["n_real"] < cfg.MIN_SAMPLES
    for key in shards.RANGE_KEYS:
        payload[key] = int(old[key])
    payload["layers"] = [int(layer)]
    payload["meta"] = {"encoder": encoder, "extraction": "padded", "condition": condition,
                       "split": split, "layers": [int(layer)], "source": dict(old.get("meta", {}))}
    return payload


def copy_padded_layer(source_dir: str, target_dir: str, root: str, keys: Mapping[str, str],
                      layer: int, encoder: str, condition: str,
                      splits: Sequence[str] = cfg.SPLITS, progress: bool = True) -> pd.DataFrame:
    """Write one layer of the original padded shards into new shards.

    An existing new shard is kept. Returns the manifest of the new files
    (file, bytes, sha256 of the new file).
    """
    import torch

    rows = []
    for split in splits:
        files = shard_files(source_dir, split)
        if not files:
            raise FileNotFoundError(f"no shards of split {split!r} in the source folder")
        for path in files:
            target = os.path.join(target_dir, os.path.basename(path))
            if not os.path.exists(target):
                old = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
                cfg.makedirs(target_dir)
                shards.write_shard(target, padded_payload(old, keys, layer, encoder, condition, split),
                                   "padded")
            rows.append({"file": relative(target, root), "bytes": int(os.path.getsize(target)),
                         "sha256": sha256_file(target)})
            if progress:
                print(f"[padded layer {layer}] {relative(target, root)}")
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def check_file_list(file_list: pd.DataFrame, feature_dir: str, split: str,
                    extraction: str = "unpadded") -> dict[str, object]:
    """The file list of a split against the files stored in its shards.

    The order of the list defines the shards, so the audio ids, labels and
    system ids must be equal file by file and in the same order. Returns
    split, n_listed, n_stored, ids_equal, labels_equal, system_ids_equal, ok.
    """
    ids, labels, systems = [], [], []
    for shard in shards.split_shards(feature_dir, split, extraction):
        ids += [str(a) for a in shard["audio_ids"]]
        systems += [str(s) for s in shard["system_ids"]]
        labels.append(np.asarray(shard["labels"]).astype(np.int64))
    labels = np.concatenate(labels)
    same_length = len(ids) == len(file_list)
    ids_equal = same_length and ids == [str(a) for a in file_list["audio_id"]]
    labels_equal = same_length and bool(np.array_equal(labels, file_list["label"].to_numpy()))
    systems_equal = same_length and systems == [str(s) for s in file_list["system_id"]]
    return {"split": split, "n_listed": len(file_list), "n_stored": len(ids),
            "ids_equal": bool(ids_equal), "labels_equal": bool(labels_equal),
            "system_ids_equal": bool(systems_equal),
            "ok": bool(ids_equal and labels_equal and systems_equal)}


def shard_provenance(feature_dir: str, splits: Sequence[str] = cfg.SPLITS,
                     keys: Sequence[str] = ("audio_backends", "git_commit", "torch_version",
                                            "transformers_version", "librosa_version", "gpu_name")
                     ) -> pd.DataFrame:
    """What the shards of a folder record about their extraction.

    One row per split and key: split, key, n_shards, values (the distinct
    recorded values joined by " | "; "not recorded" for a shard without the key).
    """
    import torch

    rows = []
    for split in splits:
        found: dict[str, set[str]] = {key: set() for key in keys}
        files = shard_files(feature_dir, split)
        for path in files:
            meta = torch.load(path, map_location="cpu", weights_only=False, mmap=True).get("meta", {})
            for key in keys:
                value = meta.get(key, "not recorded")
                found[key].add("+".join(map(str, value)) if isinstance(value, (list, tuple))
                               else str(value))
        rows += [{"split": split, "key": key, "n_shards": len(files),
                  "values": " | ".join(sorted(found[key]))} for key in keys]
    return pd.DataFrame(rows)


def original_files(legacy: Mapping[str, Any]) -> dict[str, list[tuple[str, str]]]:
    """The original tables and files of the mapping as (source, name under the copy folder).

    A table ``<name>`` becomes ``tables/<name>.csv``; a file entry ``<name>``
    becomes ``files/<name><extension>``; a folder entry (its name ends in
    ``_dir``) contributes every CSV file in it as ``files/<name>/<file name>``.
    The dataset archive is not among them (it has its own place on the root).
    """
    pairs: dict[str, list[tuple[str, str]]] = {"tables": [], "files": []}
    seen: dict[str, str] = {}
    for name, spec in (legacy.get("tables") or {}).items():
        pairs["tables"].append((legacy_path(legacy, spec["file"]), f"tables/{name}.csv"))
    for name, value in (legacy.get("files") or {}).items():
        if name == "asvspoof_zip":
            continue
        source = legacy_path(legacy, value)
        if name.endswith("_dir"):
            for path in sorted(glob.glob(os.path.join(source, "*.csv"))):
                pairs["files"].append((path, f"files/{name}/{os.path.basename(path)}"))
        else:
            extension = os.path.splitext(source)[1]
            pairs["files"].append((source, f"files/{name}{extension}"))
    for group in pairs.values():
        for source, name in group:
            if seen.setdefault(name, source) != source:
                raise ValueError(f"two original files would be copied to {name}")
    return pairs


def copy_original_files(legacy: Mapping[str, Any], target_dir: str, root: str,
                        progress: bool = False) -> pd.DataFrame:
    """Copy the original tables and files of the mapping under ``target_dir``.

    Returns the manifest (file relative to ``root``, bytes, sha256). A source
    that does not exist raises FileNotFoundError before anything is copied.
    """
    groups = original_files(legacy)
    pairs = [(source, os.path.join(target_dir, *name.split("/")))
             for group in groups.values() for source, name in group]
    missing = [relative(target, root) for source, target in pairs if not os.path.exists(source)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} original files of the mapping do not exist; "
                                f"they would be copied to {missing[:5]}")
    return copy_files(pairs, root, progress)


ARCHIVE_MANIFEST = "MANIFEST.csv"
ARCHIVE_STATUS = ("identical", "different", "not in archive")


def archive_hashes(archive: str) -> dict[str, str]:
    """sha256 of every table of the paper archive, by its path in the archive.

    The archive is a zip file of the tables under their original relative
    paths, with ``MANIFEST.csv`` (path, rows, sha256) next to them; the
    paths are taken relative to the folder of the manifest. Raises unless
    every table of the manifest is in the archive with the recorded sha256
    and every CSV file of the archive is in the manifest.
    """
    import hashlib
    import io
    import zipfile

    with zipfile.ZipFile(archive) as zf:
        members = [name for name in zf.namelist() if not name.endswith("/")]
        manifests = [name for name in members if name.split("/")[-1] == ARCHIVE_MANIFEST]
        if len(manifests) != 1:
            raise ValueError(f"the archive must hold one {ARCHIVE_MANIFEST}, found {len(manifests)}")
        prefix = manifests[0][:-len(ARCHIVE_MANIFEST)]
        hashes = {name[len(prefix):]: hashlib.sha256(zf.read(name)).hexdigest()
                  for name in members
                  if name.startswith(prefix) and name.lower().endswith(".csv") and name != manifests[0]}
        manifest = pd.read_csv(io.BytesIO(zf.read(manifests[0])), dtype=str)
    recorded = dict(zip(manifest["path"].str.replace("\\", "/", regex=False), manifest["sha256"]))
    if len(recorded) != len(manifest):
        raise ValueError("the manifest of the archive names a table twice")
    wrong = sorted(path for path in set(recorded) | set(hashes) if recorded.get(path) != hashes.get(path))
    if wrong:
        raise ValueError(f"{len(wrong)} tables of the archive do not agree with its manifest "
                         f"(missing, not listed, or another sha256)")
    return hashes


def compare_with_archive(legacy: Mapping[str, Any], target_dir: str, root: str,
                         archive: str) -> pd.DataFrame:
    """The copied original tables against the tables of the paper archive, by sha256.

    The counterpart of a copy is the table of the archive at the original
    relative path named in the mapping. One row per copied table and file
    of the mapping (the CSV files of a folder entry are per-file tables
    that the archive never holds; they are left out): file (the copy,
    relative to ``root``), sha256, archive_sha256, status ("identical",
    "different" or "not in archive"), same_name_in_archive (tables of the
    archive with the same file name at another path),
    same_content_in_archive (the archive holds this content at another
    path) and ok (False for "different" only). The original paths are not
    in the table.
    """
    hashes = archive_hashes(archive)
    rows = []
    for group in original_files(legacy).values():
        for source, name in group:
            if name.split("/")[1].endswith("_dir"):
                continue
            original = relative(source, legacy["root"])
            copy = os.path.join(target_dir, *name.split("/"))
            digest = sha256_file(copy)
            expected = hashes.get(original)
            others = {path: value for path, value in hashes.items() if path != original}
            status = ("not in archive" if expected is None
                      else "identical" if expected == digest else "different")
            rows.append({"file": relative(copy, root), "sha256": digest,
                         "archive_sha256": expected if expected is not None else "",
                         "status": status,
                         "same_name_in_archive": sum(path.split("/")[-1] == original.split("/")[-1]
                                                     for path in others),
                         "same_content_in_archive": digest in others.values(),
                         "ok": status != "different"})
    return pd.DataFrame(rows, columns=["file", "sha256", "archive_sha256", "status",
                                       "same_name_in_archive", "same_content_in_archive", "ok"])


def other_sampling_rates(tables: Mapping[str, pd.DataFrame],
                         sample_rate: int = cfg.SAMPLE_RATE) -> pd.DataFrame:
    """Files that were not recorded at ``sample_rate``, per table: table,
    n_files, n_other_rate, rates (the other rates found)."""
    rows = []
    for name, table in tables.items():
        other = table[table["sr_orig"] != sample_rate]
        rows.append({"table": name, "n_files": len(table), "n_other_rate": len(other),
                     "rates": " | ".join(str(int(r)) for r in sorted(other["sr_orig"].unique()))})
    return pd.DataFrame(rows)
