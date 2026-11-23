"""File lists of ASVspoof 2019 LA and In-the-Wild.

A file list has one row per file, in protocol order, with the columns
dataset, split, audio_id, path, label (0 = bonafide, 1 = spoof), system_id
and speaker. Its order defines the feature shards.
"""

from __future__ import annotations

import csv
import glob
import os
import subprocess
import zipfile

import numpy as np
import pandas as pd

from . import config as cfg
from .provenance import sha256_file

ITEM_COLUMNS = ["dataset", "split", "audio_id", "path", "label", "system_id", "speaker"]
ASVSPOOF = "asvspoof19"
IN_THE_WILD = "in_the_wild"
# Strings that pandas reads as a missing value from a CSV file.
NA_LIKE = frozenset({"", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan",
                     "1.#IND", "1.#QNAN", "<NA>", "N/A", "NA", "NULL", "NaN", "None", "n/a",
                     "nan", "null"})


# ASVspoof 2019 LA ----------------------------------------------------------------

def protocol_path(split: str, local_dir: str | None = None) -> str:
    """CM protocol file of an ASVspoof 2019 LA split inside the unzipped archive."""
    return os.path.join(local_dir or cfg.LOCAL_ASVSPOOF, "LA", "ASVspoof2019_LA_cm_protocols",
                        cfg.PROTOCOL_FILENAMES[split])


def flac_dir(split: str, local_dir: str | None = None) -> str:
    """Folder of the audio files of an ASVspoof 2019 LA split inside the unzipped archive."""
    return os.path.join(local_dir or cfg.LOCAL_ASVSPOOF, "LA", f"ASVspoof2019_LA_{split}", "flac")


def read_asvspoof_protocol(path: str) -> tuple[list[str], np.ndarray, list[str], list[str]]:
    """Parse an ASVspoof 2019 LA CM protocol file.

    Line format: ``SPEAKER FILE_ID - SYSTEM_ID bonafide|spoof``; SYSTEM_ID is
    "-" for bonafide trials. Lines with fewer than five fields are skipped.

    Returns (audio ids, labels (N,), system ids, speakers) in file order.
    """
    ids, labels, system_ids, speakers = [], [], [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            fields = line.split()
            if len(fields) < 5:
                continue
            speakers.append(fields[0])
            ids.append(fields[1])
            system_ids.append(fields[3])
            labels.append(cfg.LABEL_BONAFIDE if fields[-1] == "bonafide" else cfg.LABEL_SPOOF)
    return ids, np.array(labels), system_ids, speakers


def asvspoof_items(split: str, protocol: str | None = None,
                   audio_dir: str | None = None) -> pd.DataFrame:
    """File list of one ASVspoof 2019 LA split ("train", "dev", "eval"), in protocol order."""
    ids, labels, system_ids, speakers = read_asvspoof_protocol(protocol or protocol_path(split))
    audio_dir = audio_dir or flac_dir(split)
    return pd.DataFrame({
        "dataset": ASVSPOOF, "split": split, "audio_id": ids,
        "path": [os.path.join(audio_dir, i + ".flac") for i in ids],
        "label": labels, "system_id": system_ids, "speaker": speakers})


def ensure_asvspoof(zip_path: str, local_dir: str | None = None) -> None:
    """Unzip LA.zip to the local disk unless the training split is already there."""
    local_dir = local_dir or cfg.LOCAL_ASVSPOOF
    if os.path.exists(flac_dir("train", local_dir)) and os.path.exists(protocol_path("train", local_dir)):
        return
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(local_dir)


# In-the-Wild ---------------------------------------------------------------------

def find_itw_meta(itw_dir: str | None = None) -> str:
    """meta.csv inside the extracted In-the-Wild release (the first in sorted order)."""
    itw_dir = itw_dir or cfg.LOCAL_ITW
    hits = sorted(glob.glob(os.path.join(itw_dir, "**", "meta.csv"), recursive=True))
    if not hits:
        raise FileNotFoundError(f"no meta.csv under {itw_dir}")
    return hits[0]


def itw_items(itw_dir: str | None = None) -> pd.DataFrame:
    """File list of In-the-Wild in meta.csv order.

    meta.csv has the columns file, speaker and label ("bona-fide" or
    "spoof"). The split is "itw"; system_id is "-" for bonafide and "itw"
    for spoof files.
    """
    meta = find_itw_meta(itw_dir)
    root = os.path.dirname(meta)
    files, labels, speakers = [], [], []
    with open(meta, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            files.append(row["file"].strip())
            speakers.append(row["speaker"].strip())
            labels.append(cfg.LABEL_BONAFIDE if row["label"].strip().lower().startswith("bona")
                          else cfg.LABEL_SPOOF)
    labels = np.array(labels)
    return pd.DataFrame({
        "dataset": IN_THE_WILD, "split": "itw", "audio_id": files,
        "path": [os.path.join(root, name) for name in files], "label": labels,
        "system_id": np.where(labels == cfg.LABEL_SPOOF, "itw", "-"), "speaker": speakers})


def itw_url(revision: str | None = cfg.ITW_REVISION) -> str:
    """Download address of the In-the-Wild archive at a pinned revision of its repository.

    Raises without a revision: the archive is never taken from an unpinned branch.
    """
    if not revision or revision in ("main", "master"):
        raise ValueError("the In-the-Wild revision is not pinned (config.ITW_REVISION)")
    return f"https://huggingface.co/datasets/{cfg.ITW_REPOSITORY}/resolve/{revision}/{cfg.ITW_FILENAME}"


def download_itw(zip_path: str, revision: str | None = cfg.ITW_REVISION) -> str:
    """Download the In-the-Wild archive once, from the pinned revision, to ``zip_path``.

    An existing archive is kept and not downloaded again.

    Returns the sha256 of the archive at ``zip_path``.
    """
    if not os.path.exists(zip_path):
        url = itw_url(revision)
        cfg.makedirs(os.path.dirname(zip_path))
        subprocess.run(["wget", "-q", "-O", zip_path + ".part", url], check=True)
        os.replace(zip_path + ".part", zip_path)
    return sha256_file(zip_path)


def ensure_itw(zip_path: str, itw_dir: str | None = None,
               sha256: str | None = cfg.ITW_ZIP_SHA256) -> None:
    """Unzip the stored In-the-Wild archive to the local disk unless it is already there.

    The archive must have the checksum ``sha256``; a different or missing
    checksum raises before anything is unzipped. The stored archive is kept.
    """
    itw_dir = itw_dir or cfg.LOCAL_ITW
    if glob.glob(os.path.join(itw_dir, "**", "meta.csv"), recursive=True):
        return
    if sha256 is None:
        raise ValueError("the sha256 of the In-the-Wild archive is not set (config.ITW_ZIP_SHA256)")
    digest = sha256_file(zip_path)
    if digest != sha256:
        raise ValueError(f"In-the-Wild archive has sha256 {digest}, expected {sha256}")
    os.makedirs(itw_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(itw_dir)


# File lists without the audio ----------------------------------------------------

def archive_file_lists(la_zip: str, itw_zip: str, local_dir: str | None = None
                       ) -> dict[str, pd.DataFrame]:
    """The file lists of all splits, read from the two stored archives.

    Only the protocol files of ASVspoof 2019 LA and the ``meta.csv`` of
    In-the-Wild are unpacked, to ``local_dir`` (default
    ``config.LOCAL_LISTS``), and every list is checked with ``check_items``.
    For a run that needs the ids and labels and no audio: the ``path`` column
    names files that are not unpacked.
    """
    local_dir = local_dir or cfg.LOCAL_LISTS
    la_dir, itw_dir = os.path.join(local_dir, "asvspoof2019_la"), os.path.join(local_dir, "in_the_wild")
    for zip_path, target, wanted in (
            (la_zip, la_dir, lambda name: os.path.basename(name) in cfg.PROTOCOL_FILENAMES.values()
             and "ASVspoof2019_LA_cm_protocols" in name),
            (itw_zip, itw_dir, lambda name: os.path.basename(name) == "meta.csv")):
        with zipfile.ZipFile(zip_path) as archive:
            members = [name for name in archive.namelist() if wanted(name)]
            if not members:
                raise FileNotFoundError(f"{zip_path} holds none of the list files")
            archive.extractall(target, members)
    lists = {split: asvspoof_items(split, protocol_path(split, la_dir), flac_dir(split, la_dir))
             for split in cfg.PROTOCOL_FILENAMES}
    lists["itw"] = itw_items(itw_dir)
    for split, file_list in lists.items():
        check_items(file_list, split)
    return {split: lists[split] for split in cfg.SPLITS}


# Any split -----------------------------------------------------------------------

def items(split: str) -> pd.DataFrame:
    """File list of a split ("train", "dev", "eval", "itw") from the local dataset folders."""
    if split == "itw":
        return itw_items()
    if split in cfg.PROTOCOL_FILENAMES:
        return asvspoof_items(split)
    raise KeyError(f"unknown split {split!r}; known: {cfg.SPLITS}")


def speaker_map(file_list: pd.DataFrame) -> dict[str, str]:
    """audio_id -> speaker."""
    return dict(zip(file_list["audio_id"].astype(str), file_list["speaker"].astype(str)))


def attack_map(file_list: pd.DataFrame) -> dict[str, str]:
    """audio_id -> system id ("-" for bonafide files)."""
    return dict(zip(file_list["audio_id"].astype(str), file_list["system_id"].astype(str)))


def na_like_values(file_list: pd.DataFrame) -> dict[str, list[str]]:
    """Ids and speaker names that pandas would read as missing from a CSV file, per column."""
    return {column: sorted({str(v) for v in file_list[column] if str(v) in NA_LIKE})
            for column in ("audio_id", "system_id", "speaker")}


def check_items(file_list: pd.DataFrame, split: str) -> dict[str, int]:
    """Raise unless a file list is the complete split.

    Checks the columns, the number of files per class, unique audio ids, the
    attack ids of the split and that no id or speaker name is a string pandas
    reads as missing.

    Returns n_files, n_bonafide, n_spoof, n_speakers.
    """
    problems = []
    if list(file_list.columns) != ITEM_COLUMNS:
        raise ValueError(f"columns {list(file_list.columns)}, expected {ITEM_COLUMNS}")
    labels = file_list["label"].to_numpy()
    found = (int((labels == cfg.LABEL_BONAFIDE).sum()), int((labels == cfg.LABEL_SPOOF).sum()))
    if found != cfg.CLASS_COUNTS[split] or found[0] + found[1] != len(file_list):
        problems.append(f"(bonafide, spoof) = {found} of {len(file_list)} files, "
                        f"expected {cfg.CLASS_COUNTS[split]}")
    if file_list["audio_id"].duplicated().any():
        problems.append("duplicate audio ids")
    if set(file_list["split"]) != {split}:
        problems.append(f"split column holds {sorted(set(file_list['split']))}")
    bonafide_ids = set(file_list.loc[labels == cfg.LABEL_BONAFIDE, "system_id"])
    spoof_ids = set(file_list.loc[labels == cfg.LABEL_SPOOF, "system_id"])
    if bonafide_ids - {"-"}:
        problems.append(f"bonafide files with system ids {sorted(bonafide_ids - {'-'})}")
    if spoof_ids != set(cfg.SPOOF_SYSTEM_IDS[split]):
        problems.append(f"spoof system ids {sorted(spoof_ids)}, "
                        f"expected {list(cfg.SPOOF_SYSTEM_IDS[split])}")
    na_like = {column: values for column, values in na_like_values(file_list).items() if values}
    if na_like:
        problems.append(f"values read as missing by pandas: {na_like}")
    if problems:
        raise ValueError(f"file list of {split!r}: " + "; ".join(problems))
    return {"n_files": len(file_list), "n_bonafide": found[0], "n_spoof": found[1],
            "n_speakers": int(file_list["speaker"].nunique())}
