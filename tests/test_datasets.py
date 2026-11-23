"""File lists on hand-written protocol and metadata files."""

from __future__ import annotations

import os
import zipfile

import numpy as np
import pandas as pd
import pytest

from ssl_silence import config as cfg
from ssl_silence import datasets as D
from ssl_silence.provenance import sha256_file

PROTOCOL = """LA_0079 LA_T_1138215 - - bonafide
LA_0079 LA_T_1271820 - A01 spoof

LA_0080 LA_T_2000001 - A06 spoof
too short line
LA_0081\tLA_T_3000001\t-\t-\tbonafide
LA_0082   LA_T_4000001   -   A03   spoof   \n"""


def test_protocol_known_values(tmp_path):
    path = tmp_path / "protocol.txt"
    path.write_text(PROTOCOL, encoding="utf-8")
    ids, labels, system_ids, speakers = D.read_asvspoof_protocol(str(path))
    assert ids == ["LA_T_1138215", "LA_T_1271820", "LA_T_2000001", "LA_T_3000001", "LA_T_4000001"]
    assert labels.tolist() == [0, 1, 1, 0, 1]
    assert system_ids == ["-", "A01", "A06", "-", "A03"]
    assert speakers == ["LA_0079", "LA_0079", "LA_0080", "LA_0081", "LA_0082"]


def test_asvspoof_items(tmp_path):
    path = tmp_path / "protocol.txt"
    path.write_text(PROTOCOL, encoding="utf-8")
    table = D.asvspoof_items("train", protocol=str(path), audio_dir="/audio")
    assert list(table.columns) == D.ITEM_COLUMNS and len(table) == 5
    assert set(table["dataset"]) == {"asvspoof19"} and set(table["split"]) == {"train"}
    assert table.loc[1, "path"].replace("\\", "/") == "/audio/LA_T_1271820.flac"
    assert table["label"].tolist() == [0, 1, 1, 0, 1]
    assert D.speaker_map(table)["LA_T_2000001"] == "LA_0080"
    assert D.attack_map(table) == {"LA_T_1138215": "-", "LA_T_1271820": "A01",
                                   "LA_T_2000001": "A06", "LA_T_3000001": "-",
                                   "LA_T_4000001": "A03"}


def test_paths_inside_the_unzipped_archive():
    assert D.protocol_path("eval", "/data").replace("\\", "/") == \
        "/data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
    assert D.flac_dir("dev", "/data").replace("\\", "/") == "/data/LA/ASVspoof2019_LA_dev/flac"
    assert cfg.la_zip("/root").replace("\\", "/") == "/root/datasets/ASVspoof2019_LA/LA.zip"


def write_meta(folder, rows, name="meta.csv"):
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / name, "w", newline="", encoding="utf-8") as fh:
        fh.write("file,speaker,label\n")
        for row in rows:
            fh.write(",".join(row) + "\n")


def test_itw_items(tmp_path):
    release = tmp_path / "itw" / "release_in_the_wild"
    write_meta(release, [("0.wav", "Alec Guinness", "spoof"), (" 1.wav ", " Barack Obama ", "bona-fide"),
                         ("2.wav", "Zoë Müller", " Bona-fide "), ("3.wav", "Alec Guinness", "SPOOF")])
    table = D.itw_items(str(tmp_path / "itw"))
    assert list(table.columns) == D.ITEM_COLUMNS
    assert table["audio_id"].tolist() == ["0.wav", "1.wav", "2.wav", "3.wav"]
    assert table["label"].tolist() == [1, 0, 0, 1]
    assert table["system_id"].tolist() == ["itw", "-", "-", "itw"]
    assert table["speaker"].tolist() == ["Alec Guinness", "Barack Obama", "Zoë Müller", "Alec Guinness"]
    assert set(table["dataset"]) == {"in_the_wild"} and set(table["split"]) == {"itw"}
    assert os.path.dirname(table.loc[0, "path"]) == str(release)
    assert D.speaker_map(table)["2.wav"] == "Zoë Müller"


def test_first_meta_file_in_sorted_order_is_used(tmp_path):
    write_meta(tmp_path / "b", [("9.wav", "B", "spoof")])
    write_meta(tmp_path / "a", [("1.wav", "A", "spoof")])
    assert D.find_itw_meta(str(tmp_path)) == str(tmp_path / "a" / "meta.csv")
    assert D.itw_items(str(tmp_path))["audio_id"].tolist() == ["1.wav"]
    with pytest.raises(FileNotFoundError):
        D.find_itw_meta(str(tmp_path / "a" / "missing"))


def test_items_dispatches_by_split(monkeypatch):
    monkeypatch.setattr(D, "itw_items", lambda: "itw list")
    monkeypatch.setattr(D, "asvspoof_items", lambda split: f"{split} list")
    assert D.items("itw") == "itw list" and D.items("eval") == "eval list"
    with pytest.raises(KeyError):
        D.items("test")


# Completeness check --------------------------------------------------------------

def split_table(split: str) -> pd.DataFrame:
    """A file list with the class counts, system ids and split name of ``split``."""
    n_bona, n_spoof = cfg.CLASS_COUNTS[split]
    attacks = cfg.SPOOF_SYSTEM_IDS[split]
    labels = np.r_[np.zeros(n_bona, int), np.ones(n_spoof, int)]
    system_ids = ["-"] * n_bona + [attacks[k % len(attacks)] for k in range(n_spoof)]
    return pd.DataFrame({"dataset": "x", "split": split,
                         "audio_id": [f"f{k}" for k in range(labels.size)],
                         "path": [f"/a/f{k}.flac" for k in range(labels.size)], "label": labels,
                         "system_id": system_ids,
                         "speaker": [f"s{k % 20}" for k in range(labels.size)]})


@pytest.mark.parametrize("split", ["train", "dev", "eval", "itw"])
def test_check_items_accepts_a_complete_split(split):
    summary = D.check_items(split_table(split), split)
    n_bona, n_spoof = cfg.CLASS_COUNTS[split]
    assert summary == {"n_files": n_bona + n_spoof, "n_bonafide": n_bona, "n_spoof": n_spoof,
                       "n_speakers": 20}


def test_class_counts_are_the_published_split_sizes():
    assert {s: sum(cfg.CLASS_COUNTS[s]) for s in cfg.CLASS_COUNTS} == {
        "train": 25_380, "dev": 24_844, "eval": 71_237, "itw": 31_779}
    assert cfg.SPOOF_SYSTEM_IDS["eval"] == tuple(f"A{k:02d}" for k in range(7, 20))
    assert cfg.SPOOF_SYSTEM_IDS["train"] == ("A01", "A02", "A03", "A04", "A05", "A06")


def test_check_items_rejects_incomplete_or_inconsistent_lists():
    table = split_table("train")
    with pytest.raises(ValueError, match="expected"):
        D.check_items(table.iloc[:-1], "train")
    duplicate = table.copy()
    duplicate.loc[1, "audio_id"] = duplicate.loc[0, "audio_id"]
    with pytest.raises(ValueError, match="duplicate"):
        D.check_items(duplicate, "train")
    wrong_attack = table.copy()
    wrong_attack.loc[len(table) - 1, "system_id"] = "A07"
    with pytest.raises(ValueError, match="system ids"):
        D.check_items(wrong_attack, "train")
    bonafide_with_attack = table.copy()
    bonafide_with_attack.loc[0, "system_id"] = "A01"
    with pytest.raises(ValueError, match="bonafide files"):
        D.check_items(bonafide_with_attack, "train")
    with pytest.raises(ValueError, match="split column"):
        D.check_items(table, "dev")
    with pytest.raises(ValueError, match="columns"):
        D.check_items(table.drop(columns="speaker"), "train")
    swapped = table.copy()
    swapped["label"] = 1 - swapped["label"]
    with pytest.raises(ValueError):
        D.check_items(swapped, "train")


def test_names_that_pandas_reads_as_missing_are_reported():
    table = split_table("train")
    assert D.na_like_values(table) == {"audio_id": [], "system_id": [], "speaker": []}
    table.loc[5, "speaker"], table.loc[6, "speaker"] = "NA", "null"
    assert D.na_like_values(table)["speaker"] == ["NA", "null"]
    with pytest.raises(ValueError, match="missing"):
        D.check_items(table, "train")
    # The reason for the check: such a name does not survive a CSV round trip.
    assert "-" not in D.NA_LIKE and {"NA", "null", "None", "nan", ""} <= D.NA_LIKE


# Archives ------------------------------------------------------------------------

def test_ensure_asvspoof_unzips_once(tmp_path):
    archive = tmp_path / "LA.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt", PROTOCOL)
        z.writestr("LA/ASVspoof2019_LA_train/flac/LA_T_1138215.flac", b"audio")
    local = str(tmp_path / "local")
    D.ensure_asvspoof(str(archive), local)
    assert len(D.asvspoof_items("train", D.protocol_path("train", local),
                                D.flac_dir("train", local))) == 5
    archive.unlink()
    D.ensure_asvspoof(str(archive), local)  # already unzipped: the archive is not opened


def test_the_file_lists_are_read_from_the_archives_without_the_audio(tmp_path, monkeypatch):
    la, itw = tmp_path / "LA.zip", tmp_path / "release_in_the_wild.zip"
    with zipfile.ZipFile(la, "w") as z:
        for name in cfg.PROTOCOL_FILENAMES.values():
            z.writestr(f"LA/ASVspoof2019_LA_cm_protocols/{name}", PROTOCOL)
        z.writestr("LA/ASVspoof2019_LA_asv_protocols/" + cfg.PROTOCOL_FILENAMES["train"], "not a CM protocol")
        z.writestr("LA/ASVspoof2019_LA_train/flac/LA_T_1138215.flac", b"audio")
    with zipfile.ZipFile(itw, "w") as z:
        z.writestr("release_in_the_wild/meta.csv", "file,speaker,label\n0.wav,A,spoof\n1.wav,B,bona-fide\n")
        z.writestr("release_in_the_wild/0.wav", b"audio")
    checked = []
    monkeypatch.setattr(D, "check_items", lambda file_list, split: checked.append(split))
    local = tmp_path / "lists"
    lists = D.archive_file_lists(str(la), str(itw), str(local))
    assert list(lists) == list(cfg.SPLITS) and sorted(checked) == sorted(cfg.SPLITS)
    for split in ("train", "dev", "eval"):
        expected = D.read_asvspoof_protocol(D.protocol_path(split, str(local / "asvspoof2019_la")))
        assert lists[split]["audio_id"].tolist() == expected[0] and len(expected[0]) == 5
        assert lists[split]["label"].tolist() == expected[1].tolist()
        assert list(lists[split].columns) == D.ITEM_COLUMNS
    assert lists["itw"]["audio_id"].tolist() == ["0.wav", "1.wav"] and lists["itw"]["label"].tolist() == [1, 0]
    # Only the list files were unpacked.
    unpacked = sorted(path.name for path in local.rglob("*") if path.is_file())
    assert unpacked == sorted([*cfg.PROTOCOL_FILENAMES.values(), "meta.csv"])
    assert cfg.LOCAL_LISTS not in (cfg.LOCAL_ASVSPOOF, cfg.LOCAL_ITW)   # the audio folders stay "not unpacked"
    with zipfile.ZipFile(tmp_path / "empty.zip", "w") as z:
        z.writestr("readme.txt", "nothing")
    with pytest.raises(FileNotFoundError, match="none of the list files"):
        D.archive_file_lists(str(tmp_path / "empty.zip"), str(itw), str(tmp_path / "other"))


def test_an_incomplete_list_in_an_archive_stops(tmp_path):
    la, itw = tmp_path / "LA.zip", tmp_path / "release_in_the_wild.zip"
    with zipfile.ZipFile(la, "w") as z:
        for name in cfg.PROTOCOL_FILENAMES.values():
            z.writestr(f"LA/ASVspoof2019_LA_cm_protocols/{name}", PROTOCOL)
    make_itw_archive(itw)
    with pytest.raises(ValueError, match="file list of"):
        D.archive_file_lists(str(la), str(itw), str(tmp_path / "lists"))


def make_itw_archive(path):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("release_in_the_wild/meta.csv", "file,speaker,label\n0.wav,A,spoof\n")
        z.writestr("release_in_the_wild/0.wav", b"audio")


def test_in_the_wild_is_never_taken_from_an_unpinned_branch():
    assert cfg.ITW_REVISION == "eee168f92c367f8c82ff2cf42b6f61e362fd6211"   # read on Colab
    # sha256 of the archive downloaded once from that revision (read on Colab)
    assert cfg.ITW_ZIP_SHA256 == "46665f30a6758b45642c659c7e0e80da2ba3d121b064716824a56a010ddf9e1a"
    assert D.itw_url() == ("https://huggingface.co/datasets/mueller91/In-The-Wild/resolve/"
                           "eee168f92c367f8c82ff2cf42b6f61e362fd6211/release_in_the_wild.zip")
    for revision in (None, "", "main", "master"):
        with pytest.raises(ValueError, match="not pinned"):
            D.itw_url(revision)
    revision = "0123456789abcdef0123456789abcdef01234567"
    assert D.itw_url(revision) == ("https://huggingface.co/datasets/mueller91/In-The-Wild/resolve/"
                                   f"{revision}/release_in_the_wild.zip")
    stored = cfg.itw_zip("/root").replace("\\", "/")
    assert stored == "/root/datasets/in_the_wild/release_in_the_wild.zip"


def test_download_itw_once_from_the_pinned_revision(tmp_path, monkeypatch):
    calls = []

    def fake_download(command, check):
        calls.append(command)
        make_itw_archive(command[command.index("-O") + 1])

    monkeypatch.setattr(D.subprocess, "run", fake_download)
    zip_path = str(tmp_path / "datasets" / "in_the_wild" / "release_in_the_wild.zip")
    with pytest.raises(ValueError, match="not pinned"):
        D.download_itw(zip_path, None)
    assert not calls and not os.path.exists(zip_path)
    digest = D.download_itw(zip_path, "abc123")
    assert digest == sha256_file(zip_path) and len(calls) == 1
    assert calls[0][0] == "wget" and calls[0][-1].endswith("/resolve/abc123/release_in_the_wild.zip")
    assert calls[0][calls[0].index("-O") + 1] == zip_path + ".part"
    assert not os.path.exists(zip_path + ".part")
    assert D.download_itw(zip_path, None) == digest and len(calls) == 1  # stored copy: no download


def test_ensure_itw_reads_the_stored_archive_and_checks_its_sha256(tmp_path, monkeypatch):
    def no_download(*args, **kwargs):
        raise AssertionError("ensure_itw never downloads")

    monkeypatch.setattr(D.subprocess, "run", no_download)
    zip_path = tmp_path / "stored" / "release_in_the_wild.zip"
    zip_path.parent.mkdir()
    make_itw_archive(zip_path)
    digest = sha256_file(zip_path)
    local = tmp_path / "local"
    with pytest.raises(ValueError, match="not set"):
        D.ensure_itw(str(zip_path), str(local), sha256=None)
    with pytest.raises(ValueError, match="sha256"):
        D.ensure_itw(str(zip_path), str(local), sha256="0" * 64)
    assert not local.exists()
    D.ensure_itw(str(zip_path), str(local), sha256=digest)
    assert zip_path.exists()  # the stored archive is kept
    assert D.itw_items(str(local))["audio_id"].tolist() == ["0.wav"]
    D.ensure_itw(str(tmp_path / "missing.zip"), str(local), sha256=digest)  # already unzipped
