"""Copies to the new root, their manifests and the checks of the copied features."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
import torch

from ssl_silence import config as cfg
from ssl_silence import migration as G
from ssl_silence import provenance
from ssl_silence import shards
from ssl_silence import verify as V
from synthetic import write_feature_split

KEYS = {"all_frames_mean": "plain_mean", "all_frames_std": "plain_std",
        "real_frames_mean": "kept_mean", "real_frames_std": "kept_std", "n_real": "n_real",
        "n_frames": "n_frames", "n_real_frames": "n_real_frames", "trim_fallback": "trim_fallback",
        "labels": "labels", "audio_ids": "audio_ids", "system_ids": "system_ids"}


def write_original_padded(folder, split: str = "train", n: int = 70, n_layers: int = 6,
                          dim: int = 4, shard_size: int = 32) -> None:
    """Padded shards in an invented original layout (all layers, other key names)."""
    os.makedirs(folder, exist_ok=True)
    rng = np.random.default_rng(0)
    for k, lo in enumerate(range(0, n, shard_size)):
        hi = min(lo + shard_size, n)
        m = hi - lo
        tensor = lambda: torch.from_numpy(rng.normal(size=(m, n_layers, dim)).astype(np.float32))
        n_real = torch.from_numpy(rng.integers(300, 64_001, m))
        n_real[0] = 399                                       # shorter than the minimum input
        torch.save({"plain_mean": tensor(), "plain_std": tensor(), "kept_mean": tensor(),
                    "kept_std": tensor(), "n_real": n_real,
                    "n_frames": torch.full((m,), 199), "n_real_frames": n_real // 320,
                    "other_frames": n_real // 321, "trim_fallback": torch.zeros(m, dtype=torch.bool),
                    "labels": torch.from_numpy(rng.integers(0, 2, m)),
                    "audio_ids": [f"{split}_{j}" for j in range(lo, hi)],
                    "system_ids": ["-"] * m, "lo": lo, "hi": hi, "n_total": n,
                    "shard_size": shard_size, "meta": {"audio_backends": ["soundfile"],
                                                       "torch_version": "2.11.0", "split": split}},
                   os.path.join(folder, f"{split}_shard_{k:04d}.pt"))


def test_files_are_copied_with_a_manifest_of_new_paths(tmp_path):
    old, root = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    (old / "a.bin").write_bytes(b"alpha")
    (old / "b.bin").write_bytes(b"beta" * 100)
    pairs = [(str(old / "a.bin"), str(root / "x" / "a.bin")), (str(old / "b.bin"), str(root / "b.bin"))]
    manifest = G.copy_files(pairs, str(root))
    assert list(manifest.columns) == ["file", "bytes", "sha256"]
    assert manifest["file"].tolist() == ["x/a.bin", "b.bin"] and manifest["bytes"].tolist() == [5, 400]
    assert manifest["sha256"].tolist() == [provenance.sha256_file(old / "a.bin"),
                                           provenance.sha256_file(old / "b.bin")]
    assert (root / "x" / "a.bin").read_bytes() == b"alpha"
    assert sorted(os.listdir(old)) == ["a.bin", "b.bin"]              # the source is untouched
    pd.testing.assert_frame_equal(G.copy_files(pairs, str(root)), manifest)   # resumable
    assert G.relative(str(root / "x" / "a.bin"), str(root)) == "x/a.bin"


def test_unpadded_shards_are_copied_file_by_file(tmp_path):
    old, root = str(tmp_path / "old"), str(tmp_path / "new")
    write_feature_split(old, "train", n=300, n_layers=3, dim=4, seed=1)
    write_feature_split(old, "dev", n=100, n_layers=3, dim=4, seed=2)
    target = os.path.join(root, "features", "wavlm-base", "unpadded", "clean")
    manifest = G.copy_feature_dir(old, target, root, splits=("train", "dev"), progress=False)
    assert manifest["file"].tolist() == [
        f"features/wavlm-base/unpadded/clean/{name}" for name in
        ("train_shard_0000.pt", "train_shard_0001.pt", "train_shard_0002.pt", "dev_shard_0000.pt")]
    for name in os.listdir(old):
        assert open(os.path.join(old, name), "rb").read() == open(os.path.join(target, name), "rb").read()
    check = V.compare_feature_dirs(target, old, splits=("train", "dev"),
                                   expected_sizes={"train": 300, "dev": 100})
    assert check["ok"].all()
    V.assert_clean(manifest, {"values": [old], "patterns": []})
    with pytest.raises(FileNotFoundError):
        G.copy_feature_dir(old, target, root, splits=("eval",), progress=False)


def test_one_layer_of_the_padded_shards_is_written_with_new_keys(tmp_path):
    old, root = str(tmp_path / "old"), str(tmp_path / "new")
    write_original_padded(old)
    target = os.path.join(root, "features", "wavlm-base", "padded", "trim40")
    manifest = G.copy_padded_layer(old, target, root, KEYS, layer=4, encoder="wavlm-base",
                                   condition="trim40", splits=("train",), progress=False)
    assert len(manifest) == 3 and manifest["file"].iloc[0].endswith("padded/trim40/train_shard_0000.pt")
    new = shards.split_shards(target, "train", "padded")
    first = torch.load(os.path.join(old, "train_shard_0000.pt"), weights_only=False)
    assert new[0]["all_frames_mean"].shape == (32, 1, 4) and new[0]["layers"] == [4]
    assert torch.equal(new[0]["all_frames_mean"][:, 0, :], first["plain_mean"][:, 3, :])
    assert torch.equal(new[0]["real_frames_std"][:, 0, :], first["kept_std"][:, 3, :])
    # too_short is derived: the first file of every shard has 399 samples.
    assert new[0]["too_short"].tolist() == [True] + [False] * 31
    assert new[0]["meta"]["condition"] == "trim40" and new[0]["meta"]["source"]["split"] == "train"
    assert "other_frames" not in new[0]
    x, _, ids = shards.load_layer(target, "train", 4, "stats", "padded", "real-frames")
    assert x.shape == (70, 8) and ids[0] == "train_0"
    check = V.compare_feature_dirs(target, old, keys=KEYS, splits=("train",), extraction="padded",
                                   old_layer=4, expected_sizes={"train": 70})
    assert check["ok"].all() and len(check) == len(KEYS)
    with pytest.raises(ValueError, match="stores layers"):        # another layer is not what was copied
        V.compare_feature_dirs(target, old, keys=KEYS, splits=("train",), extraction="padded",
                               old_layer=3, expected_sizes={"train": 70})
    stamps = [os.path.getmtime(os.path.join(target, name)) for name in sorted(os.listdir(target))]
    again = G.copy_padded_layer(old, target, root, KEYS, layer=4, encoder="wavlm-base",
                                condition="trim40", splits=("train",), progress=False)
    assert again.equals(manifest)                                # existing shards are kept
    assert stamps == [os.path.getmtime(os.path.join(target, name)) for name in sorted(os.listdir(target))]


def test_file_list_against_the_shards(tmp_path):
    folder = str(tmp_path / "features")
    write_feature_split(folder, "eval", n=150, n_layers=2, dim=3, seed=3)
    lengths = shards.shard_lengths(folder, splits=("eval",))
    stored = shards.split_shards(folder, "eval")
    systems = [s for shard in stored for s in shard["system_ids"]]
    file_list = pd.DataFrame({"audio_id": lengths["audio_id"], "label": lengths["label"],
                              "system_id": systems})
    assert G.check_file_list(file_list, folder, "eval") == {
        "split": "eval", "n_listed": 150, "n_stored": 150, "ids_equal": True, "labels_equal": True,
        "system_ids_equal": True, "ok": True}
    swapped = file_list.copy()
    swapped.iloc[[3, 4]] = swapped.iloc[[4, 3]].to_numpy()        # same files, another order
    row = G.check_file_list(swapped, folder, "eval")
    assert not row["ids_equal"] and not row["ok"]
    relabelled = file_list.assign(label=1 - file_list["label"])
    row = G.check_file_list(relabelled, folder, "eval")
    assert row["ids_equal"] and not row["labels_equal"] and not row["ok"]
    assert not G.check_file_list(file_list.iloc[:-1], folder, "eval")["ok"]
    assert not G.check_file_list(file_list.assign(system_id="A99"), folder, "eval")["system_ids_equal"]


def test_what_the_shards_record(tmp_path):
    folder = str(tmp_path / "old")
    write_original_padded(folder)
    table = G.shard_provenance(folder, splits=("train",)).set_index("key")
    assert table.loc["audio_backends", "values"] == "soundfile" and table.loc["torch_version", "n_shards"] == 3
    assert table.loc["librosa_version", "values"] == "not recorded"
    assert "git_commit" in table.index


def test_original_tables_and_files_are_copied_under_new_names(tmp_path):
    old, root = tmp_path / "old", tmp_path / "new"
    (old / "tabs").mkdir(parents=True)
    (old / "meas").mkdir()
    (old / "tabs" / "fx.csv").write_text("a,b\n1,2\n")
    (old / "sample.csv").write_text("id\n7\n")
    (old / "meas" / "train.csv").write_text("x\n1\n")
    (old / "meas" / "dev.csv").write_text("x\n2\n")
    (old / "meas" / "notes.txt").write_text("not a table")
    (old / "LA.zip").write_bytes(b"zip")
    legacy = {"root": str(old), "forbidden": {"values": ["tabs/fx.csv"], "patterns": []},
              "conditions": {"clean": "clean"},
              "tables": {"effects_2x2_trim40": {"file": "tabs/fx.csv"}},
              "files": {"probe_sample": "sample.csv", "measurements_dir": "meas",
                        "asvspoof_zip": "LA.zip"}}
    target = str(root / "migration" / "original")
    manifest = G.copy_original_files(legacy, target, str(root))
    assert manifest["file"].tolist() == [
        "migration/original/tables/effects_2x2_trim40.csv", "migration/original/files/probe_sample.csv",
        "migration/original/files/measurements_dir/dev.csv",
        "migration/original/files/measurements_dir/train.csv"]
    assert (root / "migration/original/tables/effects_2x2_trim40.csv").read_text() == "a,b\n1,2\n"
    V.assert_clean(manifest, legacy["forbidden"])
    assert not (root / "migration/original/files/asvspoof_zip.zip").exists()
    missing = {**legacy, "tables": {"rank_agreement_trim40": {"file": "tabs/none.csv"}}}
    with pytest.raises(FileNotFoundError, match="1 original files"):
        G.copy_original_files(missing, str(root / "other"), str(root))
    assert not (root / "other").exists()                          # nothing is copied before the check
    twice = {**legacy, "files": {"probe_sample": "sample.csv"},
             "tables": {"a": {"file": "tabs/fx.csv"}}}
    assert len(G.original_files(twice)["tables"]) == 1


def test_files_at_another_sampling_rate_are_counted():
    tables = {"itw": pd.DataFrame({"sr_orig": [16_000] * 5}),
              "eval": pd.DataFrame({"sr_orig": [16_000, 44_100, 16_000, 48_000, 44_100]})}
    table = G.other_sampling_rates(tables).set_index("table")
    assert table.loc["itw", ["n_files", "n_other_rate", "rates"]].tolist() == [5, 0, ""]
    assert table.loc["eval", ["n_other_rate", "rates"]].tolist() == [3, "44100 | 48000"]


def test_environment_report(tmp_path):
    report = provenance.environment_report(str(tmp_path)).set_index("item")["value"]
    for item in ("python", "torch", "torchaudio", "soundfile", "torchcodec", "scipy", "librosa",
                 "praat-parselmouth", "praat", "text_encoding", "gpu_name", "git_commit",
                 "free_space_gb", "cuda_available"):
        assert item in report.index and report[item] != "", item
    assert report["torch"] == torch.__version__.split("+")[0] or report["torch"] == torch.__version__
    assert report["praat-parselmouth"] == "0.4.7" and float(report["free_space_gb"]) > 0
    assert "free_space_gb" not in set(provenance.environment_report()["item"])
    assert float(report["local_disk_free_gb"]) > 0
    assert cfg.MIN_FREE_GB == 150


def test_environment_report_of_a_drive_root_holds_the_quota(monkeypatch):
    root = "/content/drive/MyDrive/ssl_silence"
    monkeypatch.setattr(cfg.os.path, "ismount", lambda path: path == cfg.DRIVE_MOUNT)
    quota = {"limit_gb": 429.4967, "usage_gb": 151.6123, "free_gb": 277.8844}
    report = provenance.environment_report(root, quota=quota).set_index("item")["value"]
    assert (report["drive_limit_gb"], report["drive_usage_gb"], report["drive_free_gb"]) == ("429.5", "151.6", "277.9")
    # The free space of the mounted folder is not in the table; that of the local disk is.
    assert "free_space_gb" not in report.index and float(report["local_disk_free_gb"]) > 0
    with pytest.raises(ValueError, match="needs the quota"):
        provenance.environment_report(root)


# The copied original tables against the paper archive ------------------------------------------

def write_archive(path, tables: dict, manifest: dict | None = None, prefix: str = "") -> str:
    """A zip file of tables under their original paths with its manifest."""
    import hashlib
    import zipfile

    listed = {name: hashlib.sha256(text.encode()).hexdigest() for name, text in tables.items()}
    listed.update(manifest or {})
    with zipfile.ZipFile(path, "w") as zf:
        for name, text in tables.items():
            zf.writestr(prefix + name, text)
        zf.writestr(prefix + G.ARCHIVE_MANIFEST,
                    "path,rows,sha256\n" + "".join(f"{name},1,{digest}\n" for name, digest in listed.items()))
    return str(path)


def archive_case(tmp_path):
    old, root = tmp_path / "old", tmp_path / "new"
    for folder in ("official", "first", "meas"):
        (old / folder).mkdir(parents=True)
    (old / "official" / "fx.csv").write_text("a,b\n1,2\n", newline="")
    (old / "first" / "fx.csv").write_text("a,b\n1,3\n", newline="")
    (old / "official" / "rank.csv").write_text("tau\n0.5\n", newline="")
    (old / "sample.csv").write_text("id\n7\n", newline="")
    (old / "meas" / "train.csv").write_text("x\n1\n", newline="")
    legacy = {"root": str(old), "forbidden": {"values": ["official/fx.csv", "first/fx.csv"], "patterns": []},
              "conditions": {"clean": "clean"},
              "tables": {"effects_2x2_trim40": {"file": "official/fx.csv"},
                         "rank_agreement_trim40": {"file": "official/rank.csv"}},
              "files": {"probe_sample": "sample.csv", "measurements_dir": "meas"}}
    target = str(root / "migration" / "original")
    G.copy_original_files(legacy, target, str(root))
    return legacy, target, str(root)


@pytest.mark.parametrize("prefix", ["", "paper_tables/"])
def test_copied_tables_equal_the_paper_archive(tmp_path, prefix):
    legacy, target, root = archive_case(tmp_path)
    archive = write_archive(tmp_path / "archive.zip", {"official/fx.csv": "a,b\n1,2\n",
                                                       "official/rank.csv": "tau\n0.5\n",
                                                       "other/unused.csv": "id\n7\n"}, prefix=prefix)
    table = G.compare_with_archive(legacy, target, root, archive).set_index("file")
    assert table.index.tolist() == ["migration/original/tables/effects_2x2_trim40.csv",
                                    "migration/original/tables/rank_agreement_trim40.csv",
                                    "migration/original/files/probe_sample.csv"]
    assert table["status"].tolist() == ["identical", "identical", "not in archive"]
    assert table["ok"].all() and (table["sha256"][:2] == table["archive_sha256"][:2]).all()
    # The sample is not at its path in the archive, but the archive holds its content elsewhere.
    sample = table.loc["migration/original/files/probe_sample.csv"]
    assert sample["archive_sha256"] == "" and sample["same_content_in_archive"]
    assert sample["same_name_in_archive"] == 0
    V.assert_clean(table.reset_index(), legacy["forbidden"])
    V.require_ok(table.reset_index(), "archive")


def test_a_table_from_a_duplicate_folder_is_found(tmp_path):
    legacy, target, root = archive_case(tmp_path)
    archive = write_archive(tmp_path / "archive.zip", {"official/fx.csv": "a,b\n1,2\n",
                                                       "official/rank.csv": "tau\n0.5\n"})
    # The mapping names the table of the other folder: same file name, other content.
    other = {**legacy, "tables": {**legacy["tables"], "effects_2x2_trim40": {"file": "first/fx.csv"}}}
    target = str(tmp_path / "second" / "migration" / "original")
    G.copy_original_files(other, target, str(tmp_path / "second"))
    row = G.compare_with_archive(other, target, str(tmp_path / "second"), archive).iloc[0]
    assert row["status"] == "not in archive" and row["same_name_in_archive"] == 1
    assert not row["same_content_in_archive"] and row["ok"]
    # The archive holds another content at the path of the mapping: a hard stop.
    changed = write_archive(tmp_path / "changed.zip", {"official/fx.csv": "a,b\n1,9\n",
                                                       "official/rank.csv": "tau\n0.5\n"})
    table = G.compare_with_archive(legacy, str(tmp_path / "new" / "migration" / "original"),
                                   str(tmp_path / "new"), changed)
    assert table["status"].tolist() == ["different", "identical", "not in archive"]
    assert table["ok"].tolist() == [False, True, True]
    with pytest.raises(V.CheckFailed):
        V.require_ok(table, "archive")


def test_the_archive_must_agree_with_its_manifest(tmp_path):
    import zipfile

    tables = {"official/fx.csv": "a,b\n1,2\n"}
    assert list(G.archive_hashes(write_archive(tmp_path / "good.zip", tables))) == ["official/fx.csv"]
    with pytest.raises(ValueError, match="do not agree"):
        G.archive_hashes(write_archive(tmp_path / "wrong.zip", tables, {"official/fx.csv": "0" * 64}))
    with pytest.raises(ValueError, match="do not agree"):
        G.archive_hashes(write_archive(tmp_path / "missing.zip", tables, {"official/none.csv": "0" * 64}))
    with zipfile.ZipFile(tmp_path / "bare.zip", "w") as zf:
        zf.writestr("official/fx.csv", "a,b\n1,2\n")
    with pytest.raises(ValueError, match="one MANIFEST.csv"):
        G.archive_hashes(str(tmp_path / "bare.zip"))
