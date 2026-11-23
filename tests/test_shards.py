"""Shard format, reading and the per-file tables, on hand-made shards."""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from ssl_silence import provenance, shards

L, D = 3, 4


def payload(lo: int, hi: int, n_total: int, extraction: str = "unpadded", **extra):
    """Shard of files lo..hi-1. Feature value of (file i, layer l, dim d): i + l / 10 + d / 100
    for the mean, plus 0.5 for the standard deviation."""
    n = hi - lo
    i = torch.arange(lo, hi).view(n, 1, 1).float()
    layer = torch.arange(1, L + 1).view(1, L, 1).float() / 10
    dim = torch.arange(D).view(1, 1, D).float() / 100
    mean = i + layer + dim
    out = {"n_real": torch.arange(lo, hi) * 100 + 400, "n_frames": torch.arange(lo, hi) + 1,
           "trim_fallback": torch.tensor([j % 3 == 0 for j in range(lo, hi)]),
           "too_short": torch.tensor([j == 4 for j in range(lo, hi)]),
           "labels": torch.tensor([j % 2 for j in range(lo, hi)]),
           "audio_ids": [f"f{j}" for j in range(lo, hi)],
           "system_ids": ["-" if j % 2 == 0 else "A07" for j in range(lo, hi)],
           "lo": lo, "hi": hi, "n_total": n_total, "shard_size": 3,
           "meta": {"encoder": "hubert-base", "extraction": extraction}}
    if extraction == "unpadded":
        out.update(mean=mean, std=mean + 0.5)
    else:
        out.update(all_frames_mean=mean, all_frames_std=mean + 0.5, real_frames_mean=mean + 1000,
                   real_frames_std=mean + 1000.5, n_real_frames=torch.arange(lo, hi) + 1)
    out.update(extra)
    return out


def write_split(folder, split="dev", extraction="unpadded", n_total=7, **extra):
    folder.mkdir(parents=True, exist_ok=True)
    for k, lo in enumerate(range(0, n_total, 3)):
        shards.write_shard(shards.shard_path(str(folder), split, k),
                           payload(lo, min(lo + 3, n_total), n_total, extraction, **extra),
                           extraction)
    return str(folder)


def test_paths_and_counts():
    assert shards.shard_path("/x", "eval", 7).replace("\\", "/") == "/x/eval_shard_0007.pt"
    assert [shards.n_shards(n, 2000) for n in (1, 2000, 2001, 71_237)] == [1, 1, 2, 36]
    with pytest.raises(KeyError):
        shards.schema("masked")


def test_write_and_read_round_trip(tmp_path):
    path = shards.shard_path(str(tmp_path), "dev", 0)
    original = payload(0, 3, 3)
    shards.write_shard(path, original, "unpadded")
    assert not (tmp_path / "dev_shard_0000.pt.tmp").exists()
    for mmap in (False, True):
        loaded = shards.read_shard(path, "unpadded", mmap=mmap)
        assert set(loaded) == set(shards.schema("unpadded")) | {"meta", "layers"}
        assert torch.equal(loaded["mean"], original["mean"]) and loaded["layers"] is None
        assert loaded["audio_ids"] == ["f0", "f1", "f2"] and loaded["meta"]["encoder"] == "hubert-base"


def test_write_rejects_missing_unknown_and_inconsistent_content(tmp_path):
    path = shards.shard_path(str(tmp_path), "dev", 0)
    bad = payload(0, 3, 3)
    del bad["too_short"]
    with pytest.raises(KeyError):
        shards.write_shard(path, bad, "unpadded")
    with pytest.raises(KeyError):
        shards.write_shard(path, payload(0, 3, 3, other_key=torch.zeros(3)), "unpadded")
    short = payload(0, 3, 3)
    short["labels"] = short["labels"][:2]
    with pytest.raises(ValueError):
        shards.write_shard(path, short, "unpadded")
    with pytest.raises(KeyError):
        shards.write_shard(path, payload(0, 3, 3), "padded")


def test_reader_ignores_and_never_exposes_keys_outside_the_schema(tmp_path):
    # A copied shard keeps every key of the file it was copied from, including
    # ones this repository does not use.
    path = str(tmp_path / "train_shard_0000.pt")
    copied = payload(0, 3, 3)
    copied.update(unused_frame_count=torch.tensor([7, 8, 9]), another_unused_key="x",
                  feature_set="some label")
    torch.save(copied, path)
    for mmap in (False, True):
        loaded = shards.read_shard(path, "unpadded", mmap=mmap)
        assert set(loaded) == set(shards.schema("unpadded")) | {"meta", "layers"}
        assert "unused_frame_count" not in loaded and "another_unused_key" not in loaded
        assert torch.equal(loaded["mean"], copied["mean"])
    split = shards.split_shards(str(tmp_path), "train")
    assert all("unused_frame_count" not in shard for shard in split)
    x, _, _ = shards.load_layer(str(tmp_path), "train", 2, "stats")
    assert x.shape == (3, 2 * D)
    lengths = shards.shard_lengths(str(tmp_path), ("train",))
    assert list(lengths.columns) == ["split", "audio_id", "label", "n_real", "trim_fallback",
                                     "too_short"]


def test_read_rejects_a_file_without_the_schema_keys(tmp_path):
    path = str(tmp_path / "dev_shard_0000.pt")
    incomplete = payload(0, 3, 3)
    del incomplete["std"]
    torch.save(incomplete, path)
    with pytest.raises(KeyError, match="std"):
        shards.read_shard(path, "unpadded")


def test_split_must_be_complete_and_contiguous(tmp_path):
    folder = write_split(tmp_path / "ok")
    assert [(s["lo"], s["hi"]) for s in shards.split_shards(folder, "dev")] == [(0, 3), (3, 6), (6, 7)]
    with pytest.raises(FileNotFoundError):
        shards.split_shards(folder, "eval")
    gap = write_split(tmp_path / "gap")
    (tmp_path / "gap" / "dev_shard_0001.pt").unlink()
    with pytest.raises(RuntimeError, match="gap"):
        shards.split_shards(gap, "dev")
    cut = write_split(tmp_path / "cut")
    (tmp_path / "cut" / "dev_shard_0002.pt").unlink()
    with pytest.raises(RuntimeError, match="incomplete"):
        shards.split_shards(cut, "dev")


def test_lengths_and_fallback_counts(tmp_path):
    folder = write_split(tmp_path)
    lengths = shards.shard_lengths(folder, ("dev",))
    assert lengths["audio_id"].tolist() == [f"f{j}" for j in range(7)]
    assert lengths["n_real"].tolist() == [400 + 100 * j for j in range(7)]
    assert lengths["label"].tolist() == [0, 1, 0, 1, 0, 1, 0]
    counts = shards.fallback_counts(folder, ("dev",)).set_index("label")
    # The same counts from lengths that were read already.
    pd.testing.assert_frame_equal(shards.fallback_counts_of(shards.shard_lengths(folder, ("dev",))),
                                  shards.fallback_counts(folder, ("dev",)))
    # Fallback files: 0, 3, 6 (labels 0, 1, 0). Too short: file 4 (label 0).
    assert counts.loc["0", ["n_files", "n_fallback", "n_too_short"]].tolist() == [4, 2, 1]
    assert counts.loc["1", ["n_files", "n_fallback", "n_too_short"]].tolist() == [3, 1, 0]
    assert counts.loc["all", ["n_files", "n_fallback", "n_too_short"]].tolist() == [7, 3, 1]
    assert counts.loc["all", "pct_fallback"] == pytest.approx(100 * 3 / 7)
    assert isinstance(counts, pd.DataFrame)


def test_load_layer_is_one_based_and_concatenates_mean_and_std(tmp_path):
    folder = write_split(tmp_path)
    x, y, ids = shards.load_layer(folder, "dev", 2, "mean")
    assert x.shape == (7, D) and x.dtype == torch.float32 and y.dtype == torch.long
    assert ids == [f"f{j}" for j in range(7)] and y.tolist() == [0, 1, 0, 1, 0, 1, 0]
    assert x[5].tolist() == pytest.approx([5.2, 5.21, 5.22, 5.23])
    stats, _, _ = shards.load_layer(folder, "dev", 3, "stats")
    assert stats.shape == (7, 2 * D)
    assert stats[1].tolist() == pytest.approx([1.3, 1.31, 1.32, 1.33, 1.8, 1.81, 1.82, 1.83])
    assert torch.equal(stats[:, :D], shards.load_layer(folder, "dev", 3, "mean")[0])
    for layer in (0, 4):
        with pytest.raises(ValueError):
            shards.load_layer(folder, "dev", layer, "mean")
    with pytest.raises(ValueError):
        shards.load_layer(folder, "dev", 1, "max")
    with pytest.raises(ValueError):
        shards.load_layer(folder, "dev", 1, "mean", frames="all-frames")


def test_load_layer_of_the_padded_extraction_selects_the_frame_set(tmp_path):
    folder = write_split(tmp_path, extraction="padded")
    all_frames, _, _ = shards.load_layer(folder, "dev", 1, "stats", "padded", "all-frames")
    real_frames, _, _ = shards.load_layer(folder, "dev", 1, "stats", "padded", "real-frames")
    assert all_frames[2].tolist() == pytest.approx([2.1, 2.11, 2.12, 2.13, 2.6, 2.61, 2.62, 2.63])
    assert torch.allclose(real_frames, all_frames + 1000)
    with pytest.raises(ValueError):
        shards.load_layer(folder, "dev", 1, "mean", "padded")


def test_shard_with_a_subset_of_layers(tmp_path):
    # A shard may hold selected layers only; `layers` lists them (1-based).
    full = payload(0, 3, 3)
    subset = dict(full, mean=full["mean"][:, 1:2], std=full["std"][:, 1:2], layers=[2])
    shards.write_shard(shards.shard_path(str(tmp_path), "dev", 0), subset, "unpadded")
    x, _, _ = shards.load_layer(str(tmp_path), "dev", 2, "mean")
    assert torch.equal(x, full["mean"][:, 1])
    with pytest.raises(ValueError, match="not stored"):
        shards.load_layer(str(tmp_path), "dev", 1, "mean")


def test_the_layer_cache_keeps_what_the_shards_record(tmp_path):
    folder = write_split(tmp_path / "f", meta={"encoder": "hubert-base", "revision": "abc", "git_commit": "c1"})
    cache = str(tmp_path / "cache")
    index = shards.build_layer_cache(folder, "dev", cache)
    assert shards.cache_meta(cache, "dev") == {"encoder": ["hubert-base"], "revision": ["abc"],
                                               "git_commit": ["c1"]}
    assert shards.CACHE_META_KEYS == ("encoder", "revision", "git_commit")
    # Shards of several sessions: every distinct value, in shard order.
    path = shards.shard_path(folder, "dev", 2)
    last = torch.load(path, weights_only=False)
    last["meta"] = {"encoder": "hubert-base", "revision": "abc", "git_commit": "c2"}
    torch.save(last, path)
    other = str(tmp_path / "cache2")
    shards.build_layer_cache(folder, "dev", other)
    assert shards.cache_meta(other, "dev") == {"encoder": ["hubert-base"], "revision": ["abc"],
                                               "git_commit": ["c1", "c2"]}
    # Shards that record none of it.
    plain = write_split(tmp_path / "g", meta={"split": "dev"})
    shards.build_layer_cache(plain, "dev", str(tmp_path / "cache3"))
    assert shards.cache_meta(str(tmp_path / "cache3"), "dev") == {"encoder": [], "revision": [], "git_commit": []}
    # An index without the meta (written before it was kept) is built again, with the same features.
    before = shards.load_layer_from_cache(cache, "dev", 2, "stats")
    old = torch.load(index, weights_only=False)
    del old["meta"]
    torch.save(old, index)
    shards.build_layer_cache(folder, "dev", cache)
    assert shards.cache_meta(cache, "dev")["revision"] == ["abc"]
    after = shards.load_layer_from_cache(cache, "dev", 2, "stats")
    assert torch.equal(before[0], after[0]) and before[2] == after[2]


def test_sha256_and_versions(tmp_path):
    path = tmp_path / "a.txt"
    path.write_bytes(b"abc")
    assert provenance.sha256_file(path) == \
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    found = provenance.versions()
    assert found["torch"].startswith("2.11.0") and found["librosa"] == "0.11.0"
    assert provenance.installed_version("a-package-that-does-not-exist") == "not installed"
    assert provenance.gpu_name("cpu") == "cpu"
    commit = provenance.git_commit()
    assert commit == "unknown" or len(commit.replace("-dirty", "")) == 40
    assert provenance.git_commit(tmp_path) == "unknown"


def test_write_table_records_versions_and_checksum(tmp_path):
    import json

    table = pd.DataFrame({"layer": [1, 2], "eer": [1.5, 2.5]})
    meta_path = provenance.write_table(table, tmp_path / "out" / "cells.csv", encoder="wavlm-base")
    assert meta_path.name == "cells.meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["rows"] == 2 and meta["columns"] == ["layer", "eer"] and meta["encoder"] == "wavlm-base"
    assert meta["sha256"] == provenance.sha256_file(tmp_path / "out" / "cells.csv")
    assert {"scipy", "numpy", "pandas", "scikit-learn", "torch"} <= set(meta["versions"])
    assert pd.read_csv(tmp_path / "out" / "cells.csv").equals(table)
