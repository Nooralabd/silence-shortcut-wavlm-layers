"""Pooling and extraction on small random encoders, with known answers."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from models import tiny_entry, tiny_model
from signals import clip, write_dataset
from ssl_silence import config as cfg
from ssl_silence import audio, encoders, extraction as X, shards, verify

LENGTHS = (400, 401, 1_234, 8_000, 20_001, 64_000)


def waveform(n: int, seed: int = 0) -> torch.Tensor:
    return torch.randn(n, generator=torch.Generator().manual_seed(seed)) * 0.1


# Pooling -----------------------------------------------------------------------

def test_pool_frames_equals_numpy_population_statistics():
    h = torch.randn(3, 17, 5, generator=torch.Generator().manual_seed(0))
    mean, std = X.pool_frames(h)
    assert mean.shape == std.shape == (3, 5)
    np.testing.assert_allclose(mean.numpy(), h.numpy().mean(axis=1), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(std.numpy(), h.numpy().std(axis=1, ddof=0), rtol=1e-5, atol=1e-6)


def test_pool_frames_known_values():
    h = torch.tensor([[1.0, 10.0], [3.0, 10.0], [5.0, 40.0]])  # (T=3, D=2)
    mean, std = X.pool_frames(h)
    assert torch.allclose(mean, torch.tensor([3.0, 20.0]))
    assert torch.allclose(std, torch.tensor([np.sqrt(8 / 3), np.sqrt(200.0)], dtype=torch.float32))
    mean, std = X.pool_frames(h, n_real_frames=2)
    assert torch.allclose(mean, torch.tensor([2.0, 10.0]))
    assert torch.allclose(std, torch.tensor([1.0, 0.0]))


def test_pool_frames_of_one_frame_and_bad_frame_counts():
    h = torch.randn(2, 4, 3)
    mean, std = X.pool_frames(h, n_real_frames=1)
    assert torch.equal(mean, h[:, 0]) and torch.equal(std, torch.zeros(2, 3))
    for bad in (0, 5):
        with pytest.raises(ValueError):
            X.pool_frames(h, n_real_frames=bad)


# Hidden states -------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["hubert", "wavlm"])
def test_hidden_state_k_is_the_output_of_transformer_layer_k(kind):
    model = tiny_model(kind, n_layers=4)
    captured = []
    hooks = [layer.register_forward_hook(
        lambda module, args, output: captured.append(output[0] if isinstance(output, tuple) else output))
        for layer in model.encoder.layers]
    x = waveform(5_000)
    h = X.hidden_states(model, x, 4, "cpu")
    for hook in hooks:
        hook.remove()
    assert h.shape[0] == 4 and len(captured) == 4
    for k in range(4):
        assert torch.equal(h[k], captured[k][0])
    with torch.no_grad():
        out = model(x.unsqueeze(0), output_hidden_states=True)
    assert torch.equal(h[-1], out.last_hidden_state[0])
    assert not torch.equal(h[0], out.hidden_states[0][0])  # the CNN output is not layer 1


@pytest.mark.parametrize("kind", ["hubert", "wavlm"])
def test_frame_count_and_shape(kind):
    model = tiny_model(kind, n_layers=3)
    kernels, strides = encoders.conv_layers(model)
    for n in LENGTHS:
        h = X.hidden_states(model, waveform(n), 3, "cpu")
        assert h.shape == (3, encoders.conv_output_length(n, kernels, strides), 32)
        assert h.dtype == torch.float32 and not h.requires_grad


def test_wrong_layer_count_is_rejected():
    with pytest.raises(ValueError):
        X.hidden_states(tiny_model("hubert", n_layers=3), waveform(4_000), 12, "cpu")


# Unpadded --------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["hubert", "wavlm"])
def test_encode_unpadded_is_the_pooled_hidden_states(kind):
    model = tiny_model(kind, n_layers=3)
    for n in LENGTHS:
        x = waveform(n, seed=n)
        out = X.encode_unpadded(model, x, 3, "cpu")
        h = X.hidden_states(model, x, 3, "cpu").numpy()
        assert out["n_frames"] == h.shape[1]
        assert out["mean"].shape == out["std"].shape == (3, 32)
        assert out["mean"].dtype == out["std"].dtype == torch.float32
        np.testing.assert_allclose(out["mean"].numpy(), h.mean(axis=1), rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(out["std"].numpy(), h.std(axis=1, ddof=0), rtol=1e-4, atol=1e-6)


def test_one_frame_gives_zero_standard_deviation():
    out = X.encode_unpadded(tiny_model("hubert", n_layers=2), waveform(400), 2, "cpu")
    assert out["n_frames"] == 1 and torch.equal(out["std"], torch.zeros(2, 32))


def test_result_does_not_depend_on_previously_encoded_files():
    model = tiny_model("hubert", n_layers=3)
    x = waveform(9_000, seed=1)
    first = X.encode_unpadded(model, x, 3, "cpu")
    X.encode_unpadded(model, waveform(30_000, seed=2), 3, "cpu")
    X.encode_padded(model, waveform(64_000, seed=3), 20_000, 3, "cpu", use_attention_mask=True)
    again = X.encode_unpadded(model, x, 3, "cpu")
    assert torch.equal(first["mean"], again["mean"]) and torch.equal(first["std"], again["std"])


# Padded ------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["hubert", "wavlm"])
def test_encode_padded_frame_sets(kind):
    model = tiny_model(kind, n_layers=3)
    kernels, strides = encoders.conv_layers(model)
    real = waveform(20_000, seed=4)
    padded = torch.nn.functional.pad(real, (0, cfg.WINDOW - 20_000))
    out = X.encode_padded(model, padded, 20_000, 3, "cpu")
    assert out["n_frames"] == 199
    assert out["n_real_frames"] == encoders.conv_output_length(20_000, kernels, strides) == 62
    h = X.hidden_states(model, padded, 3, "cpu").numpy()
    np.testing.assert_allclose(out["all_frames_mean"].numpy(), h.mean(axis=1), rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(out["real_frames_mean"].numpy(), h[:, :62].mean(axis=1),
                               rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(out["real_frames_std"].numpy(), h[:, :62].std(axis=1, ddof=0),
                               rtol=1e-4, atol=1e-6)
    full = X.encode_padded(model, waveform(cfg.WINDOW, seed=5), cfg.WINDOW, 3, "cpu")
    assert full["n_real_frames"] == full["n_frames"] == 199
    assert torch.equal(full["all_frames_mean"], full["real_frames_mean"])


@pytest.mark.parametrize("kind", ["hubert", "wavlm"])
def test_padding_changes_group_normalised_features_and_the_mask_changes_the_result(kind):
    model = tiny_model(kind, n_layers=3, norm="group")
    real = waveform(20_000, seed=6)
    padded = torch.nn.functional.pad(real, (0, cfg.WINDOW - 20_000))
    reference = X.encode_unpadded(model, real, 3, "cpu")
    no_mask = X.encode_padded(model, padded, 20_000, 3, "cpu")
    with_mask = X.encode_padded(model, padded, 20_000, 3, "cpu", use_attention_mask=True)
    assert reference["n_frames"] == no_mask["n_real_frames"] == with_mask["n_real_frames"]
    assert not torch.allclose(no_mask["real_frames_mean"], reference["mean"], atol=1e-4)
    assert not torch.allclose(with_mask["real_frames_mean"], reference["mean"], atol=1e-4)
    assert not torch.equal(with_mask["real_frames_mean"], no_mask["real_frames_mean"])


def test_a_full_window_is_the_same_input_in_both_extractions():
    model = tiny_model("hubert", n_layers=2)
    x = waveform(cfg.WINDOW, seed=7)
    unpadded = X.encode_unpadded(model, x, 2, "cpu")
    padded = X.encode_padded(model, x, cfg.WINDOW, 2, "cpu")
    assert torch.equal(unpadded["mean"], padded["all_frames_mean"])
    assert torch.equal(unpadded["std"], padded["real_frames_std"])


# Files and shards ----------------------------------------------------------------

def test_prepare_file_per_extraction_and_condition(tmp_path):
    items = write_dataset(tmp_path).set_index("audio_id")
    tone = X.prepare_file(items.loc["padded_tone", "path"], None, "unpadded")
    assert tone["n_real"] == 44_000 and tone["x"].shape == (44_000,)
    # The library that reads a file depends on the machine: torchaudio where
    # torchcodec is installed (Colab), soundfile otherwise.
    assert not tone["trim_fallback"] and not tone["too_short"]
    assert tone["backend"] == audio.load_audio(items.loc["padded_tone", "path"])[2]
    assert tone["backend"] in ("torchaudio", "soundfile")
    assert tone["x"].abs().max().item() == pytest.approx(1.0, abs=1e-6)
    trimmed = X.prepare_file(items.loc["padded_tone", "path"], 40, "unpadded")
    assert abs(trimmed["n_real"] - 24_000) <= 800 and not trimmed["trim_fallback"]
    padded = X.prepare_file(items.loc["padded_tone", "path"], 40, "padded")
    assert padded["x"].shape == (cfg.WINDOW,) and padded["n_real"] == trimmed["n_real"]
    assert torch.equal(padded["x"][:padded["n_real"]], trimmed["x"])
    tiny = X.prepare_file(items.loc["tiny_file", "path"], 40, "unpadded")
    assert tiny["n_real"] == 300 and tiny["x"].shape == (400,) and tiny["too_short"]
    assert tiny["trim_fallback"]
    long = X.prepare_file(items.loc["long_file", "path"], None, "unpadded")
    assert long["n_real"] == cfg.WINDOW and long["x"].shape == (cfg.WINDOW,)
    zeros = X.prepare_file(items.loc["all_zeros", "path"], 20, "unpadded")
    assert zeros["trim_fallback"] and zeros["n_real"] == 20_000
    rate = X.prepare_file(items.loc["other_rate", "path"], None, "unpadded")
    assert rate["n_real"] == 40_000
    with pytest.raises(KeyError):
        X.prepare_file(items.loc["padded_tone", "path"], None, "masked")


@pytest.mark.parametrize("extraction", ["unpadded", "padded"])
def test_extract_shards_writes_the_schema_and_resumes(tmp_path, extraction):
    items = write_dataset(tmp_path)
    model, entry = tiny_model("hubert", n_layers=3), tiny_entry("hubert", n_layers=3)
    out = str(tmp_path / "features")
    kwargs = dict(encoder="hubert-base", extraction=extraction, condition="trim40", shard_size=4,
                  device="cpu", model=model, entry=entry, progress=False)
    first = X.extract_shards(items, out, "dev", max_new_shards=1, **kwargs)
    assert [os.path.basename(p) for p in first] == ["dev_shard_0000.pt"]
    stamp = os.path.getmtime(first[0])
    paths = X.extract_shards(items, out, "dev", **kwargs)
    assert [os.path.basename(p) for p in paths] == [f"dev_shard_{k:04d}.pt" for k in range(3)]
    assert os.path.getmtime(paths[0]) == stamp
    assert not [name for name in os.listdir(out) if name.endswith(".tmp")]

    loaded = shards.split_shards(out, "dev", extraction, mmap=False)
    assert [(s["lo"], s["hi"]) for s in loaded] == [(0, 4), (4, 8), (8, 9)]
    for shard in loaded:
        n = shard["hi"] - shard["lo"]
        for key in shards.FEATURE_KEYS[extraction]:
            assert shard[key].shape == (n, 3, 32) and shard[key].dtype == torch.float32
        assert shard["n_real"].dtype == shard["labels"].dtype == torch.long
        assert shard["trim_fallback"].dtype == shard["too_short"].dtype == torch.bool
        assert shard["n_total"] == 9 and shard["shard_size"] == 4
    ids = [a for s in loaded for a in s["audio_ids"]]
    assert ids == items["audio_id"].tolist()
    assert torch.cat([s["labels"] for s in loaded]).tolist() == items["label"].tolist()
    assert [a for s in loaded for a in s["system_ids"]] == items["system_id"].tolist()

    meta = loaded[0]["meta"]
    assert meta["encoder"] == "hubert-base" and meta["model_class"] == "HubertModel"
    assert meta["attn_implementation"] == "eager" and meta["dtype"] == "float32"
    assert meta["extraction"] == extraction and meta["condition"] == "trim40"
    assert meta["trim_top_db"] == 40 and meta["split"] == "dev"
    assert (meta["window"], meta["sample_rate"], meta["min_samples"]) == (64_000, 16_000, 400)
    backends = {audio.load_audio(path)[2] for path in items["path"][:4]}        # the files of the first shard
    assert set(meta["audio_backends"]) == backends <= {"torchaudio", "soundfile"}
    assert meta["gpu_name"] == "cpu"
    assert {"python", "torch", "torchaudio", "transformers", "librosa", "numpy", "soundfile",
            "torchcodec"} <= set(meta["versions"])
    assert meta["versions"]["librosa"] == "0.11.0" and meta["versions"]["transformers"] == "5.17.0"
    assert meta["hf_id"] == "local" and "revision" in meta and meta["git_commit"]


def test_extracted_features_are_those_of_the_single_file_functions(tmp_path):
    items = write_dataset(tmp_path)
    model, entry = tiny_model("hubert", n_layers=3), tiny_entry("hubert", n_layers=3)
    out = str(tmp_path / "features")
    X.extract_shards(items, out, "eval", encoder="hubert-base", extraction="unpadded",
                     condition="trim20", shard_size=5, device="cpu", model=model, entry=entry,
                     progress=False)
    loaded = shards.split_shards(out, "eval", mmap=False)
    mean = torch.cat([s["mean"] for s in loaded])
    lengths = shards.shard_lengths(out, ("eval",))
    for row, item in enumerate(items.itertuples()):
        prepared = X.prepare_file(item.path, 20, "unpadded")
        encoded = X.encode_unpadded(model, prepared["x"], 3, "cpu")
        assert torch.equal(mean[row], encoded["mean"])
        assert lengths.loc[row, "n_real"] == prepared["n_real"]
        assert lengths.loc[row, "trim_fallback"] == prepared["trim_fallback"]
        assert lengths.loc[row, "too_short"] == prepared["too_short"]
    flags = lengths.set_index("audio_id")
    assert flags.loc["tiny_file", "too_short"] and flags.loc["all_zeros", "trim_fallback"]
    assert flags.loc["short_tone", "trim_fallback"] and not flags.loc["padded_tone", "trim_fallback"]


def test_a_given_model_is_checked_against_the_entry(tmp_path):
    items = write_dataset(tmp_path)
    with pytest.raises(ValueError, match="model_class"):
        X.extract_shards(items, str(tmp_path / "f"), "dev", encoder="hubert-base",
                         extraction="unpadded", condition="clean", device="cpu",
                         model=tiny_model("wavlm", n_layers=3),
                         entry=tiny_entry("hubert", n_layers=3), progress=False)
    with pytest.raises(KeyError):
        X.extract_shards(items, str(tmp_path / "g"), "dev", encoder="hubert-base",
                         extraction="unpadded", condition="trimmed", device="cpu",
                         model=tiny_model("hubert", n_layers=3),
                         entry=tiny_entry("hubert", n_layers=3), progress=False)


# Re-extraction check ---------------------------------------------------------

def test_sample_positions_are_fixed_ascending_and_the_same_for_the_same_length():
    assert (cfg.REEXTRACTION_FILES, cfg.REEXTRACTION_SEED) == (200, (2026, 201))
    positions = X.sample_positions(25_380)
    assert len(positions) == len(set(positions)) == 200 and positions == sorted(positions)
    assert 0 <= positions[0] and positions[-1] < 25_380
    assert positions == X.sample_positions(25_380) != X.sample_positions(24_844)
    assert X.sample_positions(9, 9) == list(range(9))
    for n_total, n in ((9, 10), (9, 0)):
        with pytest.raises(ValueError, match="cannot draw"):
            X.sample_positions(n_total, n)


def extracted(tmp_path, extraction, condition="trim40", seed=0):
    items = write_dataset(tmp_path)
    model, entry = tiny_model("wavlm", n_layers=3, seed=seed), tiny_entry("wavlm", n_layers=3)
    out = str(tmp_path / f"features_{extraction}")
    X.extract_shards(items, out, "dev", encoder="wavlm-base", extraction=extraction,
                     condition=condition, shard_size=4, device="cpu", model=model, entry=entry,
                     progress=False)
    return items, model, out


@pytest.mark.parametrize("extraction", ["unpadded", "padded"])
def test_stored_files_are_the_rows_of_the_shards(tmp_path, extraction):
    items, _, out = extracted(tmp_path, extraction)
    positions = [1, 3, 4, 8]                                   # three shards of 4, 4 and 1 files
    stored = X.stored_files(out, "dev", positions, extraction)
    loaded = shards.split_shards(out, "dev", extraction, mmap=False)
    for key in shards.FEATURE_KEYS[extraction] + shards.PER_FILE_KEYS[extraction]:
        assert torch.equal(stored[key], torch.cat([s[key] for s in loaded])[positions])
    assert stored["audio_ids"] == items["audio_id"].iloc[positions].tolist()
    assert stored["system_ids"] == items["system_id"].iloc[positions].tolist()
    assert stored["layers"] is None
    assert stored["audio_backends"] == sorted({audio.load_audio(p)[2] for p in items["path"]})
    with pytest.raises(IndexError, match="outside"):
        X.stored_files(out, "dev", [0, 9], extraction)
    with pytest.raises(ValueError, match="ascending"):
        X.stored_files(out, "dev", [3, 1], extraction)


@pytest.mark.parametrize("extraction", ["unpadded", "padded"])
def test_reextraction_of_the_same_model_is_exactly_the_stored_features(tmp_path, extraction):
    items, model, out = extracted(tmp_path, extraction)
    table = X.reextraction_check(items, out, "dev", model=model, extraction=extraction,
                                 condition="trim40", device="cpu", n=6)
    keys = shards.FEATURE_KEYS[extraction] + shards.PER_FILE_KEYS[extraction] + shards.LIST_KEYS
    assert table["key"].tolist() == list(keys)
    assert table["ok"].all() and table["exactly_zero"].all()
    assert set(table["n_files"]) == set(table["n"]) == {6} and set(table["split"]) == {"dev"}
    assert set(table["layers"]) == {"all"}
    assert set(table["audio_backends"]) == set(table["stored_audio_backends"])
    by_key = table.set_index("key")
    assert by_key.loc[shards.FEATURE_KEYS[extraction][0], "tolerance"] == 1e-4
    assert by_key.loc["n_real", "tolerance"] == 0.0


def test_other_weights_or_another_condition_fail_the_reextraction(tmp_path):
    items, model, out = extracted(tmp_path, "unpadded")
    other = tiny_model("wavlm", n_layers=3, seed=1)
    table = X.reextraction_check(items, out, "dev", model=other, extraction="unpadded",
                                 condition="trim40", device="cpu", n=9).set_index("key")
    assert not table.loc["mean", "ok"] and not table.loc["std", "ok"]
    assert table.loc["mean", "max_abs_diff"] > 1e-4
    assert table.loc[["n_real", "trim_fallback", "labels", "audio_ids"], "ok"].all()
    # The same weights on the untrimmed files: the lengths differ, which is a failure by itself.
    table = X.reextraction_check(items, out, "dev", model=model, extraction="unpadded",
                                 condition="clean", device="cpu", n=9).set_index("key")
    assert not table.loc["n_real", "ok"] and not table.loc["mean", "ok"]
    # Within the tolerance but not zero: passes, and is reported as not exactly zero.
    close = X.reextraction_check(items, out, "dev", model=model, extraction="unpadded",
                                 condition="trim40", device="cpu", n=9, tolerance=1e-4)
    assert close["ok"].all()
    stored = X.stored_files(out, "dev", list(range(9)))
    new = X.reextract(items, list(range(9)), model=model, extraction="unpadded", condition="trim40",
                      device="cpu")
    new["mean"] = new["mean"] + 5e-5
    table = verify.compare_reextraction(new, stored, ["mean"], ["n_real"]).set_index("key")
    assert table.loc["mean", "ok"] and not table.loc["mean", "exactly_zero"]


def test_a_list_in_another_order_fails_on_the_ids(tmp_path):
    items, model, out = extracted(tmp_path, "unpadded")
    reordered = items.iloc[::-1].reset_index(drop=True)
    table = X.reextraction_check(reordered, out, "dev", model=model, extraction="unpadded",
                                 condition="trim40", device="cpu", n=9).set_index("key")
    assert not table.loc["audio_ids", "ok"]


def test_stored_layers_only_are_compared_and_integer_types_do_not_count(tmp_path):
    items, model, out = extracted(tmp_path, "padded")
    one_layer = str(tmp_path / "one_layer")
    os.makedirs(one_layer)
    for k, shard in enumerate(shards.split_shards(out, "dev", "padded", mmap=False)):
        payload = {key: shard[key] for key in shards.schema("padded")}
        for key in shards.FEATURE_KEYS["padded"]:
            payload[key] = shard[key][:, 1:2, :].clone()
        payload["n_real"] = shard["n_real"].to(torch.int32)          # another integer type, same values
        payload["layers"] = [2]
        payload["meta"] = {"source": {"audio_backends": ["torchaudio"]}}
        shards.write_shard(shards.shard_path(one_layer, "dev", k), payload, "padded")
    stored = X.stored_files(one_layer, "dev", [0, 5], "padded")
    assert stored["layers"] == [2] and stored["all_frames_mean"].shape == (2, 1, 32)
    assert stored["audio_backends"] == ["torchaudio"]
    table = X.reextraction_check(items, one_layer, "dev", model=model, extraction="padded",
                                 condition="trim40", device="cpu", n=9)
    assert table["ok"].all() and table["exactly_zero"].all() and set(table["layers"]) == {"2"}
    assert set(table["stored_audio_backends"]) == {"torchaudio"}
    # A wrong length in the other integer type is still a failure.
    path = shards.shard_path(one_layer, "dev", 0)
    shard = torch.load(path, weights_only=False)
    shard["n_real"][0] += 1
    torch.save(shard, path)
    table = X.reextraction_check(items, one_layer, "dev", model=model, extraction="padded",
                                 condition="trim40", device="cpu", n=9).set_index("key")
    assert not table.loc["n_real", "ok"] and table.loc["all_frames_mean", "ok"]


# Checks around the extraction of another encoder ---------------------------------------------------

def test_the_short_check_passes_on_the_files_and_lengths_of_the_reference(tmp_path):
    items, _, out = extracted(tmp_path, "unpadded")                 # WavLM stand-in, trim40
    reference = shards.shard_lengths(out, splits=("dev",))
    hubert = tiny_model("hubert", n_layers=3)
    table = X.short_check(items, reference, "dev", model=hubert, condition="trim40", device="cpu", n=6)
    assert len(table) == 6 and table["ok"].all()
    assert table["position"].tolist() == X.sample_positions(len(items), 6)
    assert table["audio_id"].tolist() == items["audio_id"].iloc[table["position"]].tolist()
    assert (table["n_frames"] == table["conv_frames"]).all() and (table["seconds"] > 0).all()
    assert (table["n_real"] == table["reference_n_real"]).all() and table["shape_ok"].all() and table["finite"].all()
    kernels, strides = encoders.conv_layers(hubert)
    assert table["conv_frames"].tolist() == [encoders.conv_output_length(n, kernels, strides)
                                             for n in table["n_input"]]
    # more files than the split has: all of them
    everything = X.short_check(items, reference, "dev", model=hubert, condition="trim40", device="cpu", n=50)
    assert len(everything) == len(items) and everything["ok"].all()
    tiny = everything.set_index("audio_id").loc["tiny_file"]
    assert tiny["too_short"] and tiny["n_input"] == cfg.MIN_SAMPLES and tiny["n_real"] == 300


def test_the_short_check_fails_on_other_lengths_another_order_or_another_layer_count(tmp_path):
    items, _, out = extracted(tmp_path, "unpadded")
    reference = shards.shard_lengths(out, splits=("dev",))
    hubert = tiny_model("hubert", n_layers=3)
    kwargs = dict(model=hubert, device="cpu", n=len(items))
    untrimmed = X.short_check(items, reference, "dev", condition="clean", **kwargs)
    different = untrimmed[untrimmed["n_real"] != untrimmed["reference_n_real"]]
    assert len(different) and not different["ok"].any() and (untrimmed["n_frames"] == untrimmed["conv_frames"]).all()
    reordered = X.short_check(items.iloc[::-1].reset_index(drop=True), reference, "dev", condition="trim40", **kwargs)
    assert not reordered["ok"].all() and (reordered.loc[~reordered["ok"], "reference_n_real"] == -1).any()
    changed = reference.copy()
    changed.loc[2, "trim_fallback"] = not changed.loc[2, "trim_fallback"]
    table = X.short_check(items, changed, "dev", condition="trim40", **kwargs)
    assert (~table["ok"]).tolist() == [k == 2 for k in range(len(items))]
    assert not X.short_check(items, reference[reference["split"] == "eval"], "dev", condition="trim40",
                             **kwargs)["ok"].any()


def test_the_file_list_is_compared_with_the_reference_shards(tmp_path):
    items, _, out = extracted(tmp_path, "unpadded")
    reference = shards.shard_lengths(out, splits=("dev",))
    row = X.file_list_check(items, reference, "dev").iloc[0]
    assert row["ok"] and row["n_files"] == row["n_reference"] == len(items)
    other_order = X.file_list_check(items.iloc[::-1], reference, "dev").iloc[0]
    assert not other_order["ok"] and not other_order["same_ids_in_order"] and not other_order["same_labels"]
    shorter = X.file_list_check(items.iloc[:-1], reference, "dev").iloc[0]
    assert not shorter["ok"] and shorter["n_files"] == len(items) - 1
    relabelled = items.assign(label=1 - items["label"])
    row = X.file_list_check(relabelled, reference, "dev").iloc[0]
    assert row["same_ids_in_order"] and not row["same_labels"] and not row["ok"]
    assert not X.file_list_check(items, reference, "eval").iloc[0]["ok"]


def test_the_estimate_is_the_mean_time_per_file_of_each_split_times_its_files():
    import pandas as pd

    timing = pd.DataFrame({"condition": ["clean"] * 4 + ["trim40"] * 4,
                           "split": ["train", "train", "dev", "dev"] * 2,
                           "seconds": [1.0, 3.0, 0.5, 0.5, 1.0, 1.0, 0.25, 0.25]})
    table = X.extraction_estimate(timing, {"train": 3_600, "dev": 7_200}, n_layers=12, hidden_size=768,
                                  shard_size=2_000).set_index("condition")
    assert table.loc["clean", "hours"] == pytest.approx((2.0 * 3_600 + 0.5 * 7_200) / 3_600)
    assert table.loc["trim40", "hours"] == pytest.approx((1.0 * 3_600 + 0.25 * 7_200) / 3_600)
    assert table.loc["clean", "seconds_per_file"] == pytest.approx(10_800 / 10_800)
    assert table["n_files"].tolist() == [10_800] * 2 and table["n_shards"].tolist() == [2 + 4] * 2
    assert table["n_timed_files"].tolist() == [4, 4]
    assert table.loc["clean", "feature_gb"] == pytest.approx(10_800 * 12 * 768 * 2 * 4 / 1e9)
    with pytest.raises(KeyError, match="eval"):
        X.extraction_estimate(timing, {"train": 1, "eval": 1}, 12, 768)
    # all files of the two encoders: 153,240 files, 11.3 GB per condition
    sizes = {**cfg.ASVSPOOF_SPLIT_SIZES, "itw": cfg.ITW_SIZE}
    full = X.extraction_estimate(pd.DataFrame({"condition": "clean", "split": list(sizes), "seconds": 0.1}),
                                 sizes, 12, 768)
    assert full.loc[0, "n_files"] == 153_240 and full.loc[0, "n_shards"] == 13 + 13 + 36 + 16
    assert full.loc[0, "feature_gb"] == pytest.approx(11.298, abs=1e-3)


def test_extracted_lengths_are_compared_with_the_reference_per_split(tmp_path):
    _, _, out = extracted(tmp_path, "unpadded")
    reference = shards.shard_lengths(out, splits=("dev",))
    same = X.lengths_check(reference.copy(), reference, splits=("dev",)).iloc[0]
    assert same["ok"] and same["n_files"] == same["n_reference"] == len(reference)
    assert same["n_fallback"] == int(reference["trim_fallback"].sum()) > 0
    changed = reference.copy()
    changed.loc[1, "n_real"] += 1
    changed.loc[[3, 4], "too_short"] = ~changed.loc[[3, 4], "too_short"]
    row = X.lengths_check(changed, reference, splits=("dev",)).iloc[0]
    assert not row["ok"] and (row["n_different_n_real"], row["n_different_too_short"]) == (1, 2)
    assert (row["n_different_label"], row["n_different_trim_fallback"]) == (0, 0)
    reordered = X.lengths_check(reference.iloc[::-1], reference, splits=("dev",)).iloc[0]
    assert not reordered["ok"] and reordered["n_different_n_real"] == -1
    # a split that was not extracted is a failing row
    table = X.lengths_check(reference, reference, splits=("dev", "eval"))
    assert table["ok"].tolist() == [True, False] and table.loc[1, "n_files"] == 0


def hubert_shards(tmp_path, monkeypatch, commits=("c1", "c1", "c2")):
    """Shards of the registry encoder "hubert-base", written in three sessions at ``commits``."""
    from ssl_silence import provenance

    items = write_dataset(tmp_path)
    registry = cfg.encoder("hubert-base")
    entry = dict(tiny_entry("hubert", n_layers=3), hf_id=registry["hf_id"], revision=registry["revision"])
    out = str(tmp_path / "hubert")
    for commit in commits:
        monkeypatch.setattr(provenance, "git_commit", lambda commit=commit: commit)
        X.extract_shards(items, out, "dev", encoder="hubert-base", extraction="unpadded", condition="trim40",
                         shard_size=4, device="cpu", model=tiny_model("hubert", n_layers=3), entry=entry,
                         max_new_shards=1, progress=False)
    return items, out


def test_the_meta_of_the_shards_is_listed_per_distinct_combination(tmp_path, monkeypatch):
    items, out = hubert_shards(tmp_path, monkeypatch)
    table = X.shard_meta_table(out, splits=("dev",))
    assert table["git_commit"].tolist() == ["c1", "c2"] and table["n_shards"].tolist() == [2, 1]
    assert table["n_files"].tolist() == [8, 1] and int(table["n_files"].sum()) == len(items)
    assert set(table["encoder"]) == {"hubert-base"} and set(table["gpu_name"]) == {"cpu"}
    assert set(table["revision"]) == {cfg.encoder("hubert-base")["revision"]}
    assert set(table["librosa"]) == {"0.11.0"} and set(table["transformers"]) == {"5.17.0"}
    assert set(X.META_COLUMNS) | set(X.META_VERSIONS) | {"audio_backends", "n_shards", "n_files"} == set(table.columns)

    assert X.shard_meta_check(table, "hubert-base", "trim40")["ok"].all()
    # the registry entry of another encoder, another condition, other versions
    # model class, id and revision are those of HuBERT Base, so WavLM Base fails
    assert not X.shard_meta_check(table, "wavlm-base", "trim40")["ok"].any()
    assert not X.shard_meta_check(table, "hubert-base", "trim20")["ok"].any()
    from ssl_silence import provenance

    other = dict(provenance.versions(), librosa="0.10.2")
    assert not X.shard_meta_check(table, "hubert-base", "trim40", versions=other)["ok"].any()


def test_a_shard_of_uncommitted_code_fails_the_meta_check(tmp_path, monkeypatch):
    _, out = hubert_shards(tmp_path, monkeypatch, commits=("c1", "c1-dirty", "unknown"))
    table = X.shard_meta_check(X.shard_meta_table(out, splits=("dev",)), "hubert-base", "trim40")
    assert table["git_commit"].tolist() == ["c1", "c1-dirty", "unknown"] and table["ok"].tolist() == [True, False, False]
