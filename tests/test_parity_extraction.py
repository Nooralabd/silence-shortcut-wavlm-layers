"""extraction and shards against the original analysis code, on a small random
WavLM with fixed weights. Skipped without the original code."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from models import tiny_entry, tiny_model
from signals import write_dataset
from ssl_silence import config as cfg
from ssl_silence import encoders, extraction as X, shards

N_LAYERS = 12  # the original code takes the layer count of its encoder registry
LENGTHS = (400, 401, 719, 720, 5_000, 20_001, 63_999, 64_000)
# Conditions of this repository -> condition names of the original code.
CONDITIONS = {"clean": "clean", "trim40": "trimmed", "trim20": "trimmed20", "trim15": "trimmed15"}
# Padded feature keys of this repository -> keys of the original code.
PADDED_KEYS = {"all_frames_mean": "unmasked_mean", "all_frames_std": "unmasked_std",
               "real_frames_mean": "masked_mean", "real_frames_std": "masked_std"}
SHARED_KEYS = ("n_real", "n_frames", "trim_fallback", "labels", "audio_ids", "system_ids",
               "lo", "hi", "n_total", "shard_size")


@pytest.fixture(scope="module")
def model():
    return tiny_model("wavlm", n_layers=N_LAYERS)


def waveform(n: int, seed: int) -> torch.Tensor:
    return torch.randn(n, generator=torch.Generator().manual_seed(seed)) * 0.1


def same(new, old) -> float:
    """0.0 if two values are identical, else the largest absolute difference (inf if not comparable)."""
    if isinstance(new, torch.Tensor):
        if not isinstance(old, torch.Tensor) or new.shape != old.shape or new.dtype != old.dtype:
            return np.inf
        return 0.0 if torch.equal(new, old) else float((new.double() - old.double()).abs().max())
    return 0.0 if (type(new) is type(old) and new == old) else np.inf


def compare(parity_record, name, pairs):
    worst = max(same(new, old) for new, old in pairs)
    parity_record("extraction", name, len(pairs), worst)
    assert worst == 0.0, f"{name}: max abs diff {worst}"


def test_pool_frames(original, parity_record):
    O = original("features")
    generator = torch.Generator().manual_seed(3)
    pairs = []
    for shape in ((7, 5), (12, 199, 32), (12, 1, 32), (3, 62, 768)):
        h = torch.randn(*shape, generator=generator)
        for n_real_frames in (None, 1, shape[-2]):
            new, old = X.pool_frames(h, n_real_frames), O.pool_frames(h, n_real_frames)
            pairs += [(new[0], old[0]), (new[1], old[1])]
    compare(parity_record, "pool_frames", pairs)


def test_hidden_states_are_the_model_outputs(model):
    # Not a comparison with the original code (it has no such function): the
    # hidden states returned here are the model outputs without the CNN output.
    for n in LENGTHS:
        x = waveform(n, n)
        with torch.no_grad():
            reference = model(x.unsqueeze(0), output_hidden_states=True).hidden_states
        h = X.hidden_states(model, x, N_LAYERS, "cpu")
        assert all(torch.equal(h[k], reference[k + 1][0]) for k in range(N_LAYERS))


def test_encode_unpadded(model, original, parity_record):
    O = original("features")
    pairs = []
    for n in LENGTHS:
        x = waveform(n, n)
        new, old = X.encode_unpadded(model, x, N_LAYERS, "cpu"), O.encode_unpadded(model, x, N_LAYERS, "cpu")
        pairs += [(new[key], old[key]) for key in ("mean", "std", "n_frames")]
    compare(parity_record, "encode_unpadded", pairs)


def test_encode_padded_without_and_with_attention_mask(model, original, parity_record):
    O, L = original("features"), original("latency")
    without, with_mask = [], []
    for n_real in (400, 5_000, 20_001, 40_000, 63_999, 64_000):
        wav = torch.nn.functional.pad(waveform(n_real, n_real), (0, cfg.WINDOW - n_real))
        new = X.encode_padded(model, wav, n_real, N_LAYERS, "cpu")
        old = O.encode_waveform(model, wav, n_real, N_LAYERS, "cpu")
        without += [(new[key], old[other]) for key, other in PADDED_KEYS.items()]
        without += [(new["n_frames"], old["n_frames"]), (new["n_real_frames"], old["n_real_frames"])]
        new = X.encode_padded(model, wav, n_real, N_LAYERS, "cpu", use_attention_mask=True)
        old = L.encode_with_attention_mask(model, wav, n_real, N_LAYERS, "cpu")
        with_mask += [(new[key], old[other]) for key, other in PADDED_KEYS.items()]
    compare(parity_record, "encode_padded (no attention mask)", without)
    compare(parity_record, "encode_padded (attention mask)", with_mask)


def read_all(folder: str, split: str) -> list[dict]:
    import glob
    import os

    return [torch.load(path, map_location="cpu", weights_only=False)
            for path in sorted(glob.glob(os.path.join(folder, f"{split}_shard_*.pt")))]


def test_unpadded_shards(model, original, original_function, parity_record, tmp_path):
    O = original("features")
    old_shard_lengths = original_function("features", "shard_lengths")
    items = write_dataset(tmp_path)
    entry = tiny_entry("wavlm", n_layers=N_LAYERS)
    pairs, read_back = [], []
    for condition, old_condition in CONDITIONS.items():
        new_dir, old_dir = str(tmp_path / "new" / condition), str(tmp_path / "old" / condition)
        X.extract_shards(items, new_dir, "eval", encoder="wavlm-base", extraction="unpadded",
                         condition=condition, shard_size=4, device="cpu", model=model,
                         entry=entry, progress=False)
        O.extract_unpadded_shards(items, old_dir, "eval", encoder="wavlm-base",
                                  condition=old_condition, shard_size=4, device="cpu",
                                  model=model, progress=False,
                                  extra_conditions=O.EXTRA_CONDITIONS)
        new_shards, old_shards = read_all(new_dir, "eval"), read_all(old_dir, "eval")
        assert len(new_shards) == len(old_shards) == 3
        for new, old in zip(new_shards, old_shards):
            pairs += [(new[key], old[key]) for key in ("mean", "std", "too_short") + SHARED_KEYS]
        # The reader of this repository on the shards the original code wrote.
        for new, old in zip(shards.split_shards(new_dir, "eval", mmap=False),
                            shards.split_shards(old_dir, "eval", mmap=False)):
            assert set(old) == set(new)
            read_back += [(new[key], old[key]) for key in ("mean", "std", "too_short") + SHARED_KEYS]
        new_lengths, old_lengths = shards.shard_lengths(new_dir, ("eval",)), old_shard_lengths(old_dir, ("eval",))
        for column in ("audio_id", "label", "n_real", "trim_fallback", "too_short"):
            assert new_lengths[column].tolist() == old_lengths[column].tolist()
        new_counts = shards.fallback_counts(new_dir, ("eval",))
        old_counts = O.fallback_counts_from_shards(old_dir, ("eval",))
        for column in ("label", "n_files", "n_fallback", "n_too_short", "pct_fallback"):
            assert new_counts[column].tolist() == old_counts[column].tolist()
        for layer in (1, 4, 12):
            for pooling in ("mean", "stats"):
                new_x, new_y, new_ids = shards.load_layer(new_dir, "eval", layer, pooling)
                old_x, old_y, old_ids = O.load_layer_from_shards(old_dir, "eval", layer, pooling,
                                                                 mask="none")
                read_back += [(new_x, old_x), (new_y, old_y), (new_ids, list(old_ids))]
    compare(parity_record, "extract_shards (unpadded, four conditions)", pairs)
    compare(parity_record, "shard reading (original shards, layers, lengths)", read_back)


def test_padded_shards(model, original, parity_record, tmp_path):
    O = original("features")
    items = write_dataset(tmp_path)
    entry = tiny_entry("wavlm", n_layers=N_LAYERS)
    pairs = []
    for condition in ("clean", "trim40"):
        new_dir, old_dir = str(tmp_path / "new" / condition), str(tmp_path / "old" / condition)
        X.extract_shards(items, new_dir, "dev", encoder="wavlm-base", extraction="padded",
                         condition=condition, shard_size=5, device="cpu", model=model,
                         entry=entry, progress=False)
        O.extract_pooled_shards(items, old_dir, "dev", encoder="wavlm-base",
                                condition=CONDITIONS[condition], shard_size=5, device="cpu",
                                model=model, progress=False)
        new_shards, old_shards = read_all(new_dir, "dev"), read_all(old_dir, "dev")
        assert len(new_shards) == len(old_shards) == 2
        for new, old in zip(new_shards, old_shards):
            pairs += [(new[key], old[other]) for key, other in PADDED_KEYS.items()]
            pairs += [(new[key], old[key]) for key in ("n_real_frames",) + SHARED_KEYS]
    compare(parity_record, "extract_shards (padded, clean and 40 dB)", pairs)
