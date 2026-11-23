"""Encoder loading, the encoder check and the frame arithmetic, on small random models."""

from __future__ import annotations

import pytest
import torch

from models import KERNELS, STRIDES, tiny_entry, tiny_model
from ssl_silence import config as cfg
from ssl_silence import encoders as E

# The kernel and stride lengths of the convolutional encoder of WavLM Base.
WAVLM_BASE_KERNELS = (10, 3, 3, 3, 3, 2, 2)
WAVLM_BASE_STRIDES = (5, 2, 2, 2, 2, 2, 2)


# Registry ----------------------------------------------------------------------

def test_registry_fixes_class_attention_and_conv_layers():
    registry = cfg.encoders()
    assert registry["wavlm-base"]["model_class"] == "WavLMModel"
    assert registry["hubert-base"]["model_class"] == "HubertModel"
    assert registry["wavlm-large"]["model_class"] == "WavLMModel"
    for entry in registry.values():
        assert entry["attn_implementation"] == "eager"
        assert tuple(entry["conv_kernel"]) == WAVLM_BASE_KERNELS
        assert tuple(entry["conv_stride"]) == WAVLM_BASE_STRIDES
        assert set(E.CHECKED_KEYS) <= set(entry)


def test_library_default_configurations_have_the_registry_conv_layers():
    from transformers import HubertConfig, WavLMConfig

    for Config in (WavLMConfig, HubertConfig):
        config = Config()
        assert tuple(config.conv_kernel) == WAVLM_BASE_KERNELS
        assert tuple(config.conv_stride) == WAVLM_BASE_STRIDES
        assert config.feat_extract_norm == "group" and config.num_hidden_layers == 12
        assert config.hidden_size == 768


# Loading -------------------------------------------------------------------------

@pytest.mark.parametrize("kind, expected_class", [("wavlm", "WavLMModel"), ("hubert", "HubertModel")])
def test_auto_model_loads_the_expected_class_with_identical_outputs(tmp_path, kind, expected_class):
    reference = tiny_model(kind, n_layers=3)
    reference.save_pretrained(tmp_path)
    model = E.load_model(str(tmp_path), None, "eager", "cpu")
    assert type(model).__name__ == expected_class
    assert not model.training
    assert {p.dtype for p in model.parameters()} == {torch.float32}
    assert not any(p.requires_grad for p in model.parameters())
    assert model.config._attn_implementation == "eager"
    E.check_encoder(model, tiny_entry(kind, n_layers=3))
    x = torch.randn(1, 4_000, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        new = model(x, output_hidden_states=True).hidden_states
        old = reference(x, output_hidden_states=True).hidden_states
    assert len(new) == 4 and all(torch.equal(a, b) for a, b in zip(new, old))


def test_load_model_passes_dtype_revision_and_attention(monkeypatch):
    # `dtype` is the argument of this transformers version; `torch_dtype` is deprecated.
    import transformers

    seen = {}

    def fake_from_pretrained(source, **kwargs):
        seen.update(kwargs, source=source)
        return tiny_model("wavlm", n_layers=1)

    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", fake_from_pretrained)
    E.load_model("some/checkpoint", "abc123", "eager", "cpu")
    assert seen == {"source": "some/checkpoint", "revision": "abc123",
                    "attn_implementation": "eager", "dtype": torch.float32}


def test_wavlm_has_no_other_attention_implementation(tmp_path):
    tiny_model("wavlm", n_layers=2).save_pretrained(tmp_path)
    with pytest.raises(ValueError):
        E.load_model(str(tmp_path), None, "sdpa", "cpu")


def test_hubert_attention_implementation_is_set_explicitly(tmp_path):
    tiny_model("hubert", n_layers=2).save_pretrained(tmp_path)
    eager = E.load_model(str(tmp_path), None, "eager", "cpu")
    sdpa = E.load_model(str(tmp_path), None, "sdpa", "cpu")
    assert eager.config._attn_implementation == "eager"
    assert sdpa.config._attn_implementation == "sdpa"
    entry = tiny_entry("hubert", n_layers=2)
    E.check_encoder(eager, entry)
    with pytest.raises(ValueError, match="attn_implementation"):
        E.check_encoder(sdpa, entry)


def test_load_encoder_uses_the_registry_entry(monkeypatch):
    seen = {}

    def fake_load_model(source, revision, attn_implementation, device=None):
        seen.update(source=source, revision=revision, attn_implementation=attn_implementation,
                    device=device)
        return tiny_model("hubert")

    entry = dict(tiny_entry("hubert"), hf_id="facebook/hubert-base-ls960",
                 revision="dba3bb02fda4248b6e082697eee756de8fe8aa8a")
    monkeypatch.setattr(E, "load_model", fake_load_model)
    monkeypatch.setattr(cfg, "encoder", lambda name: entry)
    model = E.load_encoder("hubert-base", "cpu")
    assert type(model).__name__ == "HubertModel"
    assert seen == {"source": "facebook/hubert-base-ls960", "attn_implementation": "eager",
                    "revision": "dba3bb02fda4248b6e082697eee756de8fe8aa8a", "device": "cpu"}


# The check -----------------------------------------------------------------------

def test_check_encoder_accepts_the_described_model():
    for kind in ("wavlm", "hubert"):
        E.check_encoder(tiny_model(kind), tiny_entry(kind))
    E.check_encoder(tiny_model("wavlm", n_layers=4, norm="layer"),
                    tiny_entry("wavlm", n_layers=4, norm="layer"))


@pytest.mark.parametrize("change, message", [
    ({"model_class": "HubertModel"}, "model_class"),
    ({"n_layers": 11}, "n_layers"),
    ({"hidden_size": 768}, "hidden_size"),
    ({"feat_extract_norm": "layer"}, "feat_extract_norm"),
    ({"attn_implementation": "sdpa"}, "attn_implementation"),
    ({"conv_kernel": [10, 3, 3, 3, 3, 2, 3]}, "conv_kernel"),
    ({"conv_stride": [5, 2, 2, 2, 2, 2, 1]}, "conv_stride"),
])
def test_check_encoder_rejects_a_different_model(change, message):
    with pytest.raises(ValueError, match=message):
        E.check_encoder(tiny_model("wavlm"), dict(tiny_entry("wavlm"), **change))


def test_check_encoder_rejects_training_mode_gradients_and_other_dtypes():
    entry = tiny_entry("wavlm", n_layers=2)
    model = tiny_model("wavlm", n_layers=2)
    model.train()
    with pytest.raises(ValueError, match="training mode"):
        E.check_encoder(model, entry)
    model = tiny_model("wavlm", n_layers=2)
    next(model.parameters()).requires_grad = True
    with pytest.raises(ValueError, match="trainable"):
        E.check_encoder(model, entry)
    with pytest.raises(ValueError, match="dtype"):
        E.check_encoder(tiny_model("wavlm", n_layers=2).double(), entry)


# Frame arithmetic --------------------------------------------------------------

def test_conv_layers_are_read_from_the_model_configuration():
    for kind in ("wavlm", "hubert"):
        assert E.conv_layers(tiny_model(kind, n_layers=2)) == (KERNELS, STRIDES)
    assert E.conv_layers(tiny_model("wavlm", n_layers=2)) == (WAVLM_BASE_KERNELS, WAVLM_BASE_STRIDES)


def test_conv_output_length_known_values():
    k, s = WAVLM_BASE_KERNELS, WAVLM_BASE_STRIDES
    assert E.conv_output_length(64_000, k, s) == 199
    assert E.conv_output_length(16_000, k, s) == 49
    assert E.conv_output_length(400, k, s) == 1 and E.conv_output_length(399, k, s) == 0
    assert E.conv_output_length(719, k, s) == 1 and E.conv_output_length(720, k, s) == 2
    with pytest.raises(ValueError):
        E.conv_output_length(400, k, s[:-1])


@pytest.mark.parametrize("kind", ["wavlm", "hubert"])
def test_conv_output_length_equals_the_model_and_the_real_frame_count(kind):
    model = tiny_model(kind, n_layers=1)
    kernels, strides = E.conv_layers(model)
    for n in (400, 401, 719, 720, 8_000, 20_001, 63_999, 64_000, 64_001):
        expected = int(model._get_feat_extract_output_lengths(n))
        assert E.conv_output_length(n, kernels, strides) == expected
        with torch.no_grad():
            frames = model(torch.zeros(1, n)).last_hidden_state.shape[1]
        assert frames == expected


def test_real_frame_count_is_clipped():
    k, s = WAVLM_BASE_KERNELS, WAVLM_BASE_STRIDES
    assert E.real_frame_count(199, 64_000, k, s) == 199
    assert E.real_frame_count(199, 32_000, k, s) == 99
    assert E.real_frame_count(199, 100, k, s) == 1
    assert E.real_frame_count(50, 64_000, k, s) == 50
