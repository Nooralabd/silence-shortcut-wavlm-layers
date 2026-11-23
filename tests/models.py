"""Small randomly initialised encoders with fixed weights, for tests without downloads."""

from __future__ import annotations

import torch

KERNELS = (10, 3, 3, 3, 3, 2, 2)
STRIDES = (5, 2, 2, 2, 2, 2, 2)


def tiny_config(kind: str, n_layers: int = 12, norm: str = "group", hidden_size: int = 32):
    """Configuration of a small WavLM ("wavlm") or HuBERT ("hubert") encoder."""
    from transformers import HubertConfig, WavLMConfig

    Config = {"wavlm": WavLMConfig, "hubert": HubertConfig}[kind]
    return Config(hidden_size=hidden_size, num_hidden_layers=n_layers, num_attention_heads=4,
                  intermediate_size=2 * hidden_size, conv_dim=(16,) * 7, conv_kernel=KERNELS,
                  conv_stride=STRIDES, feat_extract_norm=norm, num_conv_pos_embeddings=16,
                  num_conv_pos_embedding_groups=4, do_stable_layer_norm=(norm == "layer"),
                  conv_bias=(norm == "layer"))


def tiny_model(kind: str, n_layers: int = 12, norm: str = "group", seed: int = 0,
               attn_implementation: str = "eager", hidden_size: int = 32):
    """Frozen small encoder in evaluation mode with weights fixed by ``seed``."""
    from transformers import HubertModel, WavLMModel

    Model = {"wavlm": WavLMModel, "hubert": HubertModel}[kind]
    config = tiny_config(kind, n_layers, norm, hidden_size)
    config._attn_implementation = attn_implementation
    torch.manual_seed(seed)
    model = Model(config).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def tiny_entry(kind: str, n_layers: int = 12, norm: str = "group", hidden_size: int = 32) -> dict:
    """Registry entry that describes ``tiny_model`` with the same arguments."""
    return {"hf_id": "local", "revision": None,
            "model_class": {"wavlm": "WavLMModel", "hubert": "HubertModel"}[kind],
            "attn_implementation": "eager", "n_layers": n_layers, "hidden_size": hidden_size,
            "feat_extract_norm": norm, "conv_kernel": list(KERNELS), "conv_stride": list(STRIDES)}
