"""Loading of the frozen encoders and the frame arithmetic of their convolutional encoder.

An encoder is an entry of ``configs/encoders.yaml``. It is loaded through
``transformers.AutoModel`` at a pinned revision, in float32 and evaluation
mode, with a fixed attention implementation, and checked against its entry.
Kernel and stride lengths are read from the model configuration.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from . import config as cfg

DTYPE = torch.float32
CHECKED_KEYS = ("model_class", "attn_implementation", "n_layers", "hidden_size",
                "feat_extract_norm", "conv_kernel", "conv_stride")


def load_model(source: str, revision: str | None, attn_implementation: str,
               device: str | torch.device | None = None):
    """Frozen model from a checkpoint id or folder: float32, evaluation mode, no gradients."""
    from transformers import AutoModel

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(source, revision=revision,
                                      attn_implementation=attn_implementation, dtype=DTYPE)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def cache_dir() -> str:
    """The folder to which Hugging Face downloads checkpoints in this process."""
    import os

    from huggingface_hub import constants

    return os.path.realpath(constants.HF_HUB_CACHE)


def describe(model) -> dict[str, Any]:
    """The properties of a loaded model that ``check_encoder`` compares with its entry."""
    config = model.config
    return {"model_class": type(model).__name__,
            "attn_implementation": config._attn_implementation,
            "n_layers": int(config.num_hidden_layers),
            "hidden_size": int(config.hidden_size),
            "feat_extract_norm": config.feat_extract_norm,
            "conv_kernel": [int(k) for k in config.conv_kernel],
            "conv_stride": [int(s) for s in config.conv_stride]}


def check_encoder(model, entry: Mapping[str, Any]) -> None:
    """Raise if a loaded model differs from its registry entry, is not float32,
    is in training mode or has trainable parameters."""
    found = describe(model)
    problems = [f"{key}: expected {entry[key]!r}, found {found[key]!r}"
                for key in CHECKED_KEYS if found[key] != _plain(entry[key])]
    dtypes = {parameter.dtype for parameter in model.parameters()}
    if dtypes != {DTYPE}:
        problems.append(f"dtype: expected {DTYPE}, found {sorted(str(d) for d in dtypes)}")
    if model.training:
        problems.append("the model is in training mode")
    if any(parameter.requires_grad for parameter in model.parameters()):
        problems.append("the model has trainable parameters")
    if problems:
        raise ValueError("encoder check failed: " + "; ".join(problems))


def _plain(value: Any) -> Any:
    return list(value) if isinstance(value, (list, tuple)) else value


def load_encoder(name: str, device: str | torch.device | None = None):
    """The encoder ``name`` of the registry, loaded at its pinned revision and checked."""
    entry = cfg.encoder(name)
    model = load_model(entry["hf_id"], entry["revision"], entry["attn_implementation"], device)
    check_encoder(model, entry)
    return model


# Frame arithmetic ------------------------------------------------------------

def conv_layers(model) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """(kernel lengths, stride lengths) of the convolutional encoder, from the model configuration."""
    return (tuple(int(k) for k in model.config.conv_kernel),
            tuple(int(s) for s in model.config.conv_stride))


def conv_output_length(n_samples: int, kernels: Sequence[int], strides: Sequence[int]) -> int:
    """Number of frames the convolutional encoder produces for ``n_samples`` samples.

    L <- floor((L - kernel) / stride) + 1 for every layer. The result is not
    positive for an input shorter than the receptive field.
    """
    length = int(n_samples)
    for kernel, stride in zip(kernels, strides, strict=True):
        length = (length - kernel) // stride + 1
    return length


def real_frame_count(n_frames: int, n_real_samples: int, kernels: Sequence[int],
                     strides: Sequence[int]) -> int:
    """Number of leading frames whose receptive field holds real samples only.

    The convolutional output length of the real samples, clipped to
    [1, n_frames]; padding is at the end of the input.
    """
    return max(1, min(n_frames, conv_output_length(n_real_samples, kernels, strides)))
