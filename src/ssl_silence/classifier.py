"""The classifier trained on frozen, pooled features."""

from __future__ import annotations

import torch
import torch.nn as nn

from . import config as cfg


class Classifier(nn.Module):
    """Linear(in_dim, 128) -> ReLU -> Dropout(0.3) -> Linear(128, 2).

    Input (B, in_dim) -> logits (B, 2); class 1 is spoof.
    """

    def __init__(self, in_dim: int, hidden: int = cfg.HIDDEN_DIM,
                 dropout: float = cfg.DROPOUT) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
