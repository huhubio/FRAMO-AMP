"""Neural-network architecture used by all three ensemble members."""

from __future__ import annotations

from collections.abc import Sequence

from torch import nn


def build_mlp(input_dim: int, hidden: Sequence[int], dropout: float) -> nn.Sequential:
    """Build an MLP matching the architecture used during model tuning."""
    if input_dim <= 0:
        raise ValueError("input_dim must be positive")
    if any(width <= 0 for width in hidden):
        raise ValueError("all hidden layer widths must be positive")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")

    layers: list[nn.Module] = []
    previous = input_dim
    for width in hidden:
        layers.extend((nn.Linear(previous, width), nn.ReLU(), nn.Dropout(dropout)))
        previous = width
    layers.append(nn.Linear(previous, 1))
    return nn.Sequential(*layers)
