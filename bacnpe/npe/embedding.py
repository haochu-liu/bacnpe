"""Embedding networks for the summary statistics."""
from __future__ import annotations

from typing import Optional, Sequence, Type

import torch
import torch.nn as nn


class LeaveLengthOut_NN(nn.Module):
    """
    MLP embedding network that passes the sequence length through unchanged.

    The whole input ``x`` goes through an MLP. The last column of ``x``
    (``L``, entry 29 of the summary vector) is then appended to the MLP
    output, so the output dimension is ``num_outputs + 1``. With
    ``z_score_x="independent"`` in sbi, ``x`` is standardised before the
    embedding, so the appended value is the standardised ``L``.

    Parameters
    ----------
    input_dim : int, default 30
        Number of input features (the length of the summary vector).
    num_hiddens : int, default 48
        Width of each hidden layer when `hidden_dims` is not given.
    num_hidden_layers : int, default 2
        Number of hidden layers when `hidden_dims` is not given.
    num_outputs : int, default 4
        Output dimension of the MLP, before ``L`` is appended.
    activation : type of nn.Module, default nn.ReLU
        Activation class, e.g. ``nn.ReLU``, ``nn.ELU``, ``nn.GELU``.
    hidden_dims : sequence of int, optional
        Explicit hidden-layer sizes, e.g. ``[64, 128, 64]``. Overrides
        `num_hiddens` and `num_hidden_layers`.
    """

    def __init__(
        self,
        input_dim: int = 30,
        num_hiddens: int = 48,
        num_hidden_layers: int = 2,
        num_outputs: int = 4,
        activation: Type[nn.Module] = nn.ReLU,
        hidden_dims: Optional[Sequence[int]] = None,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [num_hiddens] * num_hidden_layers

        layers = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(activation())
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, num_outputs))

        # Attribute name kept as in the original, so old checkpoints still load
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.mlp(x), x[:, -1:]], dim=-1)
