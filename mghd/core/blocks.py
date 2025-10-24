"""MGHD building blocks extracted from legacy poc modules."""
from __future__ import annotations

import torch
import torch.nn as nn

from .panq_functions import GNNDecoder


class ChannelSE(nn.Module):
    """Squeeze-and-Excitation for channel attention."""

    def __init__(self, channels: int, reduction: int = 4, use_hsigmoid: bool = False) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)
        self.act = nn.SiLU()
        if use_hsigmoid:
            self.gate = lambda t: torch.clamp((t + 3.0) / 6.0, 0.0, 1.0)
        else:
            self.gate = torch.sigmoid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect [B, T, C]; reuse across arbitrary batch/time dims.
        squeeze = x.mean(dim=1)
        excite = self.fc2(self.act(self.fc1(squeeze)))
        weights = self.gate(excite).unsqueeze(1)
        return x * weights


class AstraGNN(nn.Module):
    """Wrapper over the Astra GNN decoder that normalizes ctor kwargs."""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        normalized = dict(kwargs)
        if 'n_node_features' not in normalized and 'n_node_inputs' in normalized:
            normalized['n_node_features'] = normalized['n_node_inputs']
        if 'msg_net_size' not in normalized and 'n_node_inputs' in normalized:
            normalized['msg_net_size'] = max(96, normalized['n_node_inputs'])
        if 'n_node_outputs' not in normalized:
            normalized['n_node_outputs'] = 2
        self.core = GNNDecoder(**normalized)

    def forward(self, node_inputs: torch.Tensor, src_ids: torch.Tensor, dst_ids: torch.Tensor) -> torch.Tensor:
        return self.core(node_inputs, src_ids, dst_ids)


__all__ = ["ChannelSE", "AstraGNN"]
