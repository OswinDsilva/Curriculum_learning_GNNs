from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

DecoderType = Literal["inner_product", "hadamard_mlp", "mlp"]


class HadamardMLPDecoder(nn.Module):
    def __init__(self, out_channels: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or out_channels
        self.mlp = nn.Sequential(
            nn.Linear(out_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: Tensor, edge_label_index: Tensor) -> Tensor:
        src, dst = edge_label_index
        edge_feat = z[src] * z[dst]
        return self.mlp(edge_feat).squeeze(-1)


class EdgeMLPDecoder(nn.Module):
    def __init__(self, out_channels: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or out_channels
        self.mlp = nn.Sequential(
            nn.Linear(2 * out_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: Tensor, edge_label_index: Tensor) -> Tensor:
        src, dst = edge_label_index
        edge_feat = torch.cat([z[src], z[dst]], dim=-1)
        return self.mlp(edge_feat).squeeze(-1)


class LinkPredictor(nn.Module, ABC):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        decoder: DecoderType = "inner_product",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.decoder_type: DecoderType = decoder

        if decoder == "hadamard_mlp":
            self.learned_decoder: nn.Module | None = HadamardMLPDecoder(out_channels)
        elif decoder == "mlp":
            self.learned_decoder = EdgeMLPDecoder(out_channels)
        else:
            self.learned_decoder = None

    @abstractmethod
    def encode(self, x: Tensor, edge_index: Tensor) -> Tensor:
        raise NotImplementedError

    def decode(self, z: Tensor, edge_label_index: Tensor) -> Tensor:
        src, dst = edge_label_index
        if self.decoder_type == "inner_product":
            return (z[src] * z[dst]).sum(dim=-1)
        if self.learned_decoder is None:
            raise RuntimeError("Learned decoder is not initialized.")
        return self.learned_decoder(z, edge_label_index)

    def forward(self, x: Tensor, edge_index: Tensor, edge_label_index: Tensor) -> Tensor:
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)
