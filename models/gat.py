from __future__ import annotations

import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GATConv

from models.base import DecoderType, LinkPredictor


class GAT(LinkPredictor):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.5,
        decoder: DecoderType = "inner_product",
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            decoder=decoder,
        )

        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")

        self.convs = nn.ModuleList()
        self.convs.append(
            GATConv(
                in_channels,
                hidden_channels,
                heads=heads,
                concat=True,
                dropout=dropout,
            )
        )

        hidden_dim = hidden_channels * heads
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    hidden_dim,
                    hidden_channels,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                )
            )
            hidden_dim = hidden_channels * heads

        self.convs.append(
            GATConv(
                hidden_dim,
                out_channels,
                heads=1,
                concat=False,
                dropout=dropout,
            )
        )

        self.activation = nn.ELU()
        self.dropout_layer = nn.Dropout(dropout)

    def encode(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = self.activation(x)
            x = self.dropout_layer(x)
        x = self.convs[-1](x, edge_index)
        return x
