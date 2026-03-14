from __future__ import annotations

import torch

from models import GAT, GCN
from utils.data_utils import prepare_link_prediction_data


def test_gcn_forward_shapes() -> None:
    prepared = prepare_link_prediction_data("cora", seed=1)
    model = GCN(
        in_channels=prepared["num_features"],
        hidden_channels=64,
        out_channels=32,
        num_layers=2,
    )
    edge_label_index = prepared["train_pos_edge_index"][:, :32]
    z = model.encode(prepared["x"], prepared["train_edge_index"])
    logits = model.decode(z, edge_label_index)

    assert z.shape == (prepared["num_nodes"], 32)
    assert logits.shape == (32,)
    assert torch.isfinite(logits).all()


def test_gat_forward_shapes() -> None:
    prepared = prepare_link_prediction_data("cora", seed=2)
    model = GAT(
        in_channels=prepared["num_features"],
        hidden_channels=16,
        out_channels=32,
        heads=2,
        num_layers=2,
    )
    edge_label_index = prepared["train_pos_edge_index"][:, :32]
    z = model.encode(prepared["x"], prepared["train_edge_index"])
    logits = model.decode(z, edge_label_index)

    assert z.shape == (prepared["num_nodes"], 32)
    assert logits.shape == (32,)
    assert torch.isfinite(logits).all()
