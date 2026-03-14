from __future__ import annotations

import torch
import torch.nn.functional as F

from models import GCN
from utils.data_utils import get_random_negatives, prepare_link_prediction_data, to_undirected_unique


def test_single_training_step_smoke() -> None:
    prepared = prepare_link_prediction_data("cora", seed=0)

    model = GCN(
        in_channels=prepared["num_features"],
        hidden_channels=64,
        out_channels=32,
        num_layers=2,
        dropout=0.3,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    pos_edge_index = prepared["train_pos_edge_index"][:, :128]
    all_pos = to_undirected_unique(prepared["data"].edge_index, prepared["num_nodes"])
    neg_edge_index = get_random_negatives(
        edge_index=all_pos,
        num_nodes=prepared["num_nodes"],
        num_samples=pos_edge_index.size(1),
        seed=123,
    )

    model.train()
    optimizer.zero_grad()

    z = model.encode(prepared["x"], prepared["train_edge_index"])
    pos_logits = model.decode(z, pos_edge_index)
    neg_logits = model.decode(z, neg_edge_index)

    logits = torch.cat([pos_logits, neg_logits], dim=0)
    labels = torch.cat(
        [
            torch.ones(pos_logits.size(0), dtype=torch.float32),
            torch.zeros(neg_logits.size(0), dtype=torch.float32),
        ],
        dim=0,
    )

    loss = F.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
