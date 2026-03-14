from __future__ import annotations

from utils.data_utils import (
    edge_index_to_edge_set,
    get_random_negatives,
    load_dataset,
    split_edges_for_link_prediction,
    to_undirected_unique,
)


def test_edge_splits_are_disjoint() -> None:
    data = load_dataset("cora")[0]
    splits = split_edges_for_link_prediction(data, seed=7)

    train_set = edge_index_to_edge_set(splits["train_pos_edge_index"])
    val_set = edge_index_to_edge_set(splits["val_pos_edge_index"])
    test_set = edge_index_to_edge_set(splits["test_pos_edge_index"])

    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)


def test_split_reproducibility() -> None:
    data = load_dataset("cora")[0]
    a = split_edges_for_link_prediction(data, seed=123)
    b = split_edges_for_link_prediction(data, seed=123)

    assert (a["train_pos_edge_index"] == b["train_pos_edge_index"]).all()
    assert (a["val_pos_edge_index"] == b["val_pos_edge_index"]).all()
    assert (a["test_pos_edge_index"] == b["test_pos_edge_index"]).all()


def test_random_negatives_are_true_non_edges() -> None:
    data = load_dataset("cora")[0]
    pos = to_undirected_unique(data.edge_index, data.num_nodes)
    neg = get_random_negatives(pos, data.num_nodes, num_samples=512, seed=11)

    pos_set = edge_index_to_edge_set(pos)
    neg_set = edge_index_to_edge_set(neg)

    assert len(neg_set) == 512
    assert pos_set.isdisjoint(neg_set)
    for u, v in neg_set:
        assert u != v
