from __future__ import annotations

import argparse
from typing import Dict

import torch
from torch_geometric.data import Data

from utils.data_utils import prepare_link_prediction_data, to_undirected_unique


def compute_graph_statistics(data: Data) -> Dict[str, float]:
    undirected_unique = to_undirected_unique(data.edge_index, data.num_nodes)
    num_nodes = int(data.num_nodes)
    num_edges = int(undirected_unique.size(1))

    avg_degree = (2.0 * num_edges) / max(num_nodes, 1)
    density = 0.0
    if num_nodes > 1:
        density = (2.0 * num_edges) / (num_nodes * (num_nodes - 1))

    return {
        "num_nodes": float(num_nodes),
        "num_edges_undirected": float(num_edges),
        "num_features": float(data.num_node_features),
        "avg_degree": avg_degree,
        "density": density,
    }


def summarize_link_prediction_splits(split_dict: Dict[str, torch.Tensor]) -> Dict[str, int]:
    return {
        "train_pos": int(split_dict["train_pos_edge_index"].size(1)),
        "val_pos": int(split_dict["val_pos_edge_index"].size(1)),
        "test_pos": int(split_dict["test_pos_edge_index"].size(1)),
        "val_neg": int(split_dict["val_neg_edge_index"].size(1)),
        "test_neg": int(split_dict["test_neg_edge_index"].size(1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset exploration for link prediction")
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--root", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepared = prepare_link_prediction_data(
        dataset_name=args.dataset,
        root=args.root,
        seed=args.seed,
    )

    data = prepared["data"]
    stats = compute_graph_statistics(data)
    split_stats = summarize_link_prediction_splits(prepared)

    print(f"Dataset: {args.dataset}")
    print(f"Nodes: {int(stats['num_nodes'])}")
    print(f"Unique undirected edges: {int(stats['num_edges_undirected'])}")
    print(f"Features: {int(stats['num_features'])}")
    print(f"Average degree: {stats['avg_degree']:.4f}")
    print(f"Density: {stats['density']:.6f}")
    print()
    print(f"Train positives: {split_stats['train_pos']}")
    print(f"Val positives: {split_stats['val_pos']}")
    print(f"Test positives: {split_stats['test_pos']}")
    print(f"Val negatives: {split_stats['val_neg']}")
    print(f"Test negatives: {split_stats['test_neg']}")


if __name__ == "__main__":
    main()
