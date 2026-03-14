from __future__ import annotations

from pathlib import Path
from typing import Dict, Literal, Optional

import torch
from torch import Tensor
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import Coauthor, Planetoid

DatasetName = Literal[
    "cora",
    "citeseer",
    "pubmed",
    "coauthor-cs",
    "coauthor-physics",
]


def canonicalize_edge(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u < v else (v, u)


def _pairs_to_edge_index(pairs: list[tuple[int, int]], undirected: bool = False) -> Tensor:
    if not pairs:
        return torch.empty((2, 0), dtype=torch.long)

    edges = pairs.copy()
    if undirected:
        edges.extend((v, u) for u, v in pairs)

    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def load_dataset(name: DatasetName, root: str | Path = "data") -> InMemoryDataset:
    normalized = name.lower().strip()
    root_path = Path(root)

    planetoid_map = {
        "cora": "Cora",
        "citeseer": "Citeseer",
        "pubmed": "PubMed",
    }
    coauthor_map = {
        "coauthor-cs": "CS",
        "coauthor-physics": "Physics",
    }

    if normalized in planetoid_map:
        return Planetoid(root=str(root_path / "Planetoid"), name=planetoid_map[normalized])
    if normalized in coauthor_map:
        return Coauthor(root=str(root_path / "Coauthor"), name=coauthor_map[normalized])

    supported = sorted([*planetoid_map.keys(), *coauthor_map.keys()])
    raise ValueError(f"Unsupported dataset '{name}'. Supported: {supported}")


def to_undirected_unique(edge_index: Tensor, num_nodes: int) -> Tensor:
    del num_nodes  # kept for API stability
    pair_set: set[tuple[int, int]] = set()
    src, dst = edge_index

    for u, v in zip(src.tolist(), dst.tolist()):
        if u == v:
            continue
        pair_set.add(canonicalize_edge(int(u), int(v)))

    return _pairs_to_edge_index(sorted(pair_set), undirected=False)


def edge_index_to_edge_set(edge_index: Tensor) -> set[tuple[int, int]]:
    edge_set: set[tuple[int, int]] = set()
    src, dst = edge_index
    for u, v in zip(src.tolist(), dst.tolist()):
        if u == v:
            continue
        edge_set.add(canonicalize_edge(int(u), int(v)))
    return edge_set


def split_edges_for_link_prediction(
    data: Data,
    val_ratio: float = 0.05,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> Dict[str, Tensor]:
    if not (0.0 <= val_ratio < 1.0 and 0.0 <= test_ratio < 1.0):
        raise ValueError("val_ratio and test_ratio must be in [0, 1).")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0")

    pos_undirected = to_undirected_unique(data.edge_index, data.num_nodes)
    all_pairs = list(zip(pos_undirected[0].tolist(), pos_undirected[1].tolist()))

    gen = torch.Generator()
    gen.manual_seed(seed)
    perm = torch.randperm(len(all_pairs), generator=gen).tolist()
    shuffled = [all_pairs[i] for i in perm]

    num_edges = len(shuffled)
    num_val = int(num_edges * val_ratio)
    num_test = int(num_edges * test_ratio)
    num_train = num_edges - num_val - num_test

    if num_train <= 0:
        raise ValueError("Not enough edges for training after split.")

    train_pairs = shuffled[:num_train]
    val_pairs = shuffled[num_train : num_train + num_val]
    test_pairs = shuffled[num_train + num_val :]

    split_dict = {
        "train_pos_edge_index": _pairs_to_edge_index(train_pairs, undirected=False),
        "val_pos_edge_index": _pairs_to_edge_index(val_pairs, undirected=False),
        "test_pos_edge_index": _pairs_to_edge_index(test_pairs, undirected=False),
        "train_edge_index": _pairs_to_edge_index(train_pairs, undirected=True),
    }
    return split_dict


def get_random_negatives(
    edge_index: Tensor,
    num_nodes: int,
    num_samples: int,
    seed: Optional[int] = None,
) -> Tensor:
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative")
    if num_samples == 0:
        return torch.empty((2, 0), dtype=torch.long)

    pos_set = edge_index_to_edge_set(edge_index)
    neg_set: set[tuple[int, int]] = set()

    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)

    # Oversample candidates to reduce rejection overhead on sparse graphs.
    while len(neg_set) < num_samples:
        need = num_samples - len(neg_set)
        batch_size = max(need * 3, 1024)

        u = torch.randint(0, num_nodes, (batch_size,), generator=gen)
        v = torch.randint(0, num_nodes, (batch_size,), generator=gen)

        for ui, vi in zip(u.tolist(), v.tolist()):
            if ui == vi:
                continue
            e = canonicalize_edge(int(ui), int(vi))
            if e in pos_set or e in neg_set:
                continue
            neg_set.add(e)
            if len(neg_set) == num_samples:
                break

    return _pairs_to_edge_index(sorted(neg_set), undirected=False)


def prepare_link_prediction_data(
    dataset_name: DatasetName,
    root: str | Path = "data",
    val_ratio: float = 0.05,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> Dict[str, Tensor | Data | int]:
    dataset = load_dataset(dataset_name, root=root)
    data = dataset[0]

    splits = split_edges_for_link_prediction(
        data,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    all_pos = to_undirected_unique(data.edge_index, data.num_nodes)

    val_pos = splits["val_pos_edge_index"]
    test_pos = splits["test_pos_edge_index"]

    val_neg = get_random_negatives(
        edge_index=all_pos,
        num_nodes=data.num_nodes,
        num_samples=val_pos.size(1),
        seed=seed + 1,
    )
    test_neg = get_random_negatives(
        edge_index=all_pos,
        num_nodes=data.num_nodes,
        num_samples=test_pos.size(1),
        seed=seed + 2,
    )

    return {
        "data": data,
        "x": data.x,
        "train_edge_index": splits["train_edge_index"],
        "train_pos_edge_index": splits["train_pos_edge_index"],
        "val_pos_edge_index": val_pos,
        "test_pos_edge_index": test_pos,
        "val_neg_edge_index": val_neg,
        "test_neg_edge_index": test_neg,
        "num_nodes": int(data.num_nodes),
        "num_features": int(data.num_node_features),
    }
