from __future__ import annotations


def test_torch_geometric_imports() -> None:
    import torch  # noqa: F401
    import torch_geometric  # noqa: F401


def test_cora_loads() -> None:
    from utils.data_utils import load_dataset

    dataset = load_dataset("cora")
    data = dataset[0]
    assert data.num_nodes > 0
    assert data.edge_index.size(0) == 2
