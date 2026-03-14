from __future__ import annotations

import sys


def main() -> int:
    try:
        import torch
        import torch_geometric
        from torch_geometric.datasets import Planetoid
    except Exception as exc:  # pragma: no cover - direct failure path
        print(f"Import error: {exc}")
        return 1

    print("Torch version:", torch.__version__)
    print("PyG version:", torch_geometric.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))

    try:
        dataset = Planetoid(root="data/Planetoid", name="Cora")
        data = dataset[0]
    except Exception as exc:  # pragma: no cover - direct failure path
        print(f"Dataset load error: {exc}")
        return 2

    print("Dataset loaded successfully")
    print("Nodes:", data.num_nodes)
    print("Edges:", data.edge_index.size(1))
    print("Features:", data.num_node_features)
    return 0


if __name__ == "__main__":
    sys.exit(main())
