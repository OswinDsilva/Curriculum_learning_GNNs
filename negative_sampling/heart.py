from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data

from models.base import LinkPredictor
from utils.data_utils import (
    canonicalize_edge,
    edge_index_to_edge_set,
    get_random_negatives,
)


HeuristicAlias = Literal[
    "cn", "aa", "ra", "common_neighbors", "adamic_adar", "resource_allocation"
]

HEURISTIC_ALIASES = {
    "cn": "common_neighbors",
    "common_neighbors": "common_neighbors",
    "aa": "adamic_adar",
    "adamic_adar": "adamic_adar",
    "ra": "resource_allocation",
    "resource_allocation": "resource_allocation",
}


def resolve_heuristic_name(name: str) -> str:
    normalized = name.lower().strip()
    if normalized not in HEURISTIC_ALIASES:
        raise ValueError(
            f"Unknown heuristic '{name}'. Choose from {sorted(HEURISTIC_ALIASES)}."
        )
    return HEURISTIC_ALIASES[normalized]


def compute_per_positive_rank(pos_score: float, neg_scores: np.ndarray) -> int:
    return 1 + int(np.sum(neg_scores > pos_score))


def compute_hits_at_k_per_pos(rank: int, k: int) -> float:
    return 1.0 if rank <= k else 0.0


class HeaRTEvaluator:
    def __init__(
        self,
        data: Data,
        heuristics: list[HeuristicAlias] | None = None,
        num_neg_per_pos: int = 100,
        precomputed_dir: str = "data/precomputed",
        seed: int = 42,
        dataset_name: str | None = None,
    ) -> None:
        self.data = data
        self.heuristics = [
            resolve_heuristic_name(h) for h in (heuristics or ["cn", "aa", "ra"])
        ]
        self.num_neg_per_pos = num_neg_per_pos
        self.precomputed_dir = Path(precomputed_dir)
        self.seed = seed
        self.dataset_name = dataset_name
        self._rng = np.random.default_rng(seed)
        self._all_positive_edges = edge_index_to_edge_set(data.edge_index)
        self._heuristic_pools: dict[str, dict[str, Any]] = {}
        self._load_precomputed_pools()

    def _load_precomputed_pools(self) -> None:
        for heuristic in self.heuristics:
            matches = sorted(self.precomputed_dir.glob(f"*_{heuristic}.npz"))
            if self.dataset_name is not None:
                path = self.precomputed_dir / f"{self.dataset_name}_{heuristic}.npz"
                matches = [path] if path.exists() else []
            elif matches:
                matches = [
                    p for p in matches if self._precomputed_matches_graph(p)
                ] or matches
            if not matches:
                raise FileNotFoundError(
                    f"Missing precomputed scores for heuristic '{heuristic}' in {self.precomputed_dir}. "
                    f"Run scripts/precompute_scores.py first."
                )
            path = matches[0]

            payload = np.load(path, allow_pickle=True)
            candidates = payload["candidates"].astype(np.int64, copy=False)
            scores = payload["scores"].astype(np.float32, copy=False)

            node_to_indices: dict[int, list[int]] = defaultdict(list)
            for idx, (u, v) in enumerate(
                zip(candidates[0].tolist(), candidates[1].tolist())
            ):
                node_to_indices[int(u)].append(idx)
                node_to_indices[int(v)].append(idx)

            for node, indices in node_to_indices.items():
                indices.sort(key=lambda i: float(scores[i]), reverse=True)

            self._heuristic_pools[heuristic] = {
                "candidates": candidates,
                "scores": scores,
                "node_to_indices": dict(node_to_indices),
            }

    def _precomputed_matches_graph(self, path: Path) -> bool:
        payload = np.load(path, allow_pickle=True)
        num_nodes = int(payload["num_nodes"]) if "num_nodes" in payload else None
        return num_nodes == int(self.data.num_nodes)

    def _candidate_pairs_for_pair(self, u: int, v: int) -> list[tuple[int, int]]:
        merged: dict[tuple[int, int], float] = {}
        for heuristic in self.heuristics:
            pool = self._heuristic_pools[heuristic]
            node_to_indices = pool["node_to_indices"]
            scores = pool["scores"]
            candidates = pool["candidates"]
            indices = node_to_indices.get(u, []) + node_to_indices.get(v, [])
            for idx in indices:
                pair = canonicalize_edge(
                    int(candidates[0][idx]), int(candidates[1][idx])
                )
                merged[pair] = max(merged.get(pair, float("-inf")), float(scores[idx]))
        return [
            pair
            for pair, _ in sorted(
                merged.items(), key=lambda item: item[1], reverse=True
            )
        ]

    def _fallback_negatives(
        self,
        excluded: set[tuple[int, int]],
        num_nodes: int,
        count: int,
    ) -> Tensor:
        if count <= 0:
            return torch.empty((2, 0), dtype=torch.long)
        pos_edges = list(excluded)
        edge_index = (
            torch.tensor(pos_edges, dtype=torch.long).t().contiguous()
            if pos_edges
            else torch.empty((2, 0), dtype=torch.long)
        )
        return get_random_negatives(
            edge_index=edge_index,
            num_nodes=num_nodes,
            num_samples=count,
            seed=int(self._rng.integers(0, 1_000_000)),
        )

    def generate_test_set(self, pos_edge_index: Tensor) -> tuple[Tensor, list[Tensor]]:
        hard_negs_per_pos: list[Tensor] = []
        all_pos = set(self._all_positive_edges)
        num_nodes = int(self.data.num_nodes)

        for u, v in zip(pos_edge_index[0].tolist(), pos_edge_index[1].tolist()):
            selected_pairs: list[tuple[int, int]] = []
            seen_pairs: set[tuple[int, int]] = set()
            for pair in self._candidate_pairs_for_pair(int(u), int(v)):
                if pair in all_pos or pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                selected_pairs.append(pair)
                if len(selected_pairs) >= self.num_neg_per_pos:
                    break

            if len(selected_pairs) < self.num_neg_per_pos:
                excluded = all_pos | seen_pairs
                fallback = self._fallback_negatives(
                    excluded=excluded,
                    num_nodes=num_nodes,
                    count=self.num_neg_per_pos - len(selected_pairs),
                )
                selected_pairs.extend(
                    list(zip(fallback[0].tolist(), fallback[1].tolist()))
                )

            hard_negs_per_pos.append(
                torch.tensor(selected_pairs, dtype=torch.long).t().contiguous()
            )

        return pos_edge_index, hard_negs_per_pos

    @torch.no_grad()
    def evaluate_model(
        self,
        model: LinkPredictor,
        data_dict: dict[str, Any],
        device: torch.device,
    ) -> dict[str, float]:
        model.eval()
        x = data_dict["x"].to(device)
        train_edge_index = data_dict["train_edge_index"].to(device)
        pos_edge_index = data_dict["test_pos_edge_index"].cpu()

        z = model.encode(x, train_edge_index)
        _, hard_negs_per_pos = self.generate_test_set(pos_edge_index)

        reciprocal_ranks: list[float] = []
        hits10: list[float] = []
        hits50: list[float] = []
        hits100: list[float] = []

        for idx in range(pos_edge_index.size(1)):
            pos_edge = pos_edge_index[:, idx : idx + 1].to(device)
            neg_edges = hard_negs_per_pos[idx].to(device)
            pos_score = float(model.decode(z, pos_edge).item())
            neg_scores = model.decode(z, neg_edges).detach().cpu().numpy()

            rank = compute_per_positive_rank(pos_score, neg_scores)
            reciprocal_ranks.append(1.0 / rank)
            hits10.append(compute_hits_at_k_per_pos(rank, 10))
            hits50.append(compute_hits_at_k_per_pos(rank, 50))
            hits100.append(compute_hits_at_k_per_pos(rank, 100))

        return {
            "heart_mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
            "heart_hits@10": float(np.mean(hits10)) if hits10 else 0.0,
            "heart_hits@50": float(np.mean(hits50)) if hits50 else 0.0,
            "heart_hits@100": float(np.mean(hits100)) if hits100 else 0.0,
        }
