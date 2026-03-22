#!/usr/bin/env python3
"""
Precompute structural heuristic difficulty scores for candidate negative edges.

For each (dataset, heuristic) pair this script:
  1. Loads the dataset and splits edges for link prediction.
  2. Generates a large pool of random candidate negative edges (10× the
     number of training positive edges by default).
  3. Computes the chosen heuristic score for every candidate.
  4. Saves the candidates and scores to
     data/precomputed/{dataset}_{heuristic}.npz

Artefact schema
---------------
  candidates : int64   [2, N]  – source/dest node pairs
  scores     : float32 [N]     – heuristic difficulty score (higher = harder)
  edge_index : int64   [2, E]  – training graph edges used to build the nx graph
  num_nodes  : int64   scalar
  heuristic  : str             – stored as a length-1 object array

Usage
-----
  python scripts/precompute_scores.py --dataset cora --heuristic common_neighbors
  python scripts/precompute_scores.py --dataset citeseer --heuristic adamic_adar
  python scripts/precompute_scores.py --dataset pubmed --heuristic resource_allocation
  python scripts/precompute_scores.py --all_datasets --all_heuristics
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Literal

import numpy as np
import torch

# Ensure project root is on the path when run as a script
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

HeuristicName = Literal["common_neighbors", "adamic_adar", "resource_allocation"]

DATASETS: list[str] = ["cora", "citeseer", "pubmed"]
HEURISTICS: list[HeuristicName] = [
    "common_neighbors",
    "adamic_adar",
    "resource_allocation",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_candidate_negatives(
    num_nodes: int,
    train_pos_edge_index: torch.Tensor,
    n_candidates: int,
    seed: int = 0,
) -> torch.Tensor:
    """Sample candidate negative edges (no self-loops, not in training set).

    Uses a simple rejection-sampling loop.  For typical sparse graphs the
    expected number of tries is close to 1.

    Returns:
        LongTensor [2, n_candidates].
    """
    rng = np.random.default_rng(seed)
    train_set: set[tuple[int, int]] = set()
    ei = train_pos_edge_index.cpu().numpy()
    for u, v in zip(ei[0].tolist(), ei[1].tolist()):
        train_set.add((min(u, v), max(u, v)))

    collected: list[tuple[int, int]] = []
    batch = max(n_candidates * 2, 100_000)
    while len(collected) < n_candidates:
        src = rng.integers(0, num_nodes, size=batch)
        dst = rng.integers(0, num_nodes, size=batch)
        for u, v in zip(src.tolist(), dst.tolist()):
            if u == v:
                continue
            key = (min(u, v), max(u, v))
            if key in train_set:
                continue
            collected.append((u, v))
            if len(collected) >= n_candidates:
                break

    arr = np.array(collected, dtype=np.int64).T  # [2, N]
    return torch.from_numpy(arr)


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------


def precompute(
    dataset: str,
    heuristic: HeuristicName,
    neg_ratio: int = 10,
    seed: int = 0,
    output_dir: Path | None = None,
    verbose: bool = True,
) -> Path:
    """Precompute heuristic scores and persist as .npz.

    Args:
        dataset:    dataset name (cora / citeseer / pubmed).
        heuristic:  which structural heuristic to evaluate.
        neg_ratio:  number of negative candidates per training positive edge.
        seed:       RNG seed for candidate generation.
        output_dir: destination folder (default: data/precomputed/).
        verbose:    print progress to stdout.

    Returns:
        Path to the saved .npz file.
    """
    from negative_sampling.heuristics import compute_heuristic_scores
    from utils.data_utils import prepare_link_prediction_data

    if output_dir is None:
        output_dir = _PROJECT_ROOT / "data" / "precomputed"
    else:
        output_dir = Path(output_dir)
        if not output_dir.is_absolute():
            output_dir = _PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / f"{dataset}_{heuristic}.npz"
    if out_path.exists():
        if verbose:
            print(f"[skip] {out_path} already exists.")
        return out_path

    if verbose:
        print(f"\n=== {dataset} / {heuristic} ===")
        t0 = time.time()

    # 1. Load data and build LP split
    split = prepare_link_prediction_data(
        dataset_name=dataset,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=seed,
    )
    train_pos = split["train_pos_edge_index"]
    num_nodes: int = split["num_nodes"]
    n_pos = train_pos.shape[1]

    if verbose:
        print(f"  Nodes: {num_nodes} | Train pos edges: {n_pos}")

    # 2. Generate candidate negatives
    n_candidates = neg_ratio * n_pos
    if verbose:
        print(
            f"  Generating {n_candidates:,} candidate negatives (ratio={neg_ratio}x)…"
        )

    candidates = _sample_candidate_negatives(
        num_nodes, train_pos, n_candidates, seed=seed
    )

    # 3. Score
    if verbose:
        print(f"  Computing '{heuristic}' scores…")

    scores = compute_heuristic_scores(
        edge_index=split["train_edge_index"],
        num_nodes=num_nodes,
        negatives=candidates,
        heuristic=heuristic,
    )

    if verbose:
        elapsed = time.time() - t0
        nonzero = int((scores > 0).sum())
        print(
            f"  Done in {elapsed:.1f}s | nonzero scores: "
            f"{nonzero}/{len(scores)} ({100 * nonzero / len(scores):.1f}%)"
        )

    # 4. Save
    np.savez_compressed(
        out_path,
        candidates=candidates.cpu().numpy(),
        scores=scores,
        edge_index=split["train_edge_index"].cpu().numpy(),
        num_nodes=np.int64(num_nodes),
        heuristic=np.array([heuristic], dtype=object),
    )
    if verbose:
        size_mb = out_path.stat().st_size / 1024 / 1024
        print(f"  Saved → {out_path.relative_to(_PROJECT_ROOT)} ({size_mb:.2f} MB)")

    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Precompute structural heuristic scores for negative edges."
    )
    p.add_argument(
        "--dataset",
        choices=DATASETS,
        default=None,
        help="Which dataset to process (omit to use --all_datasets).",
    )
    p.add_argument(
        "--heuristic",
        choices=HEURISTICS,
        default=None,
        help="Which heuristic to compute (omit to use --all_heuristics).",
    )
    p.add_argument(
        "--all_datasets",
        action="store_true",
        help="Process all datasets.",
    )
    p.add_argument(
        "--all_heuristics",
        action="store_true",
        help="Compute all three heuristics.",
    )
    p.add_argument(
        "--neg_ratio",
        type=int,
        default=10,
        help="Negative candidates per training positive edge (default: 10).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0).",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: data/precomputed/).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    datasets = (
        DATASETS if args.all_datasets else ([args.dataset] if args.dataset else None)
    )
    heuristics = (
        HEURISTICS
        if args.all_heuristics
        else ([args.heuristic] if args.heuristic else None)
    )

    if datasets is None or heuristics is None:
        print(
            "Error: specify --dataset / --heuristic or "
            "--all_datasets / --all_heuristics."
        )
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else None

    for ds in datasets:
        for h in heuristics:
            precompute(
                dataset=ds,
                heuristic=h,
                neg_ratio=args.neg_ratio,
                seed=args.seed,
                output_dir=output_dir,
                verbose=True,
            )

    print("\nAll done.")


if __name__ == "__main__":
    main()
