"""
Graph-structural heuristic similarity scores for link prediction.

All functions operate on a `networkx.Graph` (undirected, unweighted) and a
batch of candidate edges supplied as parallel arrays `src` and `dst`
(numpy int64 arrays of shape [E]).

Scores:
  common_neighbors  – |N(u) ∩ N(v)|
  adamic_adar       – Σ_{w∈N(u)∩N(v)} 1 / log(deg(w))     (deg(w) ≥ 2)
  resource_alloc    – Σ_{w∈N(u)∩N(v)} 1 / deg(w)

High-level helper:
  compute_heuristic_scores(edge_index, num_nodes, negatives, heuristic)
      – builds the nx.Graph, computes scores for every edge in `negatives`,
        returns a float32 numpy array.
"""

from __future__ import annotations

import math
from typing import Literal

import networkx as nx
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Per-pair score functions
# ---------------------------------------------------------------------------

def common_neighbors_score(G: nx.Graph, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Return |N(u) ∩ N(v)| for each (src[i], dst[i]) pair.

    Args:
        G:   undirected graph.
        src: integer node indices, shape [E].
        dst: integer node indices, shape [E].

    Returns:
        float32 array of shape [E].
    """
    scores = np.empty(len(src), dtype=np.float32)
    for i, (u, v) in enumerate(zip(src.tolist(), dst.tolist())):
        if G.has_node(u) and G.has_node(v):
            if u == v:
                scores[i] = 0.0
                continue
            scores[i] = float(len(list(nx.common_neighbors(G, u, v))))
        else:
            scores[i] = 0.0
    return scores


def adamic_adar_score(G: nx.Graph, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Return Adamic-Adar index for each (src[i], dst[i]) pair.

    Nodes with degree < 2 are skipped (log(1) = 0 causes division by zero;
    log(0) undefined). Their contribution is treated as 0.

    Args:
        G:   undirected graph.
        src: integer node indices, shape [E].
        dst: integer node indices, shape [E].

    Returns:
        float32 array of shape [E].
    """
    scores = np.empty(len(src), dtype=np.float32)
    for i, (u, v) in enumerate(zip(src.tolist(), dst.tolist())):
        if G.has_node(u) and G.has_node(v):
            if u == v:
                scores[i] = 0.0
                continue
            aa = 0.0
            for w in nx.common_neighbors(G, u, v):
                deg_w = G.degree(w)
                if deg_w >= 2:
                    aa += 1.0 / math.log(deg_w)
            scores[i] = float(aa)
        else:
            scores[i] = 0.0
    return scores


def resource_allocation_score(G: nx.Graph, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Return Resource Allocation index for each (src[i], dst[i]) pair.

    Nodes with degree 0 (isolated) are skipped.

    Args:
        G:   undirected graph.
        src: integer node indices, shape [E].
        dst: integer node indices, shape [E].

    Returns:
        float32 array of shape [E].
    """
    scores = np.empty(len(src), dtype=np.float32)
    for i, (u, v) in enumerate(zip(src.tolist(), dst.tolist())):
        if G.has_node(u) and G.has_node(v):
            if u == v:
                scores[i] = 0.0
                continue
            ra = 0.0
            for w in nx.common_neighbors(G, u, v):
                deg_w = G.degree(w)
                if deg_w > 0:
                    ra += 1.0 / deg_w
            scores[i] = float(ra)
        else:
            scores[i] = 0.0
    return scores


# ---------------------------------------------------------------------------
# High-level helper
# ---------------------------------------------------------------------------

HeuristicName = Literal["common_neighbors", "adamic_adar", "resource_allocation"]

_SCORE_FN = {
    "common_neighbors": common_neighbors_score,
    "adamic_adar": adamic_adar_score,
    "resource_allocation": resource_allocation_score,
}


def compute_heuristic_scores(
    edge_index: torch.Tensor,
    num_nodes: int,
    negatives: torch.Tensor,
    heuristic: HeuristicName = "common_neighbors",
) -> np.ndarray:
    """Compute structural heuristic scores for a batch of candidate edges.

    Args:
        edge_index: LongTensor [2, E_train], the training graph edges
                    (used to build the networkx graph).
        num_nodes:  total number of nodes in the graph.
        negatives:  LongTensor [2, E_neg], candidate negative edges to score.
        heuristic:  one of "common_neighbors", "adamic_adar",
                    "resource_allocation".

    Returns:
        float32 numpy array of shape [E_neg] with the heuristic score for
        each candidate edge. Higher score → structurally more similar pair →
        harder negative.
    """
    if heuristic not in _SCORE_FN:
        raise ValueError(
            f"Unknown heuristic '{heuristic}'. "
            f"Choose from {list(_SCORE_FN.keys())}."
        )

    # Build undirected graph from training edges
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    ei = edge_index.cpu().numpy()
    edges = list(zip(ei[0].tolist(), ei[1].tolist()))
    G.add_edges_from(edges)

    neg_np = negatives.cpu().numpy()
    src = neg_np[0]
    dst = neg_np[1]

    score_fn = _SCORE_FN[heuristic]
    return score_fn(G, src, dst)
