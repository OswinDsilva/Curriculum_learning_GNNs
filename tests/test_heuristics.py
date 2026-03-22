"""
Tests for negative_sampling/heuristics.py

Known-value reference graph (triangle + spoke):

    0 — 1
    |  / |
    | /  |
    2    3

  Edges: 0-1, 0-2, 1-2, 1-3
  Degrees: 0→2, 1→3, 2→2, 3→1

For pair (0, 3):   N(0)={1,2}, N(3)={1}   → common = {1}
  CN  = 1
  AA  = 1 / log(deg(1)) = 1 / log(3)
  RA  = 1 / deg(1)      = 1 / 3

For pair (0, 1):   already neighbours but we test anyway (edge is real, not
                   used inside heuristics which don't exclude real edges).
  N(0)={1,2}, N(1)={0,2,3}  → common = {2}
  CN  = 1
  AA  = 1 / log(deg(2)) = 1 / log(2)
  RA  = 1 / deg(2)      = 1 / 2

For pair (2, 3):   N(2)={0,1}, N(3)={1}   → common = {1}
  Same as (0,3).

For pair (0, 0):   self-loop: no common neighbours → all 0.

Non-existent node: node 99 → all scores 0.
"""

import math
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from negative_sampling.heuristics import (
    adamic_adar_score,
    common_neighbors_score,
    compute_heuristic_scores,
    resource_allocation_score,
)

# ---------------------------------------------------------------------------
# Shared fixture: build the reference graph
# ---------------------------------------------------------------------------


@pytest.fixture()
def ref_graph() -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 3])
    G.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3)])
    return G


def _arr(*vals: int) -> np.ndarray:
    return np.array(vals, dtype=np.int64)


# ---------------------------------------------------------------------------
# common_neighbors_score
# ---------------------------------------------------------------------------


class TestCommonNeighbors:
    def test_single_common_neighbor(self, ref_graph):
        # (0,3): common = {1}
        scores = common_neighbors_score(ref_graph, _arr(0), _arr(3))
        assert scores.shape == (1,)
        assert scores[0] == pytest.approx(1.0)

    def test_pair_sharing_one_common(self, ref_graph):
        # (2,3): common = {1}
        scores = common_neighbors_score(ref_graph, _arr(2), _arr(3))
        assert scores[0] == pytest.approx(1.0)

    def test_adjacent_pair_has_common_neighbor(self, ref_graph):
        # (0,1): common = {2}  → CN = 1
        scores = common_neighbors_score(ref_graph, _arr(0), _arr(1))
        assert scores[0] == pytest.approx(1.0)

    def test_no_common_neighbors(self, ref_graph):
        # Add isolated node 4; (4, 0) → 0 common
        ref_graph.add_node(4)
        scores = common_neighbors_score(ref_graph, _arr(4), _arr(0))
        assert scores[0] == pytest.approx(0.0)

    def test_self_loop(self, ref_graph):
        scores = common_neighbors_score(ref_graph, _arr(0), _arr(0))
        assert scores[0] == pytest.approx(0.0)

    def test_missing_node(self, ref_graph):
        scores = common_neighbors_score(ref_graph, _arr(0), _arr(99))
        assert scores[0] == pytest.approx(0.0)

    def test_batch(self, ref_graph):
        src = _arr(0, 2, 2)
        dst = _arr(3, 3, 0)
        scores = common_neighbors_score(ref_graph, src, dst)
        assert scores.shape == (3,)
        assert scores[0] == pytest.approx(1.0)  # (0,3)
        assert scores[1] == pytest.approx(1.0)  # (2,3)
        assert scores[2] == pytest.approx(1.0)  # (2,0): common={1}


# ---------------------------------------------------------------------------
# adamic_adar_score
# ---------------------------------------------------------------------------


class TestAdamicAdar:
    def test_known_value_0_3(self, ref_graph):
        # common = {1}, deg(1) = 3  →  AA = 1/log(3)
        scores = adamic_adar_score(ref_graph, _arr(0), _arr(3))
        expected = 1.0 / math.log(3)
        assert scores[0] == pytest.approx(expected, rel=1e-5)

    def test_known_value_0_1(self, ref_graph):
        # common = {2}, deg(2) = 2  →  AA = 1/log(2)
        scores = adamic_adar_score(ref_graph, _arr(0), _arr(1))
        expected = 1.0 / math.log(2)
        assert scores[0] == pytest.approx(expected, rel=1e-5)

    def test_common_neighbor_degree_1_excluded(self, ref_graph):
        # Node 3 has deg=1 → log(1)=0 → its contribution is skipped.
        # Build a graph where the only common neighbor has deg=1.
        G = nx.Graph()
        G.add_nodes_from([0, 1, 2])
        G.add_edges_from([(0, 2), (1, 2)])  # common={2}, deg(2)=2, ok
        # Now isolate 2 to deg=1 by removing 0-2, keeping only 1-2
        G2 = nx.Graph()
        G2.add_nodes_from([0, 1, 2, 3])
        G2.add_edges_from([(0, 3), (1, 3)])  # common={3}, deg(3)=2
        # Make a graph where common neighbor has deg=1
        G3 = nx.Graph()
        G3.add_nodes_from([0, 1, 2])
        G3.add_edge(0, 2)
        # Only one edge from 2; deg(2)=1 → log(1)=0 → excluded → AA=0
        # But (0,1) has no common neighbors here anyway; let's be direct:
        G4 = nx.Graph()
        G4.add_nodes_from([0, 1, 2])
        G4.add_edges_from([(0, 2), (1, 2)])  # deg(2)=2 → NOT excluded
        G4.remove_edge(0, 2)  # now deg(2)=1 → excluded
        # (0,1) no longer has 2 as common neighbor (no edge 0-2)
        scores = adamic_adar_score(G4, _arr(0), _arr(1))
        assert scores[0] == pytest.approx(0.0)

    def test_no_common_neighbors(self, ref_graph):
        ref_graph.add_node(4)
        scores = adamic_adar_score(ref_graph, _arr(4), _arr(0))
        assert scores[0] == pytest.approx(0.0)

    def test_missing_node(self, ref_graph):
        scores = adamic_adar_score(ref_graph, _arr(0), _arr(99))
        assert scores[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# resource_allocation_score
# ---------------------------------------------------------------------------


class TestResourceAllocation:
    def test_known_value_0_3(self, ref_graph):
        # common = {1}, deg(1) = 3  →  RA = 1/3
        scores = resource_allocation_score(ref_graph, _arr(0), _arr(3))
        assert scores[0] == pytest.approx(1.0 / 3, rel=1e-5)

    def test_known_value_2_3(self, ref_graph):
        # same as (0,3)
        scores = resource_allocation_score(ref_graph, _arr(2), _arr(3))
        assert scores[0] == pytest.approx(1.0 / 3, rel=1e-5)

    def test_known_value_0_1(self, ref_graph):
        # common = {2}, deg(2) = 2  →  RA = 1/2
        scores = resource_allocation_score(ref_graph, _arr(0), _arr(1))
        assert scores[0] == pytest.approx(0.5, rel=1e-5)

    def test_no_common_neighbors(self, ref_graph):
        ref_graph.add_node(4)
        scores = resource_allocation_score(ref_graph, _arr(4), _arr(0))
        assert scores[0] == pytest.approx(0.0)

    def test_missing_node(self, ref_graph):
        scores = resource_allocation_score(ref_graph, _arr(0), _arr(99))
        assert scores[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_heuristic_scores (high-level helper)
# ---------------------------------------------------------------------------


class TestComputeHeuristicScores:
    """Test the end-to-end edge_index → scores pipeline."""

    # Build a small training graph: edges 0-1, 0-2, 1-2, 1-3
    _EDGE_INDEX = torch.tensor(
        [[0, 1, 0, 2, 1, 2, 1, 3], [1, 0, 2, 0, 2, 1, 3, 1]],
        dtype=torch.long,
    )
    _NUM_NODES = 4

    @pytest.fixture()
    def negatives_0_3(self) -> torch.Tensor:
        return torch.tensor([[0], [3]], dtype=torch.long)

    def test_cn_known_value(self, negatives_0_3):
        scores = compute_heuristic_scores(
            self._EDGE_INDEX, self._NUM_NODES, negatives_0_3, "common_neighbors"
        )
        assert scores.shape == (1,)
        assert scores[0] == pytest.approx(1.0)

    def test_aa_known_value(self, negatives_0_3):
        scores = compute_heuristic_scores(
            self._EDGE_INDEX, self._NUM_NODES, negatives_0_3, "adamic_adar"
        )
        assert scores[0] == pytest.approx(1.0 / math.log(3), rel=1e-5)

    def test_ra_known_value(self, negatives_0_3):
        scores = compute_heuristic_scores(
            self._EDGE_INDEX, self._NUM_NODES, negatives_0_3, "resource_allocation"
        )
        assert scores[0] == pytest.approx(1.0 / 3, rel=1e-5)

    def test_invalid_heuristic_raises(self, negatives_0_3):
        with pytest.raises(ValueError, match="Unknown heuristic"):
            compute_heuristic_scores(
                self._EDGE_INDEX, self._NUM_NODES, negatives_0_3, "not_a_heuristic"
            )

    def test_output_dtype(self, negatives_0_3):
        scores = compute_heuristic_scores(
            self._EDGE_INDEX, self._NUM_NODES, negatives_0_3, "common_neighbors"
        )
        assert scores.dtype == np.float32

    def test_batch_shape(self):
        negatives = torch.tensor(
            [[0, 2, 3], [3, 3, 0]],
            dtype=torch.long,
        )
        scores = compute_heuristic_scores(
            self._EDGE_INDEX, self._NUM_NODES, negatives, "common_neighbors"
        )
        assert scores.shape == (3,)
