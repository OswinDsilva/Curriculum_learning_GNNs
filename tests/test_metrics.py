from __future__ import annotations

import numpy as np
import torch

from utils.metrics import (
    compute_all_metrics,
    compute_auc_ap,
    compute_hits_at_k,
    compute_mrr,
)


def test_perfect_predictor_auc_ap() -> None:
    """A classifier that gives pos scores all > neg scores should score AUC=AP=1."""
    pos = torch.ones(100)
    neg = torch.zeros(100)
    m = compute_auc_ap(pos, neg)
    assert abs(m["auc"] - 1.0) < 1e-6
    assert abs(m["ap"] - 1.0) < 1e-6


def test_random_predictor_auc_ap() -> None:
    """A random predictor should score AUC ≈ 0.5."""
    rng = np.random.default_rng(42)
    pos = torch.tensor(rng.random(500), dtype=torch.float32)
    neg = torch.tensor(rng.random(500), dtype=torch.float32)
    m = compute_auc_ap(pos, neg)
    assert 0.40 < m["auc"] < 0.65


def test_perfect_mrr() -> None:
    """Every positive outscores every negative → MRR = 1.0."""
    pos = torch.ones(50)
    neg = torch.zeros(200)
    assert abs(compute_mrr(pos, neg) - 1.0) < 1e-6


def test_mrr_is_reciprocal_rank() -> None:
    """Single positive with rank 2 (one negative scores higher) → MRR = 0.5."""
    pos = torch.tensor([0.5])
    neg = torch.tensor([0.9, 0.3, 0.1])
    assert abs(compute_mrr(pos, neg) - 0.5) < 1e-6


def test_hits_at_k_perfect() -> None:
    pos = torch.ones(10)
    neg = torch.zeros(990)
    assert abs(compute_hits_at_k(pos, neg, k=10) - 1.0) < 1e-6


def test_hits_at_k_zero() -> None:
    """All negatives score higher → 0 positives in top-K."""
    pos = torch.zeros(10)
    neg = torch.ones(990)
    assert compute_hits_at_k(pos, neg, k=10) == 0.0


def test_compute_all_metrics_keys() -> None:
    pos = torch.ones(50)
    neg = torch.zeros(50)
    m = compute_all_metrics(pos, neg)
    for key in ["auc", "ap", "mrr", "hits@10", "hits@50", "hits@100"]:
        assert key in m, f"Missing key: {key}"
        assert 0.0 <= m[key] <= 1.0 + 1e-6
