from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import Tensor


def compute_auc_ap(pos_scores: Tensor, neg_scores: Tensor) -> Dict[str, float]:
    """Compute AUC-ROC and Average Precision from positive and negative edge scores."""
    scores = torch.cat([pos_scores, neg_scores]).detach().cpu().numpy()
    labels = np.concatenate([
        np.ones(len(pos_scores), dtype=np.int32),
        np.zeros(len(neg_scores), dtype=np.int32),
    ])
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "ap": float(average_precision_score(labels, scores)),
    }


def compute_mrr(pos_scores: Tensor, neg_scores: Tensor) -> float:
    """
    Compute global Mean Reciprocal Rank.

    Each positive is ranked against the full set of negatives.
    Rank is 1-based; best possible MRR = 1.0.
    """
    pos_np = pos_scores.detach().cpu().numpy()
    neg_np = neg_scores.detach().cpu().numpy()

    reciprocal_ranks: list[float] = []
    for ps in pos_np:
        # rank = 1 + number of negatives that score strictly higher
        rank = 1 + int(np.sum(neg_np > ps))
        reciprocal_ranks.append(1.0 / rank)

    return float(np.mean(reciprocal_ranks))


def compute_hits_at_k(pos_scores: Tensor, neg_scores: Tensor, k: int) -> float:
    """
    Compute Hits@K.

    Returns the fraction of positive edges ranked in the top-K
    among all (positives + negatives).
    """
    pos_np = pos_scores.detach().cpu().numpy()
    neg_np = neg_scores.detach().cpu().numpy()

    all_scores = np.concatenate([pos_np, neg_np])
    threshold = np.sort(all_scores)[::-1][min(k - 1, len(all_scores) - 1)]

    hits = int(np.sum(pos_np >= threshold))
    return float(hits) / max(len(pos_np), 1)


def compute_all_metrics(
    pos_scores: Tensor,
    neg_scores: Tensor,
    k_list: List[int] = [10, 50, 100],
) -> Dict[str, float]:
    """Compute AUC, AP, MRR, and Hits@K for all K values in one call."""
    metrics = compute_auc_ap(pos_scores, neg_scores)
    metrics["mrr"] = compute_mrr(pos_scores, neg_scores)
    for k in k_list:
        metrics[f"hits@{k}"] = compute_hits_at_k(pos_scores, neg_scores, k)
    return metrics
