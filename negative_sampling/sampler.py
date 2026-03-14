"""
Difficulty-based negative edge sampler for curriculum learning.

The `DifficultyBasedSampler` assigns a difficulty score to every candidate
negative edge (based on a pre-computed heuristic or model score), then
provides three sampling strategies:

  sample_random        – uniform random (baseline behaviour)
  sample_by_difficulty – sample without replacement from a difficulty bucket
                         ("easy", "medium", "hard")
  sample_mixed         – blend across buckets according to a weight vector
  get_bucket_sizes     – introspect bucket population counts

Intended use in a curriculum training loop:

    sampler = DifficultyBasedSampler(candidate_negatives, scores, seed=42)
    # Early epochs: easy negatives
    neg = sampler.sample_mixed(n, weights=[0.6, 0.3, 0.1])
    # Late epochs: hard negatives
    neg = sampler.sample_mixed(n, weights=[0.1, 0.3, 0.6])
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
import torch


BucketName = Literal["easy", "medium", "hard"]


class DifficultyBasedSampler:
    """Difficulty-aware negative edge sampler.

    Candidates are partitioned into three equal-sized quantile buckets:
      easy   – bottom third of scores  (low structural similarity)
      medium – middle third
      hard   – top third               (high structural similarity)

    Args:
        candidates: LongTensor [2, N] of candidate negative edges.
        scores:     float32 numpy array [N] giving difficulty of each edge.
                    Higher score ↔ harder (more structurally similar to a
                    true edge).
        seed:       base random seed for reproducibility.
    """

    _BUCKETS: list[BucketName] = ["easy", "medium", "hard"]

    def __init__(
        self,
        candidates: torch.Tensor,
        scores: np.ndarray,
        seed: int = 0,
    ) -> None:
        if candidates.shape[1] != len(scores):
            raise ValueError(
                f"candidates has {candidates.shape[1]} edges but scores has "
                f"{len(scores)} entries."
            )
        self._candidates = candidates  # [2, N]
        self._scores = np.asarray(scores, dtype=np.float32)
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._build_buckets()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_buckets(self) -> None:
        """Partition candidate indices into three quantile buckets."""
        n = len(self._scores)
        sorted_idx = np.argsort(self._scores, kind="stable")  # ascending
        # Equal thirds; last bucket gets any remainder
        t1 = n // 3
        t2 = 2 * (n // 3)
        self._bucket_indices: dict[BucketName, np.ndarray] = {
            "easy":   sorted_idx[:t1],
            "medium": sorted_idx[t1:t2],
            "hard":   sorted_idx[t2:],
        }

    def get_bucket_sizes(self) -> dict[BucketName, int]:
        """Return number of candidates in each difficulty bucket."""
        return {b: int(len(idx)) for b, idx in self._bucket_indices.items()}

    # ------------------------------------------------------------------
    # Sampling methods
    # ------------------------------------------------------------------

    def sample_random(self, n: int, epoch_offset: int = 0) -> torch.Tensor:
        """Uniform random sample of n negative edges (with replacement if n
        exceeds pool size).

        Args:
            n:            number of edges to sample.
            epoch_offset: added to the base seed for epoch-level variation.

        Returns:
            LongTensor [2, n].
        """
        rng = np.random.default_rng(self._seed + epoch_offset)
        total = self._candidates.shape[1]
        replace = n > total
        chosen = rng.choice(total, size=n, replace=replace)
        return self._candidates[:, chosen]

    def sample_by_difficulty(
        self,
        n: int,
        bucket: BucketName,
        epoch_offset: int = 0,
    ) -> torch.Tensor:
        """Sample exclusively from one difficulty bucket.

        Args:
            n:            number of edges to sample.
            bucket:       "easy", "medium", or "hard".
            epoch_offset: added to base seed for variation.

        Returns:
            LongTensor [2, n].

        Raises:
            ValueError: if bucket name is invalid.
        """
        if bucket not in self._BUCKETS:
            raise ValueError(
                f"Unknown bucket '{bucket}'. Choose from {self._BUCKETS}."
            )
        pool = self._bucket_indices[bucket]
        if len(pool) == 0:
            raise RuntimeError(f"Bucket '{bucket}' is empty.")
        rng = np.random.default_rng(self._seed + epoch_offset)
        replace = n > len(pool)
        chosen = rng.choice(pool, size=n, replace=replace)
        return self._candidates[:, chosen]

    def sample_mixed(
        self,
        n: int,
        weights: Sequence[float] = (1 / 3, 1 / 3, 1 / 3),
        epoch_offset: int = 0,
    ) -> torch.Tensor:
        """Sample from all three buckets according to a weight vector.

        Args:
            n:            total number of edges to sample.
            weights:      length-3 sequence [w_easy, w_medium, w_hard].
                          Will be normalised to sum to 1.
            epoch_offset: added to base seed for variation.

        Returns:
            LongTensor [2, n].

        Raises:
            ValueError: if weights vector has wrong length or all-zero.
        """
        weights = list(weights)
        if len(weights) != 3:
            raise ValueError(
                f"weights must have length 3, got {len(weights)}."
            )
        total_w = sum(weights)
        if total_w <= 0:
            raise ValueError("weights must sum to a positive value.")

        # Normalise and compute integer counts (last bucket absorbs remainder)
        norm = [w / total_w for w in weights]
        counts = [int(round(n * w)) for w in norm]
        counts[-1] = n - sum(counts[:-1])  # ensure exact total

        parts: list[torch.Tensor] = []
        for i, (bucket, count) in enumerate(zip(self._BUCKETS, counts)):
            if count <= 0:
                continue
            pool = self._bucket_indices[bucket]
            if len(pool) == 0:
                # Fall back to random sample from all candidates
                rng = np.random.default_rng(self._seed + epoch_offset + i)
                total = self._candidates.shape[1]
                chosen = rng.choice(total, size=count, replace=True)
            else:
                rng = np.random.default_rng(self._seed + epoch_offset + i)
                replace = count > len(pool)
                chosen = rng.choice(pool, size=count, replace=replace)
            parts.append(self._candidates[:, chosen])

        if not parts:
            # n == 0 edge case
            return self._candidates[:, :0]

        return torch.cat(parts, dim=1)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_candidates(self) -> int:
        """Total number of candidate negative edges."""
        return self._candidates.shape[1]

    @property
    def scores(self) -> np.ndarray:
        """Raw difficulty scores (float32, shape [N])."""
        return self._scores

    @property
    def candidates(self) -> torch.Tensor:
        """All candidate negative edges (LongTensor [2, N])."""
        return self._candidates
