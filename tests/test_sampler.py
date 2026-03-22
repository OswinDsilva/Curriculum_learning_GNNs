"""
Tests for negative_sampling/sampler.py  (DifficultyBasedSampler)

Fixture: 30 candidate negative edges with scores 0..29 (ascending difficulty),
so the bucket partition is:
  easy   → indices 0..9   (scores 0–9)
  medium → indices 10..19 (scores 10–19)
  hard   → indices 20..29 (scores 20–29)
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from negative_sampling.sampler import DifficultyBasedSampler


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

N = 30  # total candidates


@pytest.fixture()
def sampler() -> DifficultyBasedSampler:
    """Sampler with 30 candidates and scores 0..29."""
    # Candidates: src = index, dst = index+1 (dummy pairs)
    candidates = torch.stack(
        [
            torch.arange(N),
            torch.arange(1, N + 1),
        ]
    )  # [2, 30]
    scores = np.arange(N, dtype=np.float32)  # 0, 1, …, 29
    return DifficultyBasedSampler(candidates, scores, seed=0)


# ---------------------------------------------------------------------------
# Bucket partitioning
# ---------------------------------------------------------------------------


class TestBuckets:
    def test_bucket_sizes_equal_thirds(self, sampler):
        sizes = sampler.get_bucket_sizes()
        assert sizes["easy"] == N // 3
        assert sizes["medium"] == N // 3
        assert sizes["hard"] == N - 2 * (N // 3)

    def test_total_candidates(self, sampler):
        assert sampler.num_candidates == N

    def test_scores_preserved(self, sampler):
        assert sampler.scores.shape == (N,)
        assert sampler.scores[0] == pytest.approx(0.0)
        assert sampler.scores[-1] == pytest.approx(float(N - 1))


# ---------------------------------------------------------------------------
# sample_random
# ---------------------------------------------------------------------------


class TestSampleRandom:
    def test_output_shape(self, sampler):
        out = sampler.sample_random(10)
        assert out.shape == (2, 10)

    def test_dtype(self, sampler):
        out = sampler.sample_random(5)
        assert out.dtype == torch.long

    def test_indices_in_range(self, sampler):
        out = sampler.sample_random(20)
        # src values should be in 0..N-1 (our dummy encoding)
        assert (out[0] >= 0).all() and (out[0] < N).all()

    def test_reproducible_with_same_offset(self, sampler):
        a = sampler.sample_random(10, epoch_offset=7)
        b = sampler.sample_random(10, epoch_offset=7)
        assert torch.equal(a, b)

    def test_different_offset_gives_different_samples(self, sampler):
        a = sampler.sample_random(10, epoch_offset=0)
        b = sampler.sample_random(10, epoch_offset=1)
        # With high probability these differ (probability of exact equality
        # is astronomically small for 10 samples from 30 candidates)
        assert not torch.equal(a, b)


# ---------------------------------------------------------------------------
# sample_by_difficulty
# ---------------------------------------------------------------------------


class TestSampleByDifficulty:
    def test_easy_bucket_returns_low_scores(self, sampler):
        out = sampler.sample_by_difficulty(10, "easy")
        # src encodes the original index; easy bucket is indices 0–9
        assert (out[0] < N // 3).all()

    def test_hard_bucket_returns_high_scores(self, sampler):
        out = sampler.sample_by_difficulty(10, "hard")
        assert (out[0] >= 2 * (N // 3)).all()

    def test_output_shape(self, sampler):
        out = sampler.sample_by_difficulty(5, "medium")
        assert out.shape == (2, 5)

    def test_invalid_bucket_raises(self, sampler):
        with pytest.raises(ValueError, match="Unknown bucket"):
            sampler.sample_by_difficulty(5, "extreme")  # type: ignore[arg-type]

    def test_with_replacement_when_n_exceeds_pool(self, sampler):
        # Pool size = 10; requesting 25 → must use replacement
        out = sampler.sample_by_difficulty(25, "easy")
        assert out.shape == (2, 25)

    def test_reproducible(self, sampler):
        a = sampler.sample_by_difficulty(8, "hard", epoch_offset=3)
        b = sampler.sample_by_difficulty(8, "hard", epoch_offset=3)
        assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# sample_mixed
# ---------------------------------------------------------------------------


class TestSampleMixed:
    def test_total_output_size(self, sampler):
        out = sampler.sample_mixed(21, weights=[1 / 3, 1 / 3, 1 / 3])
        assert out.shape[0] == 2
        assert out.shape[1] == 21

    def test_all_easy_weights(self, sampler):
        # weight = [1, 0, 0] → should only return easy candidates (indices 0–9)
        out = sampler.sample_mixed(9, weights=[1.0, 0.0, 0.0])
        # src values (which encode original index) must all be < N//3
        assert (out[0] < N // 3).all(), f"Got src values: {out[0].tolist()}"

    def test_all_hard_weights(self, sampler):
        out = sampler.sample_mixed(9, weights=[0.0, 0.0, 1.0])
        assert (out[0] >= 2 * (N // 3)).all()

    def test_wrong_weights_length_raises(self, sampler):
        with pytest.raises(ValueError, match="length 3"):
            sampler.sample_mixed(10, weights=[0.5, 0.5])

    def test_zero_weights_raises(self, sampler):
        with pytest.raises(ValueError, match="positive"):
            sampler.sample_mixed(10, weights=[0.0, 0.0, 0.0])

    def test_unnormalised_weights_accepted(self, sampler):
        # Unnormalised [2, 2, 2] should behave like [1/3, 1/3, 1/3]
        out = sampler.sample_mixed(12, weights=[2, 2, 2])
        assert out.shape == (2, 12)

    def test_reproducible(self, sampler):
        a = sampler.sample_mixed(15, weights=[0.2, 0.3, 0.5], epoch_offset=5)
        b = sampler.sample_mixed(15, weights=[0.2, 0.3, 0.5], epoch_offset=5)
        assert torch.equal(a, b)

    def test_zero_n(self, sampler):
        out = sampler.sample_mixed(0, weights=[1 / 3, 1 / 3, 1 / 3])
        assert out.shape[1] == 0


# ---------------------------------------------------------------------------
# Mismatched construction arguments
# ---------------------------------------------------------------------------


def test_mismatched_candidates_scores_raises():
    candidates = torch.zeros(2, 10, dtype=torch.long)
    scores = np.zeros(5, dtype=np.float32)  # wrong length
    with pytest.raises(ValueError, match="scores has 5"):
        DifficultyBasedSampler(candidates, scores)
