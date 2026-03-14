# Phase 5 — HeaRT Evaluation (Week 9)

## Goal

Implement the HeaRT (Hard Evaluation for Ranking in Transductive) protocol
from Chamberlain et al. (NeurIPS 2023). This replaces random test negatives
with hard negatives — node pairs with high structural similarity to the
positive test edges — producing a more discriminating and realistic benchmark.

---

## Exit Criteria

- [ ] `HeaRTEvaluator.generate_test_set()` produces hard negatives for every
      positive test edge.
- [ ] HeaRT evaluation scores are lower than random-negative scores for the
      same trained model (expected, confirms harder evaluation).
- [ ] The gap between random-negative MRR and HeaRT MRR is > 10 %.
- [ ] Rankings are correct: the positive edge should rank above its paired
      hard negatives at least more often than chance.
- [ ] `evaluate_model()` returns MRR and Hits@K using per-positive ranking
      (not global ranking).
- [ ] Both "random" and "HeaRT" evaluation modes accessible from the
      existing `evaluate()` function with a single flag.

---

## Background: HeaRT Protocol

Standard evaluation pools all test negatives globally and ranks positives
against all of them. HeaRT instead:

1. For each positive test edge `(u, v)`, finds the top-K most similar
   non-edges involving `u` or `v` as measured by a structural heuristic.
2. Ranks `(u, v)` against those K hard negatives only.
3. Reports MRR and Hits@K based on this per-positive ranking.

This is harder because the model must distinguish a true edge from
structurally similar but non-existent alternatives, mimicking real-world
recommendation scenarios.

---

## Task 5.1 — HeaRT Evaluator

### File: `negative_sampling/heart.py`

```python
class HeaRTEvaluator:
    def __init__(
        self,
        data: Data,
        heuristics: list[Literal["cn", "aa", "ra"]] = ["cn", "aa", "ra"],
        num_neg_per_pos: int = 100,
        precomputed_dir: str = "data/precomputed",
        seed: int = 42,
    ) -> None
```
- Loads pre-computed heuristic scores from `precomputed_dir`.
- If pre-computed scores are not available for a heuristic, raise a clear
  error instructing the user to run `scripts/precompute_scores.py` first.
- Stores the candidate pool indexed by score for fast top-K lookup.
- Builds an internal mapping from node → high-scoring non-edge candidates.

---

```python
def generate_test_set(
    self,
    pos_edge_index: Tensor,
) -> tuple[Tensor, list[Tensor]]
```
Returns:
- `pos_edge_index`: unchanged, shape `[2, N_pos]`.
- `hard_negs_per_pos`: list of N_pos tensors, each of shape
  `[2, num_neg_per_pos]`, containing the hard negatives paired to each
  positive edge.

How to select hard negatives for positive edge `(u, v)`:
1. Collect all candidate non-edges from the pre-computed pool that involve
   `u` or `v` as either endpoint.
2. Sort by heuristic score descending.
3. Take the top `num_neg_per_pos` candidates.
4. If fewer than `num_neg_per_pos` candidates exist, pad by sampling
   randomly from remaining non-edges (fallback).

If multiple heuristics are configured, select candidates that appear in
the top-K for **any** of the heuristics (union strategy).

**Important**: never include any positive edge (train, val, or test) in
the hard negatives list.

---

```python
def evaluate_model(
    self,
    model: LinkPredictor,
    data_dict: Dict,
    device: torch.device,
) -> Dict[str, float]
```
For each positive test edge `(u, v)`:
1. Score `(u, v)` with the model.
2. Score all `num_neg_per_pos` hard negatives.
3. Compute rank of positive among all `num_neg_per_pos + 1` candidates.
4. Accumulate reciprocal rank and Hits@K flags.

Returns:
```python
{
    "heart_mrr": float,
    "heart_hits@10": float,
    "heart_hits@50": float,
    "heart_hits@100": float,
}
```

No gradient computation inside (`torch.no_grad()`).

---

## Task 5.2 — Dual Evaluation Mode

### Modified `experiments/evaluate.py`

```python
def run_evaluation(
    model: LinkPredictor,
    data_dict: Dict,
    device: torch.device,
    heart_evaluator: HeaRTEvaluator | None = None,
) -> Dict[str, float]
```
- Always computes standard (random negative) metrics.
- If `heart_evaluator` is provided, also computes HeaRT metrics and merges
  into the returned dict.
- Returned dict keys:
  - Standard: `"auc"`, `"ap"`, `"mrr"`, `"hits@10"`, `"hits@50"`, `"hits@100"`
  - HeaRT: `"heart_mrr"`, `"heart_hits@10"`, `"heart_hits@50"`, `"heart_hits@100"`

---

### CLI script

```
python experiments/evaluate.py \
    --checkpoint checkpoints/cora_gcn_seed0_best.pt \
    --dataset cora \
    --model gcn \
    --heart \
    --heart_heuristic cn \
    --num_neg_per_pos 100
```

Prints both standard and HeaRT evaluation tables.

---

## Task 5.3 — HeaRT Integration Into Training Scripts

Both `train_baseline.py` and `train_curriculum.py` should accept a
`--heart` flag. When set:
- After the final test evaluation with random negatives, run
  `evaluate_model()` from `HeaRTEvaluator`.
- Log both result sets to the run's JSON result file under keys
  `"standard"` and `"heart"`.

---

## Verification Checklist

### Sanity check: harder evaluation produces lower scores

```
# Train a GCN baseline on Cora (10 seeds)
# For each seed, compare standard MRR with HeaRT MRR
# Expected: HeaRT MRR < standard MRR in all 10 seeds
```

Run the comparison:
```bash
python experiments/evaluate.py --checkpoint ... --heart
```
Log the ratio `heart_mrr / standard_mrr`. If this ratio is > 0.9 on Cora,
HeaRT negatives are not hard enough — lower `num_neg_per_pos` selection
threshold or ensure heuristic scores are correct.

### Per-positive ranking correctness

```python
# Synthetic test: perfect scorer
pos_scores = torch.tensor([1.0])
neg_scores = torch.zeros(100)  # 100 hard negatives all scored 0

# Positive should rank 1st
rank = compute_per_positive_rank(pos_scores, neg_scores)
assert rank == 1
assert compute_hits_at_k_per_pos(rank, k=10) == 1.0
```

### No positive leakage in hard negatives

```python
heart = HeaRTEvaluator(data, heuristics=["cn"])
pos_edge_index = data_dict["test_pos_edge_index"]
_, hard_negs_list = heart.generate_test_set(pos_edge_index)

all_pos_set = edge_index_to_edge_set(data.edge_index)
for neg_tensor in hard_negs_list:
    neg_set = edge_index_to_edge_set(neg_tensor)
    assert all_pos_set.isdisjoint(neg_set)
```

---

## Expected Benchmark Numbers (Approximate)

These are approximate results for a 2-layer GCN on Cora at standard
hyperparameters, for reference only.

| Metric      | Standard (random neg) | HeaRT (hard neg) |
|-------------|----------------------|------------------|
| MRR         | ~0.41                | ~0.25            |
| Hits@10     | ~0.33                | ~0.18            |
| Hits@50     | ~0.55                | ~0.36            |
| Hits@100    | ~0.69                | ~0.47            |

If your HeaRT numbers are close to the standard numbers, the hard negatives
are not hard enough. If they are near zero, there may be a bug in the ranking
logic.

---

## Debugging Tips

- **HeaRT MRR == standard MRR**: pre-computed scores may all be zero
  (empty common-neighbour sets). Verify score distribution before running
  evaluation.
- **Fallback padding dominates**: if most positives have fewer than
  `num_neg_per_pos` structural candidates, reduce `num_neg_per_pos` to 50
  or increase the candidate pool in `precompute_scores.py`.
- **Very slow HeaRT evaluation**: avoid re-building the heuristic index per
  call; it should be built once in `__init__` and reused.
- **Positive edge included in hard negatives**: ensure all three splits
  (train, val, and test positives) are excluded, not just test positives.
- **Rank computation off by one**: rank should be 1-based (best rank = 1).
  Verify using a synthetic case where the positive always scores highest.

---

## Estimated Time

| Task                          | Hours  |
|-------------------------------|--------|
| HeaRTEvaluator class          | 6–8    |
| Dual evaluation mode          | 2–3    |
| Integration into train scripts| 2–3    |
| Tests and verification        | 3–4    |
| **Total**                     | **13–18** |
