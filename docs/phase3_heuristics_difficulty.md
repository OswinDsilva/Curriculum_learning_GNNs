# Phase 3 — Heuristics & Difficulty Scoring (Weeks 5–6)

## Goal

Pre-compute graph-structural similarity scores for all candidate negative
edges and build a sampler that can serve negatives at a specified difficulty
level. This is the foundation that the curriculum scheduler (Phase 4) plugs
into directly.

---

## Exit Criteria

- [ ] `heuristics.py` functions match NetworkX reference implementations for
      all three heuristics (CN, AA, RA) on a small known graph.
- [ ] Score distributions are computed and visualisable: most pairs score 0
      (sparse graphs), long right tail.
- [ ] Pre-computed scores load from disk in < 1 second for Cora.
- [ ] `DifficultyBasedSampler.sample_by_difficulty()` returns exactly the
      requested number of negatives for each difficulty level.
- [ ] `sample_mixed()` respects the provided difficulty ratios (± 1 sample
      rounding tolerance).
- [ ] No positive edge appears in any sample returned by the sampler.

---

## Task 3.1 — Graph Heuristics

### File: `negative_sampling/heuristics.py`

These heuristics assign a structural similarity score to any node pair
`(u, v)`. Higher score = more similar to connected nodes = harder negative.

---

#### Common Neighbors (CN)

```python
def common_neighbors_score(G: nx.Graph, u: int, v: int) -> float
```
- Returns the number of nodes that are neighbours of both `u` and `v`.
- Formula: `|N(u) ∩ N(v)|`
- Range: `[0, min(deg(u), deg(v))]`

---

#### Adamic-Adar (AA)

```python
def adamic_adar_score(G: nx.Graph, u: int, v: int) -> float
```
- Weighted count of common neighbours; rare-neighbour common nodes count more.
- Formula: `Σ_{w ∈ N(u)∩N(v)} 1 / log(deg(w))`
- Returns 0.0 if there are no common neighbours or if any common neighbour
  has degree 1 (log(1) = 0, so skip those to avoid division by zero).

---

#### Resource Allocation (RA)

```python
def resource_allocation_score(G: nx.Graph, u: int, v: int) -> float
```
- Similar to AA but uses reciprocal degree instead of log.
- Formula: `Σ_{w ∈ N(u)∩N(v)} 1 / deg(w)`

---

#### Personalized PageRank (PPR) — optional, Phase 3 extension

```python
def personalized_pagerank_score(
    G: nx.Graph,
    u: int,
    v: int,
    alpha: float = 0.15,
) -> float
```
- Returns the PPR value of node `v` when the restart distribution is
  concentrated at `u`.
- Use `nx.pagerank(G, alpha=alpha, personalization={u: 1.0})` internally.
- Warning: this is O(N) per query; cache or batch before using in the sampler.

---

#### Vectorised batch computation

```python
def compute_heuristic_scores(
    data: Data,
    edge_index: Tensor,
    heuristic: Literal["cn", "aa", "ra", "ppr"] = "cn",
) -> Tensor
```
- Converts `data` to a NetworkX graph once, then computes scores for all
  edges in `edge_index`.
- Returns a 1-D tensor of scores, one per edge, same order as `edge_index`.
- Internally builds a NetworkX graph from `data.edge_index` using
  `nx.from_edgelist()`.

**Performance note**: for Cora (~5K nodes, ~5K edges), computing CN for
10,000 pairs takes roughly 0.5 s. For PubMed (~20K nodes, ~88K edges), plan
for up to 60 s. Pre-computation is mandatory; do NOT call this during training.

---

## Task 3.2 — Pre-computation Script

### File: `scripts/precompute_scores.py`

```
python scripts/precompute_scores.py \
    --dataset cora \
    --heuristic cn \
    --num_neg_candidates 500000 \
    --seed 42 \
    --output_dir data/precomputed/
```

What it does:
1. Load dataset and edge splits with `prepare_link_prediction_data`.
2. Sample a large candidate pool of non-edges (e.g. 500,000 for Cora).
3. Compute heuristic scores for every candidate.
4. Save as a compressed numpy file: `data/precomputed/cora_cn.npz`
   containing arrays `edges` (shape `[N, 2]`) and `scores` (shape `[N]`).
5. Print score distribution summary (min, max, mean, percentiles 50/75/90/99).

### Storage format

```python
np.savez_compressed(
    "data/precomputed/cora_cn.npz",
    edges=edges_array,    # int32, shape [N, 2]
    scores=scores_array,  # float32, shape [N]
)
```

Loading must complete in < 1 s for any dataset used in training.

### Score distribution summary (expected for CN on Cora)

| Percentile | Score |
|------------|-------|
| 50th       | 0     |
| 75th       | 0     |
| 90th       | 1     |
| 99th       | 3     |
| Max        | ~8    |

This confirms the long-tail distribution; most random pairs share zero
common neighbours.

---

## Task 3.3 — Difficulty-Based Sampler

### File: `negative_sampling/sampler.py`

```python
class DifficultyBasedSampler:
    def __init__(
        self,
        precomputed_path: str,
        difficulty_thresholds: list[float] = [0, 2, 5],
        pos_edge_set: set[tuple[int, int]] | None = None,
    ) -> None
```

- Loads `edges` and `scores` from the `.npz` file.
- Partitions the candidate pool into three buckets using `difficulty_thresholds`:
  - **Easy**: score in `[thresholds[0], thresholds[1])`
  - **Medium**: score in `[thresholds[1], thresholds[2])`
  - **Hard**: score `>= thresholds[2]`
- Optionally filters out any positive edges from the pool on initialisation.
- Stores the three partitioned index arrays for fast sampling.

---

```python
def sample_by_difficulty(
    self,
    num_samples: int,
    difficulty: Literal["easy", "medium", "hard"],
    seed: int | None = None,
) -> Tensor
```
- Returns `[2, num_samples]` tensor of negative edges drawn from the
  specified difficulty bucket.
- Samples uniformly (with or without replacement depending on bucket size).
- Raises `ValueError` if the bucket has fewer than `num_samples` edges and
  replacement is disabled.

---

```python
def sample_mixed(
    self,
    num_samples: int,
    difficulty_ratios: list[float],
    seed: int | None = None,
) -> Tensor
```
- `difficulty_ratios` is a length-3 list `[easy_frac, medium_frac, hard_frac]`
  that must sum to 1.0 (± 1e-6 tolerance).
- Computes the exact count per bucket (with ceiling adjustment to hit
  `num_samples` exactly).
- Calls `sample_by_difficulty` for each bucket and concatenates.
- Returns `[2, num_samples]`.
- Example: `[0.7, 0.3, 0.0]` means 70 % easy, 30 % medium, 0 % hard.

---

```python
def get_bucket_sizes(self) -> Dict[str, int]
```
- Returns `{"easy": int, "medium": int, "hard": int}`.
- Useful for sanity checks and for deciding whether a bucket is large enough
  to sample from without replacement.

---

### Threshold choice guidance

Default thresholds `[0, 2, 5]` are chosen for CN on citation networks.
If using AA or RA (continuous values), recalibrate thresholds to the
50th / 75th / 90th percentiles of the score distribution.

For each dataset + heuristic combination, log the bucket sizes after
initialisation. A reasonable target:
- Easy bucket: ≥ 60 % of the candidate pool.
- Medium bucket: 20–35 %.
- Hard bucket: 5–20 %.

---

## Verification Checklist

### Heuristic correctness

```python
import networkx as nx

G = nx.Graph()
G.add_edges_from([(0,1), (1,2), (2,3), (0,2)])

# CN: (0,3) share one common neighbour (node 2)
assert common_neighbors_score(G, 0, 3) == 1.0

# AA: (0,3) via node 2 which has degree 3 → 1/log(3) ≈ 0.910
assert abs(adamic_adar_score(G, 0, 3) - 1/math.log(3)) < 1e-6

# RA: (0,3) via node 2 → 1/3 ≈ 0.333
assert abs(resource_allocation_score(G, 0, 3) - 1/3) < 1e-6
```

### Sampler correctness

```python
sampler = DifficultyBasedSampler("data/precomputed/cora_cn.npz")
for diff in ["easy", "medium", "hard"]:
    sample = sampler.sample_by_difficulty(256, difficulty=diff)
    assert sample.shape == (2, 256)

mixed = sampler.sample_mixed(1000, [0.5, 0.3, 0.2])
assert mixed.shape == (2, 1000)
```

### No leakage

```python
neg_set = edge_index_to_edge_set(mixed)
assert pos_edge_set.isdisjoint(neg_set)
```

---

## Debugging Tips

- **Adamic-Adar division by zero**: node with degree 1 has `log(1)=0`.
  Guard with `if deg > 1` before dividing.
- **Empty hard bucket**: common on sparse citation networks; scores rarely
  exceed 5. Reduce the hard threshold or use AA/RA which produce continuous
  scores and spread better.
- **Slow pre-computation**: NetworkX is single-threaded. For PubMed, split
  the candidate pool into chunks and use `multiprocessing.Pool` if time is
  critical.
- **Score file too large**: for 500,000 candidates at float32 + int32, file
  size is roughly 8 MB — acceptable. Do not use float64.
- **Sample ratio rounding**: when allocating counts across buckets, use
  `round()` on the first two fractions and give all remainder to the last
  bucket so the total is exactly `num_samples`.

---

## Estimated Time

| Task                          | Hours  |
|-------------------------------|--------|
| Heuristic functions           | 4–5    |
| Batch score computation       | 2–3    |
| Pre-computation script        | 3–4    |
| DifficultyBasedSampler        | 5–6    |
| Tests and verification        | 4–5    |
| **Total**                     | **18–23** |
