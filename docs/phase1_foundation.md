# Phase 1 — Foundation (Weeks 1–2)

## Goal

By the end of Phase 1 the project must be runnable end-to-end on a single
graph (Cora) with a working data pipeline, two baseline GNN models, and a
passing test suite. Nothing related to hard negatives or curriculum is needed
yet.

---

## Exit Criteria (must all pass before moving to Phase 2)

- [ ] `python scripts/verify_environment.py` exits 0.
- [ ] `python experiments/data_exploration.py --dataset cora` prints sane statistics.
- [ ] `pytest -q` passes: all environment, data utility, model shape, and
      integration smoke tests.
- [ ] GCN and GAT forward/backward passes run on Cora without shape, device,
      or NaN errors.
- [ ] Edge splits satisfy: train + val + test = total unique undirected
      positive edges.
- [ ] Validation/test negatives are verified non-edges (no self-loops, no
      overlap with positives).
- [ ] Same seed produces identical splits and model outputs across two runs.

---

## Task 1.1 — Environment Setup

### What needs to be done

1. Create a project-local Python 3.11 virtual environment using `uv`:
   ```bash
   uv venv --python 3.11 .venv
   ```
2. Install all pinned dependencies:
   ```bash
   uv pip install --python .venv/bin/python -r requirements.txt
   ```
3. Run the environment verifier:
   ```bash
   .venv/bin/python scripts/verify_environment.py
   ```

### File: `scripts/verify_environment.py`

Responsibilities:
- Import `torch` and `torch_geometric`, print versions.
- Report whether CUDA is available and name the GPU if present.
- Load `Planetoid("Cora")` and print node/edge/feature counts.
- Exit with code `0` on success, non-zero on any failure.

### File: `requirements.txt`

Pinned stack for Python 3.11:
```
torch==2.2.2
torchvision==0.17.2
torchaudio==2.2.2
torch-geometric==2.5.3
networkx==3.2.1
numpy==1.26.4
pandas==2.2.1
scikit-learn==1.4.1.post1
matplotlib==3.8.3
tqdm==4.66.2
pyyaml==6.0.1
tensorboard==2.16.2
pytest==8.0.2
black==24.2.0
flake8==7.0.0
```

### Common failures and fixes

| Failure | Fix |
|---------|-----|
| `torch==2.2.2` not found | Python version mismatch; ensure Python 3.11 is used |
| PyG import error | Reinstall PyG against the exact Torch version |
| Dataset download 403 | Check write permissions on `data/`; retry with VPN off |
| `CUDA available: False` on GPU machine | Normal if CUDA wheels not installed; continue with CPU |

### Estimated time: 4–6 hours

---

## Task 1.2 — Data Pipeline

### File: `utils/data_utils.py`

#### Function signatures

```python
DatasetName = Literal[
    "cora", "citeseer", "pubmed", "coauthor-cs", "coauthor-physics"
]

def load_dataset(name: DatasetName, root: str | Path = "data") -> InMemoryDataset
```
- Maps name to `Planetoid` or `Coauthor` constructor.
- Raises `ValueError` with list of supported names on bad input.

```python
def canonicalize_edge(u: int, v: int) -> tuple[int, int]
```
- Returns `(min(u,v), max(u,v))` to give each undirected edge a unique key.

```python
def to_undirected_unique(edge_index: Tensor, num_nodes: int) -> Tensor
```
- Converts directed edge_index to unique undirected canonical edges.
- Removes self-loops.
- Returns shape `[2, num_unique_undirected_edges]`.

```python
def edge_index_to_edge_set(edge_index: Tensor) -> set[tuple[int, int]]
```
- Returns a set of canonicalized edge tuples; used for fast non-edge checks.

```python
def split_edges_for_link_prediction(
    data: Data,
    val_ratio: float = 0.05,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> Dict[str, Tensor]
```
Returns dict with keys:
- `train_pos_edge_index` — shape `[2, n_train]`
- `val_pos_edge_index` — shape `[2, n_val]`
- `test_pos_edge_index` — shape `[2, n_test]`
- `train_edge_index` — bidirectional message-passing graph (train positives only)

Important invariants:
- Uses unique undirected edges only.
- Shuffled once with the provided seed; deterministic.
- Val/test positives are NOT present in `train_edge_index`.

```python
def get_random_negatives(
    edge_index: Tensor,
    num_nodes: int,
    num_samples: int,
    seed: Optional[int] = None,
) -> Tensor
```
- Returns `[2, num_samples]` of valid non-edges.
- No self-loops, no overlap with `edge_index`, no duplicates.
- Uses rejection sampling with over-sampling factor ×3 minimum batch.

```python
def prepare_link_prediction_data(
    dataset_name: DatasetName,
    root: str | Path = "data",
    val_ratio: float = 0.05,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> Dict[str, Tensor | Data | int]
```
Returns everything a training script needs:
- `data`, `x`, `train_edge_index`
- `train_pos_edge_index`, `val_pos_edge_index`, `test_pos_edge_index`
- `val_neg_edge_index`, `test_neg_edge_index`
- `num_nodes`, `num_features`

Note: training negatives are NOT included here; they are re-sampled each
epoch during training.

### Validation invariants to assert during testing

1. `len(train_set ∩ val_set) == 0`
2. `len(train_set ∩ test_set) == 0`
3. `len(val_set ∩ test_set) == 0`
4. `n_train + n_val + n_test == total_unique_undirected_edges`
5. All val/test negatives are true non-edges.
6. No negative edge has `u == v`.
7. Two runs with the same seed produce byte-identical outputs.

### Estimated time: 6–8 hours

---

## Task 1.3 — Data Exploration Script

### File: `experiments/data_exploration.py`

CLI usage:
```bash
python experiments/data_exploration.py --dataset cora [--root data] [--seed 42]
```

Expected output:
```
Dataset: cora
Nodes: 2708
Unique undirected edges: 5278
Features: 1433
Average degree: 3.8979
Density: 0.001439

Train positives: 4489
Val positives: 264
Test positives: 525
Val negatives: 264
Test negatives: 525
```

Internal helpers:
- `compute_graph_statistics(data) -> Dict[str, float]` — returns num_nodes,
  num_edges_undirected, num_features, avg_degree, density.
- `summarize_link_prediction_splits(split_dict) -> Dict[str, int]` — returns
  train/val/test positive and negative counts.

### Estimated time: 2–3 hours

---

## Task 1.4 — Base Model Abstraction

### File: `models/base.py`

#### Decoder types

```
DecoderType = Literal["inner_product", "hadamard_mlp", "mlp"]
```

| Decoder       | Formula                              | Parameters          |
|---------------|--------------------------------------|---------------------|
| inner_product | `(z[u] * z[v]).sum(-1)`              | None (free)         |
| hadamard_mlp  | `MLP(z[u] * z[v])` → scalar          | MLP(out → out → 1)  |
| mlp           | `MLP(concat[z[u], z[v]])` → scalar   | MLP(2·out → out → 1)|

**Important:** `decode()` must return raw logits (no sigmoid), because
`BCEWithLogitsLoss` is used during training.

#### Abstract base class

```python
class LinkPredictor(nn.Module, ABC):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.5, decoder="inner_product")

    @abstractmethod
    def encode(self, x: Tensor, edge_index: Tensor) -> Tensor
        # Must return shape [num_nodes, out_channels]

    def decode(self, z: Tensor, edge_label_index: Tensor) -> Tensor
        # Must return shape [num_edges]  (raw logits, no sigmoid)

    def forward(self, x, edge_index, edge_label_index) -> Tensor
        # Calls encode then decode; shape [num_edges]
```

#### Supporting decoder modules

```python
class HadamardMLPDecoder(nn.Module):
    # Linear(out) → ReLU → Linear(1)  applied to z[u]*z[v]

class EdgeMLPDecoder(nn.Module):
    # Linear(2*out) → ReLU → Linear(1)  applied to concat(z[u], z[v])
```

### Estimated time: 4–5 hours

---

## Task 1.5 — GCN Implementation

### File: `models/gcn.py`

```python
class GCN(LinkPredictor):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.5, decoder="inner_product")
    def encode(self, x, edge_index) -> Tensor  # [num_nodes, out_channels]
```

Architecture:
- Stack of `GCNConv` layers.
- ReLU + Dropout between all hidden layers.
- No activation after final conv.
- Raises `ValueError` if `num_layers < 2`.

Layer sizing example for `num_layers=3`:
```
GCNConv(in → hidden) → ReLU → Dropout
GCNConv(hidden → hidden) → ReLU → Dropout
GCNConv(hidden → out)
```

Smoke check:
```python
z = model.encode(data.x, train_edge_index)
assert z.shape == (data.num_nodes, out_channels)
scores = model.decode(z, edge_label_index[:, :32])
assert scores.shape == (32,)
assert torch.isfinite(scores).all()
```

### Estimated time: 3–4 hours

---

## Task 1.6 — GAT Implementation

### File: `models/gat.py`

```python
class GAT(LinkPredictor):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, heads=4, dropout=0.5, decoder="inner_product")
    def encode(self, x, edge_index) -> Tensor  # [num_nodes, out_channels]
```

Architecture:
- Hidden layers use `GATConv` with `concat=True`; each multiplies
  hidden_channels by `heads`.
- Final layer uses `heads=1, concat=False` to produce exactly `out_channels`.
- ELU activation between hidden layers.
- Raises `ValueError` if `num_layers < 2`.

Layer sizing example for `num_layers=2, hidden=16, heads=4`:
```
GATConv(in → 16, heads=4, concat=True)  → output dim = 64
ELU → Dropout
GATConv(64 → out, heads=1, concat=False) → output dim = out
```

### Estimated time: 3–4 hours

---

## Tests

### `tests/test_environment.py`
- `test_torch_geometric_imports` — imports torch and torch_geometric.
- `test_cora_loads` — loads Cora and checks `num_nodes > 0`.

### `tests/test_data_utils.py`
- `test_edge_splits_are_disjoint` — train/val/test sets have no overlap.
- `test_split_reproducibility` — same seed → identical tensors.
- `test_random_negatives_are_true_non_edges` — 512 negatives, all valid.

### `tests/test_models.py`
- `test_gcn_forward_shapes` — encode → `[num_nodes, 32]`; decode → `[32]`.
- `test_gat_forward_shapes` — same invariants for GAT.

### `tests/test_integration_smoke.py`
- `test_single_training_step_smoke` — full forward + BCEWithLogitsLoss +
  backward + optimizer step on Cora; asserts `isfinite(loss)`.

---

## Build Order

Implement in this order to minimise rework:

1. `requirements.txt`
2. `scripts/verify_environment.py`
3. `utils/data_utils.py` — `load_dataset`, canonicalization, `to_undirected_unique`
4. `utils/data_utils.py` — `split_edges_for_link_prediction`
5. `utils/data_utils.py` — `get_random_negatives`
6. `utils/data_utils.py` — `prepare_link_prediction_data`
7. `experiments/data_exploration.py`
8. `models/base.py` — decoders + abstract base
9. `models/gcn.py`
10. `models/gat.py`
11. All test files
12. Integration smoke test last (depends on everything above)

---

## Debugging Tips

- **Split leakage**: print `len(train_set & val_set)` immediately if splits
  seem wrong. The most common cause is forgetting to canonicalize edges before
  set membership checks.
- **GAT dimension mismatch**: if `encode` raises size mismatch, check that
  intermediate layers receive `hidden_channels * heads` as input, not just
  `hidden_channels`.
- **NaN loss on step 1**: usually means logits are on a different device from
  labels. Print `tensor.device` for all inputs.
- **Dangling self-loops**: PyG's Planetoid data includes symmetric duplicate
  edges by default. Always pass through `to_undirected_unique` before
  splitting.
- **Negative sampling hangs**: occurs when the graph is very dense and
  candidates keep colliding. Increase the over-sample factor or switch to
  pre-shuffled index arrays for sampling.
