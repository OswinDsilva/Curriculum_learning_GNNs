# Phase 2 — Baseline Training (Weeks 3–4)

## Goal

Train GCN and GAT on Cora, Citeseer, and PubMed using standard random
negative sampling. Establish reliable numerical baselines (with mean ± std
across 10 seeds) before any hard-negative or curriculum-learning logic is
introduced.

---

## Exit Criteria

- [ ] Training loss decreases monotonically (ignoring minor fluctuations).
- [ ] Cora val AUC > 0.85, test AUC > 0.83.
- [ ] Citeseer val AUC > 0.80.
- [ ] Results are exactly reproducible when the same seed is reused.
- [ ] Checkpoints save and reload correctly; reloaded model produces
      identical predictions.
- [ ] All metrics (AUC, AP, MRR, Hits@10/50/100) are logged per epoch.
- [ ] CSV result file written to `results/baseline/` at end of each run.

---

## Task 2.1 — Training Infrastructure

### File: `experiments/train_baseline.py`

#### Top-level entry point

```
python experiments/train_baseline.py \
    --dataset cora \
    --model gcn \
    --seed 0 \
    --save_dir results/baseline/
```

#### Core function signatures

```python
def train_epoch(
    model: LinkPredictor,
    data_dict: Dict,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    neg_ratio: int = 1,
) -> float
```
- Samples fresh random negatives every epoch (size = neg_ratio × num_train_pos).
- Concatenates pos + neg edges and their labels.
- Computes `BCEWithLogitsLoss`.
- Calls `loss.backward()` and `optimizer.step()`.
- Returns scalar loss value.

```python
def evaluate(
    model: LinkPredictor,
    data_dict: Dict,
    split: Literal["val", "test"],
    device: torch.device,
) -> Dict[str, float]
```
- No gradient computation (`torch.no_grad()`).
- Uses pre-fixed val/test negatives (from `prepare_link_prediction_data`).
- Returns dict: `{"auc": float, "ap": float, "mrr": float,
  "hits@10": float, "hits@50": float, "hits@100": float}`.

```python
def main() -> None
```
- Parses args (dataset, model, hidden_channels, out_channels, lr, dropout,
  epochs, seed, save_dir).
- Sets `torch.manual_seed(seed)` and `numpy.random.seed(seed)`.
- Loads data, builds model, trains, evaluates, saves checkpoint + results.

#### Training loop structure

```
for epoch in range(epochs):
    model.train()
    loss = train_epoch(...)

    if epoch % eval_every == 0:
        model.eval()
        val_metrics = evaluate(..., split="val")
        log_metrics(epoch, loss, val_metrics)
        save_checkpoint_if_best(model, val_metrics["auc"])

model.eval()
test_metrics = evaluate(..., split="test")
save_results(test_metrics, args)
```

#### Hyperparameters (defaults)

| Parameter        | Default | Notes                              |
|------------------|---------|------------------------------------|
| hidden_channels  | 128     | Try 256 for larger datasets        |
| out_channels     | 64      | Embedding dimension                |
| num_layers       | 2       | Fixed for Phase 2                  |
| lr               | 0.01    | Adam optimizer                     |
| dropout          | 0.5     |                                    |
| epochs           | 300     |                                    |
| eval_every       | 10      | Evaluate every N epochs            |
| neg_ratio        | 1       | 1 negative per positive            |
| decoder          | inner_product |                            |

---

## Task 2.2 — Evaluation Metrics

### File: `utils/metrics.py`

```python
def compute_auc_ap(
    pos_scores: Tensor,
    neg_scores: Tensor,
) -> Dict[str, float]
```
- Concatenates pos (label=1) and neg (label=0) scores.
- Returns `{"auc": float, "ap": float}`.
- Uses `sklearn.metrics.roc_auc_score` and `average_precision_score`.

```python
def compute_mrr(
    pos_scores: Tensor,
    neg_scores: Tensor,
) -> float
```
- For each positive score, compute its rank among all (pos + neg) scores.
- MRR = mean of 1/rank across all positives.
- Note: this is the "global" MRR; Phase 5 switches to per-positive ranking
  (HeaRT style).

```python
def compute_hits_at_k(
    pos_scores: Tensor,
    neg_scores: Tensor,
    k: int,
) -> float
```
- Hits@K = fraction of positives ranked in the top-K among all candidates.

```python
def compute_all_metrics(
    pos_scores: Tensor,
    neg_scores: Tensor,
    k_list: list[int] = [10, 50, 100],
) -> Dict[str, float]
```
- Returns all metrics in one call.
- Keys: `"auc"`, `"ap"`, `"mrr"`, `"hits@10"`, `"hits@50"`, `"hits@100"`.

**Important:** all functions accept raw logits (no sigmoid needed internally;
convert internally for sklearn calls).

---

## Task 2.3 — Experiment Logging

### File: `utils/logging_utils.py`

```python
class ExperimentLogger:
    def __init__(self, log_dir: str, experiment_name: str, use_tensorboard: bool = True)

    def log_epoch(self, epoch: int, loss: float, metrics: Dict[str, float]) -> None
        # Writes to TensorBoard and appends to in-memory list

    def save_csv(self, path: str) -> None
        # Dumps all logged epoch data to CSV

    def save_results_json(self, path: str, final_metrics: Dict) -> None
        # Saves hyperparams + final test metrics as JSON

    def close(self) -> None
        # Flushes TensorBoard writer
```

```python
class CheckpointManager:
    def __init__(self, checkpoint_dir: str, monitor: str = "val_auc")

    def save(self, model: nn.Module, epoch: int, metric_value: float) -> None
        # Save if metric_value > best so far

    def load_best(self, model: nn.Module) -> nn.Module
        # Load best checkpoint weights into model
```

### Result file naming convention

```
results/baseline/{dataset}_{model}_seed{seed}.json
checkpoints/{dataset}_{model}_seed{seed}_best.pt
logs/{dataset}_{model}_seed{seed}/events.out.tfevents.*
```

---

## Task 2.4 — Baseline Config

### File: `configs/baseline_config.yaml`

```yaml
dataset: cora
model: gcn
seed: 42
val_ratio: 0.05
test_ratio: 0.10
hidden_channels: 128
out_channels: 64
num_layers: 2
dropout: 0.5
lr: 0.01
epochs: 300
eval_every: 10
neg_ratio: 1
decoder: inner_product
save_dir: results/baseline
checkpoint_dir: checkpoints
log_dir: logs
```

---

## Task 2.5 — Results Aggregation (Baseline)

After all 10 seeds per (dataset, model) combination are finished:

```python
# aggregate_baseline.py  (simple script, not a full module)
# Read all JSON files matching results/baseline/{dataset}_{model}_seed*.json
# Compute mean and std for each metric
# Print and save results/baseline/{dataset}_{model}_summary.csv
```

Expected summary format:

| metric  | mean   | std    |
|---------|--------|--------|
| auc     | 0.8721 | 0.0043 |
| ap      | 0.8889 | 0.0051 |
| mrr     | 0.4120 | 0.0091 |
| hits@10 | 0.3311 | 0.0088 |
| hits@50 | 0.5543 | 0.0067 |
| hits@100| 0.6891 | 0.0055 |

---

## Verification Checkpoints

### After Task 2.1

Run one training run manually:
```bash
.venv/bin/python experiments/train_baseline.py \
    --dataset cora --model gcn --seed 0 --epochs 50
```
- Loss should drop in the first 50 epochs.
- No NaNs in loss or metrics.

### After Task 2.2

Unit-test the metrics functions:
```python
# Synthetic test: perfect predictor
pos_scores = torch.ones(100)
neg_scores = torch.zeros(100)
m = compute_all_metrics(pos_scores, neg_scores)
assert m["auc"] == 1.0
assert m["ap"] == 1.0
assert m["mrr"] == 1.0
assert m["hits@10"] == 1.0
```

### After Task 2.3

Verify checkpoint save/load:
```python
manager.save(model, epoch=50, metric_value=0.87)
model2 = build_model(...)
manager.load_best(model2)
assert (model.state_dict()["convs.0.weight"] == model2.state_dict()["convs.0.weight"]).all()
```

### Full Phase 2 Verification

Run the full 10-seed sweep on Cora and check:
- `results/baseline/cora_gcn_seed{0..9}.json` all exist.
- Mean test AUC > 0.83.
- Std < 0.01 (reasonable reproducibility).

---

## Debugging Tips

- **Loss not decreasing after 50 epochs**: check that negatives are being
  re-sampled each epoch (not reused); check learning rate is not too small.
- **AUC stuck near 0.5**: model is not learning; check that `train_edge_index`
  does NOT contain val/test edges and that labels are correctly assigned
  (positives = 1, negatives = 0).
- **MRR is unexpectedly high**: verify that `compute_mrr` is ranking against
  the full set of negatives, not just a small subset.
- **Checkpoint load fails**: ensure model architecture kwargs match between
  save and load; store them in the checkpoint dict.
- **TensorBoard shows no data**: verify `log_dir` is correct and
  `SummaryWriter.flush()` is called at the end of training.

---

## Estimated Time

| Task                          | Hours |
|-------------------------------|-------|
| Training loop infrastructure  | 5–7   |
| Metrics module                | 3–4   |
| Logging and checkpointing     | 4–5   |
| Config file + CLI wiring      | 2–3   |
| Running 10-seed experiments   | 3–4   |
| Debugging and verification    | 4–6   |
| **Total**                     | **21–29** |
