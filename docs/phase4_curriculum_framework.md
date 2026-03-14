# Phase 4 — Curriculum Framework (Weeks 7–8)

## Goal

Build the curriculum learning wrapper that controls which difficulty mix is
used for negative sampling at each training epoch, advancing through phases
as the model's competence on the validation set improves.

---

## Exit Criteria

- [ ] Curriculum advances through all 4 phases during a full training run.
- [ ] Phase transitions are logged with epoch numbers.
- [ ] Competence thresholds are actually met before each transition (not just
      triggered by a fixed epoch count).
- [ ] Training does NOT crash or degrade when the difficulty mix changes.
- [ ] Adaptive mode (competence-driven) and fixed mode (epoch-driven) both
      work.
- [ ] Ablation mode: fixed ratios with no progression also works.
- [ ] All phase-transition events written to `logs/` and to a JSON results
      file.

---

## Background: The 4-Phase Curriculum

| Phase | Difficulty Mix (easy/medium/hard) | Advance when val AUC ≥ |
|-------|----------------------------------|------------------------|
| 0     | 100 / 0 / 0                      | 0.75                   |
| 1     | 70 / 30 / 0                      | 0.85                   |
| 2     | 30 / 40 / 30                     | 0.90                   |
| 3     | 0 / 30 / 70                      | — (final phase)        |

The model starts on pure easy negatives. As it learns, the competence meter
tracks recent validation AUC. Once the smoothed AUC exceeds the threshold,
the scheduler advances to the next phase, increasing hard-negative exposure.

---

## Task 4.1 — Competence Measurement

### File: `curriculum/competence.py`

```python
class CompetenceMeter:
    def __init__(
        self,
        metric: str = "val_auc",
        window_size: int = 5,
    ) -> None
```
- `metric`: name for logging purposes only.
- `window_size`: number of recent values to average for smoothed competence.
- Internally stores a deque of the most recent metric values.

---

```python
def update(self, metric_value: float) -> None
```
- Appends `metric_value` to the history deque.
- Drops oldest value when the deque exceeds `window_size`.

---

```python
def get_competence(self) -> float
```
- Returns the unweighted moving average of values in the deque.
- Returns 0.0 if the history is empty.

---

```python
def is_threshold_reached(self, threshold: float) -> bool
```
- Returns `True` if `get_competence() >= threshold`.
- Used by the scheduler inside `should_advance()`.

---

```python
def reset(self) -> None
```
- Clears history. Call this when advancing to a new phase to avoid carrying
  over stale competence values.

---

### Design notes

- A window of 5 is appropriate for `eval_every=10` (covers 50 training
  epochs).
- Do NOT reset the competence meter between phases — the scheduler can call
  `reset()` explicitly only if a fresh measurement window is desired.
- Optionally support exponential moving average (EMA) as an alternative to
  simple moving average; controlled by a `smoothing` parameter.

---

## Task 4.2 — Curriculum Scheduler

### File: `curriculum/scheduler.py`

```python
@dataclass
class CurriculumPhase:
    difficulty_ratios: list[float]     # [easy_frac, medium_frac, hard_frac]
    threshold: float | None            # val AUC needed to advance; None = final
    name: str = ""                     # optional human-readable label
```

---

```python
DEFAULT_PHASES = [
    CurriculumPhase([1.0, 0.0, 0.0], threshold=0.75, name="easy_only"),
    CurriculumPhase([0.7, 0.3, 0.0], threshold=0.85, name="easy_medium"),
    CurriculumPhase([0.3, 0.4, 0.3], threshold=0.90, name="mixed"),
    CurriculumPhase([0.0, 0.3, 0.7], threshold=None, name="hard_focus"),
]
```

---

```python
class CurriculumScheduler:
    def __init__(
        self,
        phases: list[CurriculumPhase] = DEFAULT_PHASES,
        adaptive: bool = True,
        fixed_phase_epochs: int = 75,  # used only when adaptive=False
        competence_window: int = 5,
    ) -> None
```
- `adaptive=True`: advance based on competence thresholds.
- `adaptive=False`: advance every `fixed_phase_epochs` regardless of
  performance.
- `self.current_phase_idx` starts at 0.
- `self.phase_changed` flag set to `True` on the epoch phase advances;
  cleared at the start of the next `step()` call.
- `self.phase_history` list of `(epoch, phase_name)` tuples for logging.

---

```python
def step(self, metric_value: float, epoch: int) -> None
```
- Clears `self.phase_changed`.
- Calls `self.competence_meter.update(metric_value)`.
- If `adaptive`: calls `should_advance()` → `advance_phase(epoch)` if True.
- If not `adaptive`: advances when `epoch % fixed_phase_epochs == 0`.

---

```python
def should_advance(self) -> bool
```
- Returns `False` if already at the final phase.
- Returns `True` if `competence_meter.is_threshold_reached(current_threshold)`.

---

```python
def advance_phase(self, epoch: int) -> None
```
- Increments `current_phase_idx`.
- Sets `phase_changed = True`.
- Appends `(epoch, new_phase.name)` to `phase_history`.
- Calls `competence_meter.reset()`.
- Prints: `"[Epoch {epoch}] Advancing to Phase {idx}: {name}"`.

---

```python
def get_current_difficulty_ratios(self) -> list[float]
```
- Returns `phases[current_phase_idx].difficulty_ratios`.

---

```python
def get_phase_summary(self) -> Dict
```
- Returns `{"current_phase": int, "phase_name": str,
  "phase_history": list[...], "competence": float}`.
- Used for JSON checkpoint logging.

---

## Task 4.3 — Curriculum Training Loop

### File: `experiments/train_curriculum.py`

#### Entry point

```
python experiments/train_curriculum.py \
    --dataset cora \
    --model gcn \
    --heuristic cn \
    --seed 0 \
    --adaptive \
    --save_dir results/curriculum/
```

#### Core training function

```python
def train_with_curriculum(
    model: LinkPredictor,
    data_dict: Dict,
    optimizer: torch.optim.Optimizer,
    scheduler: CurriculumScheduler,
    sampler: DifficultyBasedSampler,
    device: torch.device,
    epochs: int = 300,
    eval_every: int = 10,
    logger: ExperimentLogger | None = None,
) -> Dict[str, float]
```

Loop structure:
```
for epoch in range(epochs):
    # 1. Get current difficulty mix from scheduler
    ratios = scheduler.get_current_difficulty_ratios()

    # 2. Sample negatives using current mix
    neg_edges = sampler.sample_mixed(num_neg_samples, ratios, seed=epoch)

    # 3. Standard training step (same as baseline but with pre-sampled neg)
    model.train()
    loss = train_epoch_with_negatives(model, data_dict, optimizer, neg_edges, device)

    # 4. Periodic evaluation
    if epoch % eval_every == 0:
        model.eval()
        val_metrics = evaluate(model, data_dict, "val", device)

        # 5. Update curriculum scheduler
        scheduler.step(val_metrics["auc"], epoch)

        # 6. Log everything
        if logger:
            logger.log_epoch(epoch, loss, val_metrics,
                             extra={"phase": scheduler.current_phase_idx,
                                    "phase_changed": scheduler.phase_changed})

        # 7. Log phase transition if it happened
        if scheduler.phase_changed:
            print(f"Epoch {epoch}: transitioned to phase {scheduler.current_phase_idx}")

    # 8. Save best checkpoint
    checkpoint_manager.save(model, epoch, val_metrics.get("auc", 0.0))

# Final test evaluation
model.eval()
test_metrics = evaluate(model, data_dict, "test", device)
return test_metrics
```

#### Modified `train_epoch_with_negatives`

```python
def train_epoch_with_negatives(
    model: LinkPredictor,
    data_dict: Dict,
    optimizer: torch.optim.Optimizer,
    neg_edge_index: Tensor,
    device: torch.device,
) -> float
```
- Takes pre-sampled negatives instead of sampling inside.
- Otherwise identical to `train_epoch` from Phase 2.

---

## Task 4.4 — Curriculum Config

### File: `configs/curriculum_config.yaml`

```yaml
dataset: cora
model: gcn
heuristic: cn
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
adaptive: true
fixed_phase_epochs: 75
competence_window: 5
difficulty_thresholds: [0, 2, 5]
phases:
  - ratios: [1.0, 0.0, 0.0]
    threshold: 0.75
    name: easy_only
  - ratios: [0.7, 0.3, 0.0]
    threshold: 0.85
    name: easy_medium
  - ratios: [0.3, 0.4, 0.3]
    threshold: 0.90
    name: mixed
  - ratios: [0.0, 0.3, 0.7]
    threshold: null
    name: hard_focus
save_dir: results/curriculum
checkpoint_dir: checkpoints
log_dir: logs
precomputed_dir: data/precomputed
```

---

## Verification Checklist

### Unit tests for Competence Meter

```python
meter = CompetenceMeter(window_size=3)
meter.update(0.80)
meter.update(0.82)
meter.update(0.84)
assert abs(meter.get_competence() - 0.82) < 1e-6
assert meter.is_threshold_reached(0.80)
assert not meter.is_threshold_reached(0.85)
```

### Unit tests for Scheduler

```python
sched = CurriculumScheduler(adaptive=True)
assert sched.current_phase_idx == 0
assert sched.get_current_difficulty_ratios() == [1.0, 0.0, 0.0]

# Simulate reaching phase 0 threshold
for _ in range(5):
    sched.step(0.76, epoch=_)  # above 0.75

assert sched.current_phase_idx == 1
assert sched.phase_changed  # True on the transition epoch
```

### Integration test

Run curriculum training for 50 epochs on Cora and confirm:
- Phase history JSON is non-empty.
- Training loss is finite throughout.
- At least Phase 0 is entered; Phase 1 may or may not be reached in 50
  epochs depending on model speed.

### Full curriculum run verification

Run 10 seeds on Cora with GCN + CN heuristic:
- All 4 phases should be visited in most seeds.
- Average competence at each transition should match configured thresholds.
- Results saved to `results/curriculum/cora_gcn_cn_seed{0..9}.json`.

---

## Ablation Modes

The curriculum scheduler supports these modes, all controlled by config:

| Mode            | adaptive | fixed_phase_epochs | Notes                                  |
|-----------------|----------|--------------------|----------------------------------------|
| Adaptive (main) | true     | —                  | Phase advances on competence threshold |
| Fixed schedule  | false    | 75                 | Phase advances every 75 epochs         |
| No curriculum   | false    | 999999             | Stays in Phase 0 forever (easy only)   |
| Hard from start | manually set initial phase to 3 | Starts with hard negatives only |

To test "no curriculum" baseline: set initial phase to Phase 3 directly in
the script, bypassing the scheduler entirely.

---

## Debugging Tips

- **Phase never advances**: most likely the competence smoothing window is
  too long or `eval_every` is too large. Temporarily lower the threshold to
  0.5 in the config to verify the mechanism works.
- **Training degrades after phase transition**: a sudden jump to hard
  negatives destabilises the model. Confirm that Phase 2 mixes hard and easy
  (30/40/30), not hard-only. If still degrading, increase competence window
  or lower hard-negative fraction.
- **Phase advances too quickly**: the window is too short or the threshold is
  too low. Increase `competence_window` to 10 or raise thresholds.
- **Sampler returns too few hard negatives**: the hard bucket may be
  undersized. Check `get_bucket_sizes()` and either lower the hard threshold
  or generate more candidate negatives during pre-computation.
- **`phase_changed` flag not cleared**: ensure `scheduler.step()` clears
  `phase_changed` at the start of each call.

---

## Estimated Time

| Task                          | Hours  |
|-------------------------------|--------|
| CompetenceMeter               | 2–3    |
| CurriculumScheduler           | 4–5    |
| Curriculum training loop      | 5–7    |
| Config + CLI wiring           | 2–3    |
| Unit tests                    | 3–4    |
| Debugging and full runs       | 5–7    |
| **Total**                     | **21–29** |
