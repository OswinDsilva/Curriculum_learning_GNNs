# Phase 6 — Comprehensive Experiments (Weeks 10–12)

## Goal

Execute the full experimental protocol across all datasets, models,
heuristics, and seeds. Produce comparable, statistically tested results for
the baseline and all curriculum variants. Collect enough data to draw
conclusions.

---

## Exit Criteria

- [ ] All baseline experiments complete: 3 datasets × 2 models × 10 seeds
      = 60 runs.
- [ ] All curriculum experiments complete: 3 datasets × 2 models × 3
      heuristics × 10 seeds = 180 runs.
- [ ] All ablation experiments complete (see Task 6.2).
- [ ] Results are reproducible: re-running any single seed gives identical
      JSON output.
- [ ] Variance across seeds is reasonable: std < 5 % of mean for AUC.
- [ ] Aggregated summary tables written to `results/`.
- [ ] Statistical significance tests run and p-values logged.

---

## Task 6.1 — Full Experimental Protocol

### Baseline experiments

Datasets: Cora, Citeseer, PubMed  
Models: GCN, GAT  
Seeds: 0–9  
Negatives: random

```bash
for dataset in cora citeseer pubmed; do
    for model in gcn gat; do
        for seed in $(seq 0 9); do
            .venv/bin/python experiments/train_baseline.py \
                --dataset $dataset \
                --model $model \
                --seed $seed \
                --heart \
                --save_dir results/baseline/
        done
    done
done
```

### Curriculum experiments

Datasets: Cora, Citeseer, PubMed  
Models: GCN, GAT  
Heuristics: CN, AA, RA  
Seeds: 0–9  
Mode: adaptive

```bash
for dataset in cora citeseer pubmed; do
    for model in gcn gat; do
        for heuristic in cn aa ra; do
            for seed in $(seq 0 9); do
                .venv/bin/python experiments/train_curriculum.py \
                    --dataset $dataset \
                    --model $model \
                    --heuristic $heuristic \
                    --seed $seed \
                    --adaptive \
                    --heart \
                    --save_dir results/curriculum/
            done
        done
    done
done
```

### Result file naming convention

```
results/baseline/{dataset}_{model}_seed{seed}.json
results/curriculum/{dataset}_{model}_{heuristic}_seed{seed}.json
results/ablation/{ablation_name}_{dataset}_{model}_seed{seed}.json
```

### JSON result file schema

```json
{
  "config": {
    "dataset": "cora",
    "model": "gcn",
    "heuristic": "cn",
    "seed": 0,
    "epochs": 300,
    "hidden_channels": 128,
    "out_channels": 64,
    "lr": 0.01,
    "dropout": 0.5,
    "adaptive": true
  },
  "standard": {
    "auc": 0.8912,
    "ap": 0.8974,
    "mrr": 0.4231,
    "hits@10": 0.3401,
    "hits@50": 0.5612,
    "hits@100": 0.6988
  },
  "heart": {
    "heart_mrr": 0.2811,
    "heart_hits@10": 0.2012,
    "heart_hits@50": 0.3890,
    "heart_hits@100": 0.5011
  },
  "phase_history": [
    [42, "easy_medium"],
    [119, "mixed"],
    [201, "hard_focus"]
  ],
  "training_time_seconds": 124.3
}
```

---

## Task 6.2 — Ablation Studies

Run these additional conditions to isolate the contribution of each
curriculum component. Each ablation should be run for 10 seeds on Cora
with GCN.

| Ablation ID | Description                              | Config change                                      |
|-------------|------------------------------------------|----------------------------------------------------|
| abl-1       | No curriculum; random negatives only     | Standard baseline (already done)                   |
| abl-2       | Hard negatives from epoch 0              | Set initial phase to Phase 3; skip scheduler       |
| abl-3       | Fixed 50/50 easy-hard split; no schedule | `difficulty_ratios=[0.5, 0.0, 0.5]`; no advance   |
| abl-4       | Fixed-epoch schedule (non-adaptive)      | `adaptive=false`, `fixed_phase_epochs=75`          |
| abl-5       | 3-phase curriculum (no Phase 2 mixed)    | Skip Phase 2; go Phase 0 → Phase 1 → Phase 3      |
| abl-6       | 5-phase curriculum (extra intermediate)  | Add Phase between current Phase 1 and Phase 2      |
| abl-7       | CN heuristic only                        | Already run in main curriculum experiments         |
| abl-8       | AA heuristic only                        | Already run in main curriculum experiments         |
| abl-9       | RA heuristic only                        | Already run in main curriculum experiments         |
| abl-10      | Lower competence thresholds (0.65/0.75/0.82) | Modify phases config                          |
| abl-11      | Higher competence thresholds (0.80/0.90/0.95)| Modify phases config                          |

Save all ablation results to `results/ablation/` following the same JSON
schema.

---

## Task 6.3 — Results Aggregation

### File: `experiments/aggregate_results.py`

```
python experiments/aggregate_results.py \
    --results_dir results/ \
    --output_dir results/summaries/
```

What it does:
1. Recursively walk `results_dir` for all `.json` files.
2. Group by `(dataset, model, heuristic_or_none, condition)`.
3. For each group, compute mean and std for every metric.
4. Write two outputs:
   - `results/summaries/baseline_summary.csv`
   - `results/summaries/curriculum_summary.csv`
   - `results/summaries/ablation_summary.csv`

### Summary CSV columns

```
dataset, model, heuristic, condition, metric, mean, std, n_seeds
```

### Statistical significance tests

After aggregation:
```python
# For each (dataset, model, heuristic) combination:
from scipy.stats import ttest_rel

baseline_scores = [load(f) for f in baseline_files]
curriculum_scores = [load(f) for f in curriculum_files]

# Paired t-test on HeaRT MRR across seeds
t_stat, p_value = ttest_rel(curriculum_scores, baseline_scores)
effect_size = cohens_d(curriculum_scores, baseline_scores)
```

Save to `results/summaries/significance_tests.csv` with columns:
```
dataset, model, heuristic, metric, t_stat, p_value, effect_size, significant
```

Where `significant = (p_value < 0.05)`.

---

## Task 6.4 — Compute Resource Planning

| Dataset  | Nodes  | Edges  | GCN (300 ep) | GAT (300 ep) |
|----------|--------|--------|--------------|--------------|
| Cora     | 2,708  | 10,556 | ~30 s        | ~60 s        |
| Citeseer | 3,327  | 9,104  | ~35 s        | ~70 s        |
| PubMed   | 19,717 | 88,648 | ~4 min       | ~8 min       |

Estimated total compute for all main experiments (60 baseline + 180
curriculum runs) on CPU:

- Baseline: 60 × 2 min avg = ~2 hours
- Curriculum: 180 × 3 min avg = ~9 hours
- Ablations: 110 × 2 min avg = ~4 hours
- Total: ~15 hours

Tips to speed up:
- Parallelize with `xargs -P 4` or `gnu parallel` for seed loops.
- Run PubMed experiments last (largest dataset).
- Cache NetworkX graphs and pre-computed scores (already planned).
- If GPU is available: all experiments drop to ~15 % of CPU time.

---

## Task 6.5 — Experiment Automation Script

### File: `scripts/run_all_experiments.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

PYTHON=".venv/bin/python"
SEEDS=$(seq 0 9)

# --- Baseline ---
for dataset in cora citeseer pubmed; do
    for model in gcn gat; do
        for seed in $SEEDS; do
            $PYTHON experiments/train_baseline.py \
                --dataset $dataset --model $model \
                --seed $seed --heart \
                --save_dir results/baseline/
        done
    done
done

# --- Curriculum ---
for dataset in cora citeseer pubmed; do
    for model in gcn gat; do
        for heuristic in cn aa ra; do
            for seed in $SEEDS; do
                $PYTHON experiments/train_curriculum.py \
                    --dataset $dataset --model $model \
                    --heuristic $heuristic --seed $seed \
                    --adaptive --heart \
                    --save_dir results/curriculum/
            done
        done
    done
done

# --- Aggregation ---
$PYTHON experiments/aggregate_results.py \
    --results_dir results/ \
    --output_dir results/summaries/

echo "All experiments complete."
```

---

## Verification Checklist

- [ ] Run a single baseline experiment end-to-end and verify the JSON file.
- [ ] Run a single curriculum experiment and verify phase transitions logged.
- [ ] Run `aggregate_results.py` on a subset (Cora, GCN only) and check CSV.
- [ ] Confirm that re-running any seed produces identical JSON.
- [ ] Confirm that variance across the 10-seed Cora/GCN baseline is < 1 %
      std for AUC.

---

## Debugging Tips

- **JSON files missing after run**: check that the `save_dir` directory
  exists and is writable. Create it with `mkdir -p results/baseline/`.
- **Variance > 5 %**: check whether `torch.manual_seed` and `numpy.random.seed`
  are both set at the start of each script; also seed the DataLoader if used.
- **Aggregation script crashes on missing keys**: add a fallback for runs
  where HeaRT was not computed (set HeaRT metrics to `None` and skip them
  in aggregation).
- **Parallel runs conflict on checkpoint files**: use a unique file name per
  `(dataset, model, heuristic, seed)` combination; never use a shared
  `best.pt` filename across experiments.
- **PubMed runs OOM**: reduce `hidden_channels` to 64 or use mini-batching
  with PyG's `DataLoader`.

---

## Estimated Time

| Task                          | Hours  |
|-------------------------------|--------|
| Automation script             | 2–3    |
| Running all baseline runs     | 3–4    |
| Running all curriculum runs   | 8–12   |
| Running ablations             | 6–8    |
| Aggregation script            | 4–5    |
| Significance tests            | 2–3    |
| Debugging failed runs         | 4–6    |
| **Total**                     | **29–41** |
