# Phase 7 — Analysis & Documentation (Weeks 13–14)

## Goal

Interpret and communicate the experimental results through statistical
analysis, visualisations, and written documentation. This is the phase
that turns a working implementation into a finished research project.

---

## Exit Criteria

- [ ] All visualisation plots generated and saved to `results/figures/`.
- [ ] Statistical analysis complete: t-tests, effect sizes, confidence
      intervals computed for all main comparisons.
- [ ] Jupyter notebook runs from top to bottom without errors.
- [ ] README updated with results summary and usage instructions.
- [ ] All functions have docstrings (Python-style, one-liner minimum).
- [ ] `ruff format` and `ruff check` pass on the entire codebase.
- [ ] Technical report outline written.

---

## Task 7.1 — Statistical Analysis

Perform all significance testing in `experiments/statistical_analysis.py`
(or inside the notebook).

### Tests to run

**Primary comparison: curriculum vs baseline**

For each `(dataset, model, heuristic)` combination:
- Metric: HeaRT MRR (primary), plus Hits@10 and Hits@50.
- Test: paired t-test across 10 seeds (`scipy.stats.ttest_rel`).
- Report: t-statistic, p-value, whether significant at p < 0.05.

**Effect size**

```python
def cohens_d(a: list[float], b: list[float]) -> float:
    diff = np.mean(a) - np.mean(b)
    pooled_std = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    return diff / pooled_std
```

Interpret: |d| < 0.2 = negligible, 0.2–0.5 = small, 0.5–0.8 = medium,
> 0.8 = large.

**Confidence intervals**

```python
from scipy.stats import t as t_dist

def confidence_interval_95(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mean = np.mean(values)
    se = np.std(values, ddof=1) / np.sqrt(n)
    margin = t_dist.ppf(0.975, df=n-1) * se
    return mean - margin, mean + margin
```

**Ablation comparison**

Run paired t-tests between:
- Main curriculum (adaptive) vs fixed-schedule curriculum.
- Main curriculum vs hard-from-start (ablation abl-2).
- Main curriculum vs no-curriculum (ablation abl-1 = baseline).

### Output file: `results/summaries/full_significance_table.csv`

```
dataset, model, heuristic, metric, baseline_mean, baseline_std,
curriculum_mean, curriculum_std, improvement_pct, t_stat, p_value,
cohens_d, significant
```

---

## Task 7.2 — Visualisations

All plots saved to `results/figures/` as `.pdf` (publication quality) and
`.png` (for quick preview).

### Plot 1 — Learning curves

**File**: `results/figures/learning_curves_{dataset}_{model}.pdf`

- X-axis: epoch.
- Y-axis: val AUC.
- Two lines: baseline (random neg) and curriculum (best heuristic).
- Vertical dashed lines at phase transition epochs (median across seeds).
- Shaded bands for ± 1 std across 10 seeds.

```python
def plot_learning_curves(
    baseline_epoch_data: Dict,     # {seed: {epoch: val_auc}}
    curriculum_epoch_data: Dict,
    phase_transitions: list[int],  # median epoch per phase transition
    title: str,
    save_path: str,
) -> None
```

### Plot 2 — Performance comparison bar chart

**File**: `results/figures/performance_comparison_{dataset}.pdf`

- Grouped bar chart: one group per model (GCN, GAT).
- Within each group: baseline, curriculum-CN, curriculum-AA, curriculum-RA.
- Show HeaRT MRR on Y-axis with error bars (95 % CI).
- Annotate bars with asterisks for significant results (` *` p<0.05,
  `**` p<0.01, `***` p<0.001).

### Plot 3 — HeaRT gap analysis

**File**: `results/figures/heart_gap_analysis.pdf`

- Scatter plot: X = standard MRR, Y = HeaRT MRR.
- One point per (dataset, model, condition) combination.
- Color by dataset, marker shape by model.
- Add diagonal reference line `y=x`.
- Points below the diagonal are expected (HeaRT < standard).

### Plot 4 — Heuristic comparison

**File**: `results/figures/heuristic_comparison.pdf`

- Line plot across datasets (X-axis) for HeaRT MRR (Y-axis).
- One line per heuristic: CN, AA, RA.
- Split into subplots by model.
- Shows which heuristic performs best overall.

### Plot 5 — Phase transition analysis

**File**: `results/figures/phase_transitions.pdf`

- For each (dataset, model, heuristic): distribution of transition epochs for
  each phase (box plot or violin).
- X-axis: phase name (easy_medium, mixed, hard_focus).
- Y-axis: epoch at which transition occurred.
- Shows how quickly the model progresses through the curriculum and whether
  the schedule is well-calibrated.

### Plot 6 — Competence progression

**File**: `results/figures/competence_progression.pdf`

- Val AUC over epochs, annotated with phase transition points.
- One subplot per dataset.
- Shows whether model competence is growing smoothly or stalling before
  transitions.

### Plot 7 — Ablation study results

**File**: `results/figures/ablation_study.pdf`

- Horizontal bar chart: one bar per ablation condition.
- Metric: HeaRT MRR on Cora, GCN.
- Sorted by performance.
- Colour the main curriculum bar distinctly.

---

## Task 7.3 — Jupyter Notebook

### File: `notebooks/results_analysis.ipynb`

Structure:

**Cell 1 — Setup**: install/import libraries, set paths.

**Cell 2 — Load results**: read all JSON files, build a flat `pandas.DataFrame`
with one row per run.

**Cell 3 — Summary tables**: pivot tables showing mean ± std for all
(dataset, model, condition) combinations. Display both standard and HeaRT
metrics.

**Cell 4 — Statistical tests**: run paired t-tests and display a significance
table as a styled DataFrame.

**Cell 5 — Learning curves**: generate Plot 1.

**Cell 6 — Performance comparison**: generate Plot 2.

**Cell 7 — HeaRT gap**: generate Plot 3.

**Cell 8 — Heuristic comparison**: generate Plot 4.

**Cell 9 — Phase transitions**: generate Plot 5.

**Cell 10 — Ablation study**: generate Plot 7.

**Cell 11 — Interpretation**: Markdown cell discussing findings, relating
them to the research question.

---

## Task 7.4 — README Update

Update `README.md` to include:

1. **Installation** (already present, verify still correct).
2. **Quickstart** — one command to run a single experiment.
3. **Project structure** — brief tour of directories.
4. **Reproducing results** — reference to `scripts/run_all_experiments.sh`.
5. **Results summary table** — paste the main performance comparison from
   the notebook.
6. **Citation** — BibTeX for HeaRT paper.
7. **Negative result note** — brief note that negative results are expected
   and documented.

---

## Task 7.5 — Code Quality Sweep

### Docstrings

Add a one-line docstring to every public function and class that does not
already have one. Priority order:
1. All functions in `utils/`, `models/`, `curriculum/`, `negative_sampling/`
2. Training scripts
3. Test files (docstrings optional here)

### Type hints

Every function in the main codebase should have typed arguments and return
annotations. Functions that interact with PyG `Data` or `Tensor` objects
are most important.

### Linting

```bash
# Format
.venv/bin/ruff format gnn_curriculum_learning/

# Style check
.venv/bin/ruff check gnn_curriculum_learning/ \
  --line-length 100 \
  --exclude .venv,data,results,checkpoints,logs

# Optional auto-fix
.venv/bin/ruff check gnn_curriculum_learning/ \
  --line-length 100 \
  --exclude .venv,data,results,checkpoints,logs \
  --fix
```

Fix all reported style and lint violations.

---

## Task 7.6 — Technical Report Outline

This is not a full paper but a structured write-up for the project.
Suggested sections:

1. **Introduction** — problem statement, motivation, HeaRT gap.
2. **Related Work** — GNN link prediction, curriculum learning, hard negative
   mining.
3. **Method** — curriculum phases, competence meter, heuristics, HeaRT
   evaluation.
4. **Experiments** — datasets, models, hyperparameters, evaluation protocol.
5. **Results** — main comparison table, learning curves, HeaRT gap analysis.
6. **Ablation Study** — contribution of each curriculum component.
7. **Discussion** — when does curriculum help, when does it fail, why.
8. **Conclusion** — summary, limitations, future work.

---

## Checklist for Phase 7 Completion

- [ ] `results/figures/` contains all 7 plots as both PDF and PNG.
- [ ] Notebook runs clean from top to bottom.
- [ ] Significance table written to CSV.
- [ ] README updated with results table.
- [ ] `ruff format --check` passes with no output.
- [ ] `ruff check` passes with zero errors.
- [ ] Technical report outline exists in `docs/report_outline.md`.

---

## Final Results (4 April 2026)

Phase 7 execution is complete with full experiment coverage and regenerated
analysis artifacts.

### Completed run matrix

- Baseline: 60 runs (3 datasets x 2 models x 10 seeds)
- Curriculum: 180 runs (3 datasets x 2 models x 3 heuristics x 10 seeds)
- Ablation: 80 runs (8 conditions x 10 seeds, Cora + GCN)

### Main HeaRT MRR outcome (baseline vs best curriculum per dataset/model)

- Cora + GCN: 0.5055 -> 0.4124 (AA), -18.43%
- Cora + GAT: 0.5226 -> 0.5140 (RA), -1.65%
- Citeseer + GCN: 0.4761 -> 0.4593 (CN), -3.53%
- Citeseer + GAT: 0.5781 -> 0.5435 (RA), -6.00%
- PubMed + GCN: 0.5420 -> 0.5480 (AA), +1.12%
- PubMed + GAT: 0.4752 -> 0.4769 (RA), +0.37%

### Statistical summary

- `results/summaries/full_significance_table.csv`: 69 rows total
- HeaRT MRR rows: 23
- Significant HeaRT MRR rows (p < 0.05): 4

### Ablation highlight (HeaRT MRR, Cora + GCN)

- Best: `abl-1` = 0.5051
- Lowest: `abl-4` = 0.3419

### Final artifacts

- Summaries: `results/summaries/`
- Significance table: `results/summaries/full_significance_table.csv`
- Figures: `results/figures/`
- Updated report draft: `docs/paper_first_draft.md`

---

## Estimated Time

| Task                          | Hours  |
|-------------------------------|--------|
| Statistical analysis          | 4–6    |
| Learning curve plots          | 3–4    |
| Bar charts and comparisons    | 3–4    |
| HeaRT gap and heuristic plots | 2–3    |
| Phase transition plots        | 2–3    |
| Ablation plot                 | 1–2    |
| Jupyter notebook assembly     | 4–6    |
| README update                 | 2–3    |
| Docstrings and type hints     | 4–6    |
| Linting                       | 1–2    |
| Report outline                | 3–4    |
| **Total**                     | **29–43** |
