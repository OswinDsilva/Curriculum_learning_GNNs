# Project Overview

## Research Goal

Implement a curriculum learning framework that progressively trains GNN
models on negative samples of increasing difficulty for link prediction.

## Core Problem

Standard GNN link prediction training uses random negative sampling, which
produces unrealistically easy examples. Models trained this way generalise
poorly when evaluated against hard negatives — node pairs that are
structurally very similar to connected nodes but share no actual edge.

## Solution

A 4-phase curriculum that begins with easy random negatives and
progressively transitions to hard negatives selected by graph heuristics
(Common Neighbors, Adamic-Adar, Resource Allocation).

---

## Directory Layout

```
gnn_curriculum_learning/
├── data/                        # Auto-downloaded datasets (PyG)
├── models/                      # GNN encoder-decoder implementations
│   ├── __init__.py
│   ├── base.py                  # Abstract LinkPredictor, decoders
│   ├── gcn.py                   # GCN encoder-decoder
│   └── gat.py                   # GAT encoder-decoder
├── negative_sampling/           # Negative sampling strategies
│   ├── __init__.py
│   ├── heuristics.py            # CN, AA, RA, PPR implementations
│   ├── sampler.py               # Difficulty-based sampler
│   └── heart.py                 # HeaRT evaluation protocol
├── curriculum/                  # Curriculum learning framework
│   ├── __init__.py
│   ├── scheduler.py             # Curriculum phase scheduler
│   └── competence.py            # Competence tracking
├── utils/                       # Shared utilities
│   ├── __init__.py
│   ├── data_utils.py            # Dataset loading and edge splitting
│   ├── metrics.py               # AUC, AP, MRR, Hits@K
│   └── logging_utils.py         # TensorBoard, CSV, checkpoint management
├── experiments/                 # Runnable scripts
│   ├── data_exploration.py      # Dataset statistics
│   ├── train_baseline.py        # Baseline training (random negatives)
│   ├── train_curriculum.py      # Curriculum training
│   └── evaluate.py              # Model evaluation
├── configs/
│   ├── baseline_config.yaml
│   └── curriculum_config.yaml
├── notebooks/
│   └── results_analysis.ipynb
├── scripts/
│   └── verify_environment.py
├── tests/
│   ├── test_environment.py
│   ├── test_data_utils.py
│   ├── test_models.py
│   └── test_integration_smoke.py
├── results/                     # Per-run JSON/CSV result files
├── checkpoints/                 # Model checkpoint files
├── logs/                        # TensorBoard and text logs
├── docs/                        # This documentation
├── requirements.txt
└── README.md
```

---

## Datasets

| Name             | Type          | Loader                    | Notes                    |
|------------------|---------------|---------------------------|--------------------------|
| Cora             | Citation      | PyG Planetoid             | Smallest; start here     |
| Citeseer         | Citation      | PyG Planetoid             |                          |
| PubMed           | Citation      | PyG Planetoid             |                          |
| Coauthor-CS      | Co-authorship | PyG Coauthor              |                          |
| Coauthor-Physics | Co-authorship | PyG Coauthor              |                          |
| ogbl-collab      | Co-authorship | OGB (optional)            |                          |
| ogbl-ddi         | Drug interact | OGB (optional)            |                          |

---

## Models

| Name      | Reference                    | Phase added |
|-----------|------------------------------|-------------|
| GCN       | Kipf & Welling 2017          | 1           |
| GAT       | Veličković et al. 2018       | 1           |
| GraphSAGE | Hamilton et al. 2017         | optional    |

---

## Evaluation Metrics

| Metric     | Standard eval | HeaRT eval |
|------------|---------------|------------|
| AUC-ROC    | Yes           | No         |
| AP         | Yes           | No         |
| MRR        | Yes           | Yes        |
| Hits@10    | Yes           | Yes        |
| Hits@50    | Yes           | Yes        |
| Hits@100   | Yes           | Yes        |

---

## Phase Summary

| Phase | Title                     | Weeks  | Key Deliverable                        |
|-------|---------------------------|--------|----------------------------------------|
| 1     | Foundation                | 1–2    | Data pipeline + baseline model stubs   |
| 2     | Baseline Training         | 3–4    | Trained GCN/GAT with random negatives  |
| 3     | Heuristics & Difficulty   | 5–6    | Pre-computed scores + sampler          |
| 4     | Curriculum Framework      | 7–8    | Adaptive 4-phase curriculum            |
| 5     | HeaRT Evaluation          | 9      | Hard-negative evaluation protocol      |
| 6     | Comprehensive Experiments | 10–12  | Full results across datasets/models    |
| 7     | Analysis & Documentation  | 13–14  | Visualisations, stats, final report    |

---

## Success Criteria

### Minimum (pass)
- All baseline models match literature performance.
- Curriculum framework implemented and functional.
- Experiments run on at least 3 datasets.
- HeaRT evaluation implemented.
- Clear methodology documentation.

### Target (strong)
- Curriculum improves HeaRT MRR by 5–15 % over baseline.
- Statistical significance: p < 0.05 (paired t-test).
- Ablations isolate the contribution of each component.
- Code is clean, documented, reproducible.

### Excellent (publication-quality)
- Consistent improvement across all datasets and both models.
- Novel curriculum design insights documented.
- Extensive ablations and visualisations.
- Open-source release with tutorial notebook.

---

## Negative Result Policy

If curriculum does not improve performance, that is a valid research
contribution. In that case:
1. Analyse why (dataset density, model capacity, heuristic quality).
2. Document failure modes thoroughly.
3. Try alternative curriculum schedules.
4. Compare against hard-negative-only training as an alternative.
5. Write up the analysis of why curriculum failed as the main contribution.

---

## Key Reference

Chamberlain et al., "Evaluating Graph Neural Networks for Link Prediction:
Current Pitfalls and New Benchmarking," NeurIPS 2023.
Introduces HeaRT evaluation with hard negatives per positive edge.
