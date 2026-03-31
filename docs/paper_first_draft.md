# First Draft Paper (Simplified Version)

## Working Title

Curriculum Learning with Difficulty-Aware Negative Sampling for GNN Link Prediction

## Abstract

This project studies link prediction in graphs. The main idea is simple: instead of training only with easy negative examples, we gradually introduce harder negatives during training. We built baseline models, a difficulty scoring pipeline, a curriculum scheduler, and hard-negative evaluation support. Current results are early and come from smoke tests, not full experiments. On the available Cora + GCN smoke run, curriculum slightly improves HeaRT MRR but slightly reduces some standard metrics. This is a useful early signal, but final conclusions need full multi-seed runs and significance testing.

## 1. Introduction

Link prediction means guessing which node pairs should have an edge but currently do not. In many pipelines, negative samples are random pairs, and those are often too easy. That can make model performance look better than it is.

This project tries to fix that gap using curriculum learning. We start training with easy negatives, then slowly add medium and hard negatives. The goal is to make the model more reliable when evaluated with difficult candidates.

Main question:

Does a difficulty-based curriculum improve hard-negative evaluation performance for GNN link prediction?

## 2. What Has Been Built

1. Baseline training scripts for GCN and GAT.
2. Heuristic-based difficulty scoring using CN, AA, and RA.
3. A sampler that can draw easy, medium, and hard negatives.
4. A curriculum scheduler with adaptive phase transitions.
5. HeaRT-style evaluation integrated into training outputs.
6. Smoke-test outputs for baseline and curriculum.

## 3. Method (Simple Explanation)

### 3.1 Baseline Training

Baseline training uses random negatives at each epoch. It logs standard metrics and saves the best checkpoint.

### 3.2 Difficulty Scoring

Each candidate non-edge gets a structural score using one heuristic:

1. CN: count shared neighbors.
2. AA: shared neighbors with stronger weight for low-degree neighbors.
3. RA: similar to AA but with inverse degree weighting.

These scores are precomputed to avoid expensive calculations during training.

### 3.3 Difficulty Buckets

Candidates are split into:

1. Easy
2. Medium
3. Hard

The sampler can draw only one bucket or a custom mix.

### 3.4 Curriculum Schedule

Training follows 4 stages:

1. easy_only: 1.0 / 0.0 / 0.0
2. easy_medium: 0.7 / 0.3 / 0.0
3. mixed: 0.3 / 0.4 / 0.3
4. hard_focus: 0.0 / 0.3 / 0.7

In adaptive mode, transition happens when validation performance reaches thresholds.

### 3.5 Evaluation

We report both:

1. Standard metrics (AUC, AP, MRR, Hits@K)
2. HeaRT metrics (hard-negative ranking metrics)

This helps show whether performance is truly robust or only good on easy negatives.

## 4. Current Experimental Setup

Planned datasets:

1. Cora
2. Citeseer
3. PubMed

Planned models:

1. GCN
2. GAT

Planned conditions:

1. Baseline
2. Curriculum (CN, AA, RA)
3. Ablations

Current completed runs are smoke tests on Cora + GCN + seed 0 for both baseline and curriculum.

## 5. Results So Far (Smoke Tests)

Data source files:

1. results/baseline_smoke/cora_gcn_seed0.json
2. results/curriculum_smoke/cora_gcn_common_neighbors_seed0.json

### 5.1 Baseline (Cora, GCN, seed 0)

Standard:

1. AUC: 0.9138
2. AP: 0.9274
3. MRR: 0.3391
4. Hits@10: 0.0190
5. Hits@50: 0.0949
6. Hits@100: 0.1898

HeaRT:

1. HeaRT MRR: 0.4324
2. HeaRT Hits@10: 0.7875
3. HeaRT Hits@50: 0.9526
4. HeaRT Hits@100: 0.9981

### 5.2 Curriculum CN (Cora, GCN, seed 0)

Standard:

1. AUC: 0.9036
2. AP: 0.9180
3. MRR: 0.2579
4. Hits@10: 0.0190
5. Hits@50: 0.0949
6. Hits@100: 0.1879

HeaRT:

1. HeaRT MRR: 0.4564
2. HeaRT Hits@10: 0.7856
3. HeaRT Hits@50: 0.9488
4. HeaRT Hits@100: 1.0000

Phase progression observed:

1. Epoch 0: easy_only
2. Epoch 10: easy_medium
3. Epoch 30: mixed
4. Epoch 90: hard_focus

### 5.3 Quick Interpretation

1. HeaRT MRR improved from 0.4324 to 0.4564 (about +5.5% relative).
2. Some standard metrics became lower in this run.
3. Result is useful but not final because this is only one seed.

## 6. What This Means Right Now

The current evidence suggests a possible tradeoff:

1. Better performance on hard-negative ranking.
2. Slightly weaker performance on standard easy-negative metrics.

This is not a final claim yet. Full multi-seed and multi-dataset results are required.

## 7. Current Status and Limits

What is done:

1. End-to-end baseline and curriculum pipeline is implemented.
2. Smoke tests are completed and stored.
3. Draft reporting document is prepared.

What is not done yet:

1. Full 10-seed matrix for all settings.
2. Complete ablation table.
3. Final significance analysis and confidence intervals.

## 8. Immediate Next Steps

1. Run full baseline and curriculum sweeps.
2. Run ablations.
3. Aggregate all result files.
4. Run significance tests.
5. Update paper tables and final conclusions.

## 9. Term Guide (Meaning and Significance)

This section explains each technical term used in this draft and why it matters.

1. Graph: A structure of nodes and edges. Significance: this is the data format of the whole project.
2. Node: One entity in the graph. Significance: model embeddings are learned per node.
3. Edge: A connection between two nodes. Significance: link prediction tries to predict missing edges.
4. Link Prediction: Predicting whether an edge should exist. Significance: this is the core task.
5. GNN (Graph Neural Network): A model that learns from graph structure. Significance: main model family used.
6. GCN: A specific GNN architecture based on graph convolution. Significance: one baseline model.
7. GAT: A GNN architecture with attention on neighbors. Significance: second baseline model.
8. Baseline: The reference training setup without curriculum. Significance: required comparison point.
9. Negative Sample: A node pair treated as not connected. Significance: needed to train binary edge classifiers.
10. Random Negative Sampling: Sampling negatives randomly. Significance: standard method, often too easy.
11. Hard Negative: A non-edge that looks structurally similar to a true edge. Significance: more realistic and challenging.
12. Curriculum Learning: Training from easier examples to harder ones. Significance: core idea being tested.
13. Difficulty-Aware Sampling: Sampling negatives by difficulty level. Significance: how curriculum is implemented.
14. Heuristic: A graph-based scoring rule. Significance: used to estimate difficulty.
15. CN (Common Neighbors): Number of shared neighbors between two nodes. Significance: simple and strong hardness signal.
16. AA (Adamic-Adar): Shared-neighbor score weighted by inverse log degree. Significance: emphasizes informative rare neighbors.
17. RA (Resource Allocation): Shared-neighbor score weighted by inverse degree. Significance: another hardness scoring alternative.
18. Candidate Pool: Stored list of candidate non-edges with scores. Significance: enables fast repeated sampling.
19. Precompute: Calculate scores before training starts. Significance: reduces runtime cost during epochs.
20. Difficulty Buckets: Easy, medium, hard partitions. Significance: allows controlled sampling ratios.
21. Sampler: Component that returns negatives by bucket ratio. Significance: connects difficulty scores to training batches.
22. Scheduler: Component that decides when to change phases. Significance: controls curriculum progression.
23. Adaptive Mode: Phase transition depends on validation performance. Significance: data-driven progression.
24. Fixed Mode: Phase transition depends only on epoch count. Significance: useful ablation and fallback.
25. Competence Meter: Moving estimate of model readiness. Significance: prevents switching phases too early.
26. Phase History: Log of when phase transitions happened. Significance: helps analyze training dynamics.
27. Epoch: One full pass of training updates. Significance: basic time unit of training.
28. Validation Split: Data used for model selection during training. Significance: drives early decisions and scheduler updates.
29. Test Split: Data used only for final performance reporting. Significance: reduces overfitting in claims.
30. Checkpoint: Saved model state. Significance: allows best-model recovery and reproducibility.
31. AUC: Area under ROC curve. Significance: ranking quality across thresholds.
32. AP (Average Precision): Precision-recall summary area. Significance: useful under class imbalance.
33. MRR: Mean reciprocal rank. Significance: captures where true edges appear in ranked lists.
34. Hits@K: Fraction of true edges appearing in top K ranks. Significance: practical ranking quality metric.
35. HeaRT: Hard-negative evaluation protocol for link prediction. Significance: tests robustness under difficult candidates.
36. HeaRT MRR and HeaRT Hits@K: HeaRT-specific ranking metrics. Significance: primary robustness measures in this project.
37. Smoke Test: Small quick run to verify pipeline works. Significance: current results are from this stage.
38. Seed: Random-state control value. Significance: required for reproducibility and fair comparison.
39. Ablation: Controlled variant where one design part is changed. Significance: shows which component actually helps.
40. Statistical Significance Test: Test to check if differences are likely real. Significance: needed before making strong claims.
41. Confidence Interval: Range of plausible true metric values. Significance: shows uncertainty around averages.
42. Reproducibility: Ability to get same results again with same setup. Significance: essential for credible research.

## 10. Reproducibility Notes

Key scripts:

1. experiments/train_baseline.py
2. experiments/train_curriculum.py
3. experiments/aggregate_results.py
4. experiments/statistical_analysis.py
5. scripts/run_all_experiments.sh

This draft is intentionally simple and in-progress, designed for milestone submission with completed work plus partial results.
