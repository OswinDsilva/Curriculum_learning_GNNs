# First Draft Paper (Simplified Version)

## Working Title

Curriculum Learning with Difficulty-Aware Negative Sampling for GNN Link Prediction

## Abstract

This project studies link prediction in graphs using curriculum learning over difficulty-aware negative sampling. Instead of training only with random negatives, we gradually introduce medium and hard negatives during training. The implemented pipeline includes baseline models (GCN, GAT), structural difficulty scoring (CN, AA, RA), an adaptive curriculum scheduler, and HeaRT evaluation. We now report full 10-seed experiments across Cora, Citeseer, and PubMed, plus ablations on Cora + GCN. Final results are mixed: curriculum settings can help on some PubMed configurations, but baseline remains stronger in several Cora and Citeseer settings under the current presets.

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
6. Full 10-seed outputs for baseline, curriculum, and ablation conditions.

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

## 4. Experimental Setup

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

Completed run matrix:

1. Baseline: 3 datasets x 2 models x 10 seeds = 60 runs.
2. Curriculum: 3 datasets x 2 models x 3 heuristics x 10 seeds = 180 runs.
3. Ablations (Cora + GCN): 8 conditions x 10 seeds = 80 runs.

## 5. Results (Full Study)

Data source files:

1. results/summaries/baseline_summary.csv
2. results/summaries/curriculum_summary.csv
3. results/summaries/ablation_summary.csv
4. results/summaries/full_significance_table.csv

### 5.1 Baseline vs Best Curriculum (HeaRT MRR, 10-seed means)

1. Cora + GCN: baseline 0.5055 vs best curriculum 0.4124 (AA), -18.43%.
2. Cora + GAT: baseline 0.5226 vs best curriculum 0.5140 (RA), -1.65%.
3. Citeseer + GCN: baseline 0.4761 vs best curriculum 0.4593 (CN), -3.53%.
4. Citeseer + GAT: baseline 0.5781 vs best curriculum 0.5435 (RA), -6.00%.
5. PubMed + GCN: baseline 0.5420 vs best curriculum 0.5480 (AA), +1.12%.
6. PubMed + GAT: baseline 0.4752 vs best curriculum 0.4769 (RA), +0.37%.

### 5.2 Significance Overview

From `full_significance_table.csv`:

1. Total rows: 69.
2. HeaRT MRR rows: 23.
3. Significant HeaRT MRR rows (p < 0.05): 4.

### 5.3 Ablation Ranking (Cora + GCN, HeaRT MRR)

1. abl-1: 0.5051
2. abl-10: 0.3880
3. abl-6: 0.3816
4. abl-5: 0.3811
5. abl-3: 0.3664
6. abl-11: 0.3634
7. abl-2: 0.3498
8. abl-4: 0.3419

### 5.4 Quick Interpretation

1. Curriculum does not provide a universal HeaRT MRR improvement under current presets.
2. Gains appear on PubMed; Cora and Citeseer mainly favor baseline.
3. Ablation results indicate baseline-like settings remain strongest on Cora + GCN in this run budget.

## 6. What This Means Right Now

The full-run evidence indicates a dataset-dependent tradeoff:

1. Curriculum can match or slightly exceed baseline on some PubMed settings.
2. Several Cora/Citeseer settings degrade under the tested curriculum schedules.
3. Threshold and phase design are critical for stable gains.

These are final conclusions for the current experimental design and preset space.

## 7. Current Status and Limits

What is done:

1. End-to-end baseline and curriculum pipeline is implemented.
2. Full baseline/curriculum/ablation matrices are completed and aggregated.
3. Statistical tables and figures are generated from full runs.
4. Draft reporting document is updated to reflect full-study outcomes.

What is not done yet:

1. Broader hyperparameter search for curriculum thresholds and phase ratios.
2. Additional model families beyond GCN/GAT.
3. External dataset validation for stronger generalization claims.

## 8. Immediate Next Steps

1. Convert this draft into a final report with polished narrative and figure references.
2. Expand statistical discussion using confidence intervals and effect sizes.
3. Investigate why PubMed benefits while Cora/Citeseer degrade under current presets.
4. Evaluate tuned or dataset-specific curriculum schedules.
5. Package reproducibility artifacts for submission (commands, hashes, environment snapshot).

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
