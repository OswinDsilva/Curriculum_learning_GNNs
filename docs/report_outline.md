# Technical Report Outline

## 1. Introduction

- Problem statement: link prediction with graph neural networks under weak negative sampling.
- Motivation: random negatives overestimate downstream ranking quality.
- Research question: can curriculum learning with adaptive hard-negative exposure improve HeaRT performance?

## 2. Related Work

- GNN-based link prediction.
- Curriculum learning for representation learning.
- Hard-negative mining and HeaRT evaluation.

## 3. Method

- Baseline GCN/GAT link prediction pipeline.
- Structural heuristic scoring: common neighbors, Adamic-Adar, resource allocation.
- Difficulty-based sampler.
- Competence meter and curriculum scheduler.
- HeaRT evaluator and per-positive ranking metrics.

## 4. Experimental Setup

- Datasets: Cora, Citeseer, PubMed.
- Models: GCN, GAT.
- Main curriculum settings and thresholds.
- Baseline, curriculum, and ablation conditions.
- Standard and HeaRT metrics.

## 5. Results

- Main comparison table for baseline vs curriculum.
- Learning curve analysis.
- Standard-vs-HeaRT gap analysis.
- Heuristic comparison by dataset and model.

## 6. Ablation Study

- Hard-from-start vs adaptive curriculum.
- Fixed schedule vs adaptive schedule.
- Threshold sensitivity.
- Phase-design variants.

## 7. Discussion

- When curriculum helps and when it does not.
- Interpretation of HeaRT gaps.
- Stability across seeds and datasets.
- Limitations of the current experimental budget.

## 8. Conclusion

- Summary of findings.
- Practical implications.
- Future work: stronger samplers, larger graphs, better ranking losses.