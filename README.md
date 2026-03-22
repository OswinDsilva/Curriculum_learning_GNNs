# Curriculum Learning with Adaptive Negative Sampling for GNN Link Prediction

This repository implements a multi-phase research pipeline for graph link prediction with curriculum learning, difficulty-aware negative sampling, and HeaRT evaluation.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you use CUDA, install the matching PyTorch wheel first, then install `torch-geometric`.

## Verify Environment

```bash
python scripts/verify_environment.py
```

## Quickstart

Run a single baseline experiment:

```bash
.venv/bin/python experiments/train_baseline.py \
	--dataset cora \
	--model gcn \
	--seed 0 \
	--heart \
	--no_tensorboard
```

Run a single curriculum experiment:

```bash
.venv/bin/python experiments/train_curriculum.py \
	--dataset cora \
	--model gcn \
	--heuristic cn \
	--seed 0 \
	--adaptive \
	--heart \
	--no_tensorboard
```

Run dataset exploration and split checks:

```bash
python experiments/data_exploration.py --dataset cora
```

Run test suite:

```bash
pytest -q
```

## Project Structure

```text
configs/              YAML experiment settings
curriculum/           competence tracking and phase scheduling
docs/                 per-phase notes and report outline
experiments/          training, evaluation, aggregation, plotting
models/               GCN, GAT, decoders
negative_sampling/    heuristics, sampler, HeaRT evaluator
notebooks/            analysis notebook
results/              run outputs, summaries, and figures
scripts/              environment checks and orchestration scripts
tests/                unit and integration tests
utils/                data, metrics, and logging helpers
```

## Reproducing Results

Run the automation script for the desired stage:

```bash
./scripts/run_all_experiments.sh baseline
./scripts/run_all_experiments.sh curriculum
./scripts/run_all_experiments.sh ablation
./scripts/run_all_experiments.sh aggregate
```

You can also run everything end-to-end with:

```bash
./scripts/run_all_experiments.sh all
```

## Results Summary

Current checked-in outputs are smoke-test scale rather than the full 10-seed study. Based on the current available runs:

| Condition | Dataset | Model | Metric | Value |
| --- | --- | --- | --- | ---: |
| Baseline | Cora | GCN | AUC | 0.9170 |
| Baseline | Cora | GCN | MRR | 0.3222 |
| Curriculum (CN) | Cora | GCN | AUC | 0.8760 |
| Curriculum (CN) | Cora | GCN | MRR | 0.1591 |

Use [experiments/aggregate_results.py](experiments/aggregate_results.py) and [experiments/statistical_analysis.py](experiments/statistical_analysis.py) after full experiment execution to regenerate publication-ready tables.

## Citation

If you reference the evaluation setup, cite HeaRT:

```bibtex
@inproceedings{chamberlain2023heart,
	title={HeaRT: Hard Evaluation for Graph Representation Learning},
	author={Chamberlain, Benjamin and others},
	booktitle={Advances in Neural Information Processing Systems},
	year={2023}
}
```

## Negative Result Note

Negative results are expected and should be kept. If a curriculum preset or heuristic underperforms the baseline, that is part of the empirical story and should remain visible in the saved outputs.

## Notes

- Datasets are auto-downloaded to `data/` by PyG.
- The current repository includes phases 1-7 of the planned workflow.
- Full statistical conclusions require the complete multi-seed experiment matrix, not just smoke runs.
