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

The repository now contains the full 10-seed study matrix for baseline, curriculum, and ablation runs. A compact HeaRT MRR view (baseline vs best curriculum heuristic per dataset/model) is below:

| Dataset | Model | Baseline HeaRT MRR | Best Curriculum HeaRT MRR | Best Heuristic | Relative Change |
| --- | --- | ---: | ---: | --- | ---: |
| Cora | GCN | 0.5055 | 0.4124 | AA | -18.43% |
| Cora | GAT | 0.5226 | 0.5140 | RA | -1.65% |
| Citeseer | GCN | 0.4761 | 0.4593 | CN | -3.53% |
| Citeseer | GAT | 0.5781 | 0.5435 | RA | -6.00% |
| PubMed | GCN | 0.5420 | 0.5480 | AA | +1.12% |
| PubMed | GAT | 0.4752 | 0.4769 | RA | +0.37% |

Significance testing (`results/summaries/full_significance_table.csv`) contains 69 comparison rows total, with 23 rows for HeaRT MRR and 4 significant HeaRT MRR effects at p < 0.05.

Use [experiments/aggregate_results.py](experiments/aggregate_results.py), [experiments/statistical_analysis.py](experiments/statistical_analysis.py), and [experiments/generate_figures.py](experiments/generate_figures.py) to regenerate publication-ready tables and figures.

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
- Full multi-seed runs are now available under `results/` and summarized under `results/summaries/`.
