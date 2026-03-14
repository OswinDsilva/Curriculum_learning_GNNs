# Curriculum Learning with Adaptive Negative Sampling for GNN Link Prediction

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

Run dataset exploration and split checks:

```bash
python experiments/data_exploration.py --dataset cora
```

Run test suite:

```bash
pytest -q
```

## Notes

- Datasets are auto-downloaded to `data/` by PyG.
- Phase 1 includes data utilities, baseline model scaffolding (GCN/GAT), and smoke tests.
