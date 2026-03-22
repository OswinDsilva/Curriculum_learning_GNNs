#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
MODE="${1:-all}"
SEED_START="${SEED_START:-0}"
SEED_END="${SEED_END:-9}"
RESULTS_DIR="${RESULTS_DIR:-results}"

SEEDS=$(seq "$SEED_START" "$SEED_END")
DATASETS=(cora citeseer pubmed)
MODELS=(gcn gat)
HEURISTICS=(cn aa ra)

run_baselines() {
  for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
      for seed in $SEEDS; do
        "$PYTHON" experiments/train_baseline.py \
          --dataset "$dataset" \
          --model "$model" \
          --seed "$seed" \
          --heart \
          --save_dir "$RESULTS_DIR/baseline"
      done
    done
  done
}

run_curriculum() {
  for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
      for heuristic in "${HEURISTICS[@]}"; do
        for seed in $SEEDS; do
          "$PYTHON" experiments/train_curriculum.py \
            --dataset "$dataset" \
            --model "$model" \
            --heuristic "$heuristic" \
            --seed "$seed" \
            --adaptive \
            --heart \
            --save_dir "$RESULTS_DIR/curriculum"
        done
      done
    done
  done
}

run_ablations() {
  for seed in $SEEDS; do
    "$PYTHON" experiments/train_baseline.py \
      --dataset cora --model gcn --seed "$seed" \
      --heart --experiment_condition abl-1 \
      --save_dir "$RESULTS_DIR/ablation"

    "$PYTHON" experiments/train_curriculum.py \
      --dataset cora --model gcn --heuristic cn --seed "$seed" \
      --curriculum_preset hard_from_start --heart \
      --experiment_condition abl-2 \
      --save_dir "$RESULTS_DIR/ablation"

    "$PYTHON" experiments/train_curriculum.py \
      --dataset cora --model gcn --heuristic cn --seed "$seed" \
      --curriculum_preset static_easy_hard --heart \
      --experiment_condition abl-3 \
      --save_dir "$RESULTS_DIR/ablation"

    "$PYTHON" experiments/train_curriculum.py \
      --dataset cora --model gcn --heuristic cn --seed "$seed" \
      --fixed_phase_epochs 75 --heart \
      --experiment_condition abl-4 \
      --save_dir "$RESULTS_DIR/ablation"

    "$PYTHON" experiments/train_curriculum.py \
      --dataset cora --model gcn --heuristic cn --seed "$seed" \
      --adaptive --curriculum_preset skip_mixed --heart \
      --experiment_condition abl-5 \
      --save_dir "$RESULTS_DIR/ablation"

    "$PYTHON" experiments/train_curriculum.py \
      --dataset cora --model gcn --heuristic cn --seed "$seed" \
      --adaptive --curriculum_preset extra_intermediate --heart \
      --experiment_condition abl-6 \
      --save_dir "$RESULTS_DIR/ablation"

    "$PYTHON" experiments/train_curriculum.py \
      --dataset cora --model gcn --heuristic cn --seed "$seed" \
      --adaptive --curriculum_preset low_thresholds --heart \
      --experiment_condition abl-10 \
      --save_dir "$RESULTS_DIR/ablation"

    "$PYTHON" experiments/train_curriculum.py \
      --dataset cora --model gcn --heuristic cn --seed "$seed" \
      --adaptive --curriculum_preset high_thresholds --heart \
      --experiment_condition abl-11 \
      --save_dir "$RESULTS_DIR/ablation"
  done
}

run_aggregate() {
  "$PYTHON" experiments/aggregate_results.py \
    --results_dir "$RESULTS_DIR" \
    --output_dir "$RESULTS_DIR/summaries"
}

case "$MODE" in
  baseline)
    run_baselines
    ;;
  curriculum)
    run_curriculum
    ;;
  ablation)
    run_ablations
    ;;
  aggregate)
    run_aggregate
    ;;
  all)
    run_baselines
    run_curriculum
    run_ablations
    run_aggregate
    ;;
  *)
    echo "Unknown mode: $MODE"
    exit 1
    ;;
esac

echo "Experiment mode '$MODE' complete."