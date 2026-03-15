from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.train_baseline import build_model, evaluate
from negative_sampling.heart import HeaRTEvaluator, resolve_heuristic_name
from utils.data_utils import prepare_link_prediction_data


def run_evaluation(
    model,
    data_dict: Dict[str, Any],
    device: torch.device,
    heart_evaluator: HeaRTEvaluator | None = None,
) -> Dict[str, float]:
    metrics = evaluate(model, data_dict, "test", device)
    if heart_evaluator is not None:
        metrics.update(heart_evaluator.evaluate_model(model, data_dict, device))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained link prediction model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--model", type=str, default="gcn", choices=["gcn", "gat"])
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--out_channels", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--decoder", type=str, default="inner_product")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--heart", action="store_true")
    parser.add_argument("--heart_heuristic", action="append", default=[])
    parser.add_argument("--num_neg_per_pos", type=int, default=100)
    parser.add_argument("--precomputed_dir", type=str, default="data/precomputed")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dict = prepare_link_prediction_data(
        dataset_name=args.dataset,
        root=args.data_root,
        seed=args.seed,
    )

    model = build_model(args, num_features=data_dict["num_features"]).to(device)
    payload = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(payload["model_state_dict"])

    heart_evaluator = None
    if args.heart:
        heuristics = args.heart_heuristic or ["cn"]
        heart_evaluator = HeaRTEvaluator(
            data=data_dict["data"],
            heuristics=[resolve_heuristic_name(h) for h in heuristics],
            num_neg_per_pos=args.num_neg_per_pos,
            precomputed_dir=args.precomputed_dir,
            seed=args.seed,
            dataset_name=args.dataset,
        )

    metrics = run_evaluation(model, data_dict, device, heart_evaluator=heart_evaluator)
    print("=== Evaluation Results ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()