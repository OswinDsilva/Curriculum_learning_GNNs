from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from curriculum import CurriculumScheduler
from experiments.train_baseline import build_model, evaluate
from models.base import LinkPredictor
from negative_sampling.heart import HeaRTEvaluator, resolve_heuristic_name
from negative_sampling.sampler import DifficultyBasedSampler
from utils.data_utils import prepare_link_prediction_data
from utils.logging_utils import CheckpointManager, ExperimentLogger


def load_precomputed_candidates(
    dataset: str,
    heuristic: str,
    precomputed_dir: str | Path,
) -> tuple[torch.Tensor, np.ndarray]:
    canonical = resolve_heuristic_name(heuristic)
    path = Path(precomputed_dir) / f"{dataset}_{canonical}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing precomputed negatives at {path}. "
            f"Run scripts/precompute_scores.py --dataset {dataset} --heuristic {canonical}."
        )

    payload = np.load(path, allow_pickle=True)
    candidates = torch.from_numpy(payload["candidates"]).long()
    scores = payload["scores"].astype(np.float32, copy=False)
    return candidates, scores


def train_epoch_with_negatives(
    model: LinkPredictor,
    data_dict: Dict[str, Any],
    optimizer: torch.optim.Optimizer,
    neg_edge_index: Tensor,
    device: torch.device,
) -> float:
    model.train()
    optimizer.zero_grad()

    x = data_dict["x"].to(device)
    train_edge_index = data_dict["train_edge_index"].to(device)
    pos_edge_index = data_dict["train_pos_edge_index"].to(device)
    neg_edge_index = neg_edge_index.to(device)

    num_pos = pos_edge_index.size(1)
    num_neg = neg_edge_index.size(1)

    z = model.encode(x, train_edge_index)
    pos_logits = model.decode(z, pos_edge_index)
    neg_logits = model.decode(z, neg_edge_index)

    logits = torch.cat([pos_logits, neg_logits])
    labels = torch.cat([
        torch.ones(num_pos, device=device),
        torch.zeros(num_neg, device=device),
    ])

    loss = F.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()
    optimizer.step()
    return float(loss.item())


def train_with_curriculum(
    model: LinkPredictor,
    data_dict: Dict[str, Any],
    optimizer: torch.optim.Optimizer,
    scheduler: CurriculumScheduler,
    sampler: DifficultyBasedSampler,
    device: torch.device,
    epochs: int = 300,
    eval_every: int = 10,
    logger: ExperimentLogger | None = None,
    checkpoint_manager: CheckpointManager | None = None,
    neg_ratio: int = 1,
) -> Dict[str, Any]:
    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    if eval_every <= 0:
        raise ValueError("eval_every must be positive.")

    num_neg_samples = int(data_dict["train_pos_edge_index"].size(1) * neg_ratio)
    training_time_start = time.time()
    val_metrics: Dict[str, float] = {}
    last_loss = 0.0

    for epoch in range(1, epochs + 1):
        ratios = scheduler.get_current_difficulty_ratios()
        neg_edges = sampler.sample_mixed(
            num_neg_samples,
            weights=ratios,
            epoch_offset=epoch,
        )
        last_loss = train_epoch_with_negatives(
            model,
            data_dict,
            optimizer,
            neg_edges,
            device,
        )

        if epoch % eval_every == 0 or epoch == epochs:
            val_metrics = evaluate(model, data_dict, "val", device)
            scheduler.step(val_metrics["auc"], epoch)

            extra = {
                "phase": scheduler.current_phase_idx,
                "phase_name": scheduler.current_phase.name,
                "phase_changed": scheduler.phase_changed,
                "competence": scheduler.competence_meter.get_competence(),
                "easy_ratio": ratios[0],
                "medium_ratio": ratios[1],
                "hard_ratio": ratios[2],
            }
            if logger is not None:
                logger.log_epoch(epoch, last_loss, val_metrics, extra=extra)

            if scheduler.phase_changed:
                print(f"Epoch {epoch}: transitioned to phase {scheduler.current_phase_idx}")

            if checkpoint_manager is not None:
                checkpoint_manager.save(
                    model,
                    epoch,
                    val_metrics["auc"],
                    extra={"curriculum": scheduler.get_phase_summary()},
                )

    if checkpoint_manager is not None:
        checkpoint_manager.load_best(model)

    test_metrics = evaluate(model, data_dict, "test", device)
    training_time = time.time() - training_time_start
    return {
        "test_metrics": test_metrics,
        "phase_summary": scheduler.get_phase_summary(),
        "last_val_metrics": val_metrics,
        "final_loss": last_loss,
        "training_time_seconds": round(training_time, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Curriculum GNN link prediction training")
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--model", type=str, default="gcn", choices=["gcn", "gat"])
    parser.add_argument("--heuristic", type=str, default="cn")
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--out_channels", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--decoder", type=str, default="inner_product")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--neg_ratio", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--save_dir", type=str, default="results/curriculum")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--precomputed_dir", type=str, default="data/precomputed")
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--fixed_phase_epochs", type=int, default=75)
    parser.add_argument("--competence_window", type=int, default=5)
    parser.add_argument("--heart", action="store_true")
    parser.add_argument("--heart_heuristic", action="append", default=[])
    parser.add_argument("--num_neg_per_pos", type=int, default=100)
    parser.add_argument("--no_tensorboard", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_dict = prepare_link_prediction_data(
        dataset_name=args.dataset,
        root=args.data_root,
        seed=args.seed,
    )
    candidates, scores = load_precomputed_candidates(
        args.dataset,
        args.heuristic,
        args.precomputed_dir,
    )
    sampler = DifficultyBasedSampler(candidates, scores, seed=args.seed)
    scheduler = CurriculumScheduler(
        adaptive=args.adaptive,
        fixed_phase_epochs=args.fixed_phase_epochs,
        competence_window=args.competence_window,
    )

    model = build_model(args, num_features=data_dict["num_features"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    exp_name = (
        f"{args.dataset}_{args.model}_{resolve_heuristic_name(args.heuristic)}_seed{args.seed}"
    )
    logger = ExperimentLogger(
        log_dir=args.log_dir,
        experiment_name=exp_name,
        use_tensorboard=not args.no_tensorboard,
    )
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=args.checkpoint_dir,
        experiment_name=exp_name,
    )

    results = train_with_curriculum(
        model=model,
        data_dict=data_dict,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        device=device,
        epochs=args.epochs,
        eval_every=args.eval_every,
        logger=logger,
        checkpoint_manager=checkpoint_manager,
        neg_ratio=args.neg_ratio,
    )
    heart_metrics: Dict[str, float] = {}
    if args.heart:
        heuristics = args.heart_heuristic or [args.heuristic]
        heart = HeaRTEvaluator(
            data=data_dict["data"],
            heuristics=[resolve_heuristic_name(h) for h in heuristics],
            num_neg_per_pos=args.num_neg_per_pos,
            precomputed_dir=args.precomputed_dir,
            seed=args.seed,
            dataset_name=args.dataset,
        )
        heart_metrics = heart.evaluate_model(model, data_dict, device)

    print("\n=== Test Results ===")
    for key, value in results["test_metrics"].items():
        print(f"  {key}: {value:.4f}")
    if heart_metrics:
        print("\n=== HeaRT Results ===")
        for key, value in heart_metrics.items():
            print(f"  {key}: {value:.4f}")

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    result_path = Path(args.save_dir) / f"{exp_name}.json"
    epoch_csv_path = Path(args.save_dir) / f"{exp_name}_epochs.csv"
    payload = {
        "config": vars(args),
        "standard": results["test_metrics"],
        "heart": heart_metrics,
        "phase_summary": results["phase_summary"],
        "last_val_metrics": results["last_val_metrics"],
        "final_loss": results["final_loss"],
        "training_time_seconds": results["training_time_seconds"],
        "bucket_sizes": sampler.get_bucket_sizes(),
    }
    logger.save_results_json(str(result_path), payload)
    logger.save_csv(str(epoch_csv_path))
    logger.close()
    print(f"\nResults saved to {result_path}")


if __name__ == "__main__":
    main()