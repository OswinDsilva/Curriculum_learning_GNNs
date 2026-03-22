from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Literal

import numpy as np
import torch
import torch.nn.functional as F

# Allow running as a script from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import GAT, GCN
from models.base import LinkPredictor
from negative_sampling.heart import HeaRTEvaluator, resolve_heuristic_name
from utils.data_utils import (
    get_random_negatives,
    prepare_link_prediction_data,
    to_undirected_unique,
)
from utils.logging_utils import CheckpointManager, ExperimentLogger
from utils.metrics import compute_all_metrics


# ---------------------------------------------------------------------------
# Core training / evaluation functions
# ---------------------------------------------------------------------------


def train_epoch(
    model: LinkPredictor,
    data_dict: Dict,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    neg_ratio: int = 1,
    seed: int | None = None,
) -> float:
    """Run one training epoch with freshly sampled random negatives."""
    model.train()
    optimizer.zero_grad()

    x = data_dict["x"].to(device)
    train_edge_index = data_dict["train_edge_index"].to(device)
    pos_edge_index = data_dict["train_pos_edge_index"].to(device)

    num_pos = pos_edge_index.size(1)
    num_neg = num_pos * neg_ratio

    all_pos = to_undirected_unique(data_dict["data"].edge_index, data_dict["num_nodes"])
    neg_edge_index = get_random_negatives(
        edge_index=all_pos,
        num_nodes=data_dict["num_nodes"],
        num_samples=num_neg,
        seed=seed,
    ).to(device)

    z = model.encode(x, train_edge_index)
    pos_logits = model.decode(z, pos_edge_index)
    neg_logits = model.decode(z, neg_edge_index)

    logits = torch.cat([pos_logits, neg_logits])
    labels = torch.cat(
        [
            torch.ones(num_pos, device=device),
            torch.zeros(num_neg, device=device),
        ]
    )

    loss = F.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()
    optimizer.step()

    return float(loss.item())


@torch.no_grad()
def evaluate(
    model: LinkPredictor,
    data_dict: Dict,
    split: Literal["val", "test"],
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on val or test split using pre-fixed negatives."""
    model.eval()

    x = data_dict["x"].to(device)
    train_edge_index = data_dict["train_edge_index"].to(device)

    pos_edge_index = data_dict[f"{split}_pos_edge_index"].to(device)
    neg_edge_index = data_dict[f"{split}_neg_edge_index"].to(device)

    z = model.encode(x, train_edge_index)
    pos_scores = model.decode(z, pos_edge_index)
    neg_scores = model.decode(z, neg_edge_index)

    return compute_all_metrics(pos_scores, neg_scores)


# ---------------------------------------------------------------------------
# Build model helper
# ---------------------------------------------------------------------------


def build_model(args: argparse.Namespace, num_features: int) -> LinkPredictor:
    """Instantiate GCN or GAT from parsed args."""
    kwargs = dict(
        in_channels=num_features,
        hidden_channels=args.hidden_channels,
        out_channels=args.out_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
        decoder=args.decoder,
    )
    if args.model == "gcn":
        return GCN(**kwargs)
    if args.model == "gat":
        return GAT(**kwargs, heads=args.heads)
    raise ValueError(f"Unknown model '{args.model}'. Choose: gcn, gat")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baseline GNN link prediction training"
    )
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--model", type=str, default="gcn", choices=["gcn", "gat"])
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--out_channels", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4, help="GAT attention heads")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--decoder", type=str, default="inner_product")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--neg_ratio", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--save_dir", type=str, default="results/baseline")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--heart", action="store_true")
    parser.add_argument("--heart_heuristic", action="append", default=[])
    parser.add_argument("--num_neg_per_pos", type=int, default=100)
    parser.add_argument("--precomputed_dir", type=str, default="data/precomputed")
    parser.add_argument("--experiment_condition", type=str, default="baseline")
    parser.add_argument("--no_tensorboard", action="store_true")
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    print(f"Loading {args.dataset}...")
    data_dict = prepare_link_prediction_data(
        dataset_name=args.dataset,
        root=args.data_root,
        seed=args.seed,
    )
    print(
        f"  Nodes: {data_dict['num_nodes']} | "
        f"Train pos: {data_dict['train_pos_edge_index'].size(1)} | "
        f"Val pos: {data_dict['val_pos_edge_index'].size(1)} | "
        f"Test pos: {data_dict['test_pos_edge_index'].size(1)}"
    )

    # Model
    model = build_model(args, num_features=data_dict["num_features"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(
        f"Model: {model.__class__.__name__} | params: {sum(p.numel() for p in model.parameters()):,}"
    )

    # Logging
    exp_name = f"{args.dataset}_{args.model}_seed{args.seed}"
    if args.experiment_condition != "baseline":
        exp_name = (
            f"{args.experiment_condition}_{args.dataset}_{args.model}_seed{args.seed}"
        )
    logger = ExperimentLogger(
        log_dir=args.log_dir,
        experiment_name=exp_name,
        use_tensorboard=not args.no_tensorboard,
    )
    ckpt = CheckpointManager(
        checkpoint_dir=args.checkpoint_dir,
        experiment_name=exp_name,
    )

    # Training loop
    t0 = time.time()
    val_metrics: Dict[str, float] = {}
    for epoch in range(1, args.epochs + 1):
        ep_seed = args.seed * 10000 + epoch
        loss = train_epoch(
            model, data_dict, optimizer, device, neg_ratio=args.neg_ratio, seed=ep_seed
        )

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val_metrics = evaluate(model, data_dict, "val", device)
            ckpt.save(model, epoch, val_metrics["auc"])
            logger.log_epoch(epoch, loss, val_metrics)
            print(
                f"Epoch {epoch:4d} | loss {loss:.4f} | "
                f"val AUC {val_metrics['auc']:.4f} | AP {val_metrics['ap']:.4f} | "
                f"MRR {val_metrics['mrr']:.4f}"
            )

    training_time = time.time() - t0

    # Final test evaluation using best checkpoint
    ckpt.load_best(model)
    test_metrics = evaluate(model, data_dict, "test", device)
    heart_metrics: Dict[str, float] = {}
    if args.heart:
        heuristics = args.heart_heuristic or ["cn"]
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
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")
    if heart_metrics:
        print("\n=== HeaRT Results ===")
        for k, v in heart_metrics.items():
            print(f"  {k}: {v:.4f}")

    # Save results
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    result = {
        "config": vars(args),
        "condition": args.experiment_condition,
        "standard": test_metrics,
        "heart": heart_metrics,
        "training_time_seconds": round(training_time, 2),
    }
    result_path = f"{args.save_dir}/{exp_name}.json"
    logger.save_results_json(result_path, result)
    logger.save_csv(f"{args.save_dir}/{exp_name}_epochs.csv")
    logger.close()
    print(f"\nResults saved to {result_path}")


if __name__ == "__main__":
    main()
