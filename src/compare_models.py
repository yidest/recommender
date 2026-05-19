"""Compare TruncatedSVD and Neural CF ranking metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from src.config import DEFAULT_TOP_N, NEURAL_CF_MODEL_PATH, TRAINED_MODEL_PATH
from src.deep_learning.evaluate_neural_cf import evaluate_neural_cf_ranking, load_neural_cf_checkpoint
from src.model_store import load_model
from src.ranking_metrics import evaluate_ranking_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare SVD and Neural CF recommendation metrics.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--max-users", type=int, default=100)
    parser.add_argument("--rating-threshold", type=float, default=4.0)
    parser.add_argument("--svd-model-path", type=Path, default=TRAINED_MODEL_PATH)
    parser.add_argument("--neural-model-path", type=Path, default=NEURAL_CF_MODEL_PATH)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, help="Optional CSV path for comparison results.")
    args = parser.parse_args()

    if not args.svd_model_path.exists():
        raise FileNotFoundError(f"SVD model not found: {args.svd_model_path}")
    if not args.neural_model_path.exists():
        raise FileNotFoundError(f"Neural CF checkpoint not found: {args.neural_model_path}")

    print("Evaluating TruncatedSVDModel...")
    svd_model = load_model(args.svd_model_path)
    svd_metrics = evaluate_ranking_metrics(
        model=svd_model,
        top_k=args.top_k,
        rating_threshold=args.rating_threshold,
        max_users=args.max_users,
    )

    print("Evaluating NeuralCollaborativeFiltering...")
    device = torch.device(args.device)
    neural_model, neural_metadata = load_neural_cf_checkpoint(args.neural_model_path, device)
    neural_metrics = evaluate_neural_cf_ranking(
        model=neural_model,
        metadata=neural_metadata,
        top_k=args.top_k,
        rating_threshold=args.rating_threshold,
        max_users=args.max_users,
        scan_multiplier=20,
        batch_size=args.batch_size,
        device=device,
    )

    rows = []
    for model_name, metrics in [
        ("TruncatedSVDModel", svd_metrics),
        ("NeuralCollaborativeFiltering", neural_metrics),
    ]:
        rows.append(
            {
                "model": model_name,
                "top_k": metrics["top_k"],
                "rating_threshold": metrics["rating_threshold"],
                "evaluated_items": metrics["evaluated_items"],
                "hits": metrics["hits"],
                "hit_rate": metrics["hit_rate"],
                "precision_at_k": metrics["precision_at_k"],
                "ndcg_at_k": metrics["ndcg_at_k"],
                "avg_candidate_count": metrics["avg_candidate_count"],
            }
        )

    comparison = pd.DataFrame(rows)
    print("Model comparison:")
    for row in comparison.itertuples(index=False):
        print(
            f"  {row.model}: HitRate@{row.top_k}={row.hit_rate:.4f}, "
            f"Precision@{row.top_k}={row.precision_at_k:.4f}, "
            f"NDCG@{row.top_k}={row.ndcg_at_k:.4f}"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(args.output, index=False)
        print(f"Saved model comparison to: {args.output}")


if __name__ == "__main__":
    main()
