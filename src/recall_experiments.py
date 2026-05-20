"""Run SVD recall-size experiments for Neural CF reranking."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from src.benchmark_models import benchmark_models
from src.config import DEFAULT_TOP_N, REPORTS_DIR


DEFAULT_RECALL_EXPERIMENT_PATH = REPORTS_DIR / "svd_recall_experiments.csv"


def run_recall_experiments(args: argparse.Namespace) -> pd.DataFrame:
    rows = []

    for recall_size in args.candidate_counts:
        benchmark_args = argparse.Namespace(
            top_k=args.top_k,
            max_users=args.max_users,
            rating_threshold=args.rating_threshold,
            svd_model_path=args.svd_model_path,
            neural_model_path=args.neural_model_path,
            svd_recall_size=recall_size,
            batch_size=args.batch_size,
            device=args.device,
        )
        print(f"Running SVD recall experiment with candidate_count={recall_size}...")
        start_time = time.perf_counter()
        benchmark = benchmark_models(benchmark_args)
        elapsed_seconds = time.perf_counter() - start_time

        recall_rows = benchmark.loc[benchmark["model"] == "NeuralCFWithSVDRecall"].copy()
        recall_rows["candidate_count"] = recall_size
        recall_rows["elapsed_seconds"] = elapsed_seconds
        rows.append(recall_rows)

        for row in recall_rows.itertuples(index=False):
            print(
                f"  K={row.top_k}: HitRate={row.hit_rate:.4f}, "
                f"NDCG={row.ndcg_at_k:.4f}, avg_candidates={row.avg_candidate_count:.1f}"
            )

    if not rows:
        raise RuntimeError("No recall experiment rows were produced.")

    result = pd.concat(rows, ignore_index=True)
    ordered_columns = [
        "candidate_count",
        "model",
        "top_k",
        "rating_threshold",
        "positive_test_items",
        "evaluated_items",
        "skipped_items",
        "hits",
        "hit_rate",
        "precision_at_k",
        "ndcg_at_k",
        "avg_candidate_count",
        "elapsed_seconds",
    ]
    return result[ordered_columns]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare SVD recall candidate counts.")
    parser.add_argument(
        "--candidate-counts",
        type=int,
        nargs="+",
        default=[100, 250, 500, 1000, 2000],
    )
    parser.add_argument("--top-k", type=int, nargs="+", default=[DEFAULT_TOP_N, 50, 100])
    parser.add_argument("--max-users", type=int, default=1000)
    parser.add_argument("--rating-threshold", type=float, default=4.0)
    parser.add_argument("--svd-model-path", type=Path, default=None)
    parser.add_argument("--neural-model-path", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=DEFAULT_RECALL_EXPERIMENT_PATH)
    args = parser.parse_args()

    from src.config import NEURAL_CF_MODEL_PATH, TRAINED_MODEL_PATH

    if args.svd_model_path is None:
        args.svd_model_path = TRAINED_MODEL_PATH
    if args.neural_model_path is None:
        args.neural_model_path = NEURAL_CF_MODEL_PATH

    results = run_recall_experiments(args)
    print("SVD recall-size experiment summary:")
    for row in results.itertuples(index=False):
        print(
            f"  candidates={row.candidate_count}, K={row.top_k}: "
            f"HitRate={row.hit_rate:.4f}, Precision={row.precision_at_k:.4f}, "
            f"NDCG={row.ndcg_at_k:.4f}, avg_candidates={row.avg_candidate_count:.1f}"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.output, index=False)
        print(f"Saved recall experiment results to: {args.output}")


if __name__ == "__main__":
    main()
