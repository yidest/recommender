"""Run small parameter experiments for the recommendation model."""

import argparse
import time
from pathlib import Path

import pandas as pd

from src.config import (
    DEFAULT_EXPERIMENT_RESULTS_PATH,
    RANDOM_SEED,
    TEST_RATINGS_PATH,
    TRAIN_RATINGS_PATH,
)
from src.models import BiasModel, TruncatedSVDModel, evaluate_rmse
from src.ranking_metrics import evaluate_ranking_metrics


def run_svd_experiments(
    n_components_values: list[int],
    top_k: int,
    max_users: int | None,
    include_ranking: bool,
) -> pd.DataFrame:
    results = []

    for n_components in n_components_values:
        started_at = time.perf_counter()
        print(f"Training TruncatedSVDModel with n_components={n_components}...")

        model = TruncatedSVDModel(
            bias_model=BiasModel(lambda_movie=25.0, lambda_user=10.0),
            n_components=n_components,
            random_state=RANDOM_SEED,
        )
        model.fit_from_csv(TRAIN_RATINGS_PATH)
        rmse = evaluate_rmse(model, TEST_RATINGS_PATH)

        row = {
            "model": "TruncatedSVDModel",
            "n_components": n_components,
            "rmse": rmse,
            "top_k": top_k if include_ranking else None,
            "max_users": max_users if include_ranking else None,
            "hit_rate": None,
            "precision_at_k": None,
            "evaluated_items": None,
            "elapsed_seconds": None,
        }

        if include_ranking:
            metrics = evaluate_ranking_metrics(
                model=model,
                top_k=top_k,
                max_users=max_users,
            )
            row["hit_rate"] = metrics["hit_rate"]
            row["precision_at_k"] = metrics["precision_at_k"]
            row["evaluated_items"] = metrics["evaluated_items"]

        row["elapsed_seconds"] = time.perf_counter() - started_at
        results.append(row)

        print(
            f"  RMSE={row['rmse']:.4f}"
            + (
                f", HitRate@{top_k}={row['hit_rate']:.4f}, "
                f"Precision@{top_k}={row['precision_at_k']:.4f}"
                if include_ranking
                else ""
            )
            + f", elapsed={row['elapsed_seconds']:.1f}s"
        )

    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TruncatedSVD parameter experiments.")
    parser.add_argument(
        "--n-components",
        type=int,
        nargs="+",
        default=[10, 20, 50],
        help="One or more n_components values to evaluate.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--max-users",
        type=int,
        default=100,
        help="Positive test row cap for ranking metrics.",
    )
    parser.add_argument(
        "--skip-ranking",
        action="store_true",
        help="Only compute RMSE and skip ranking metrics.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_EXPERIMENT_RESULTS_PATH,
        type=Path,
        help="CSV output path for experiment results.",
    )
    args = parser.parse_args()

    output_path = args.output

    results = run_svd_experiments(
        n_components_values=args.n_components,
        top_k=args.top_k,
        max_users=args.max_users,
        include_ranking=not args.skip_ranking,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"Saved experiment results to: {output_path}")


if __name__ == "__main__":
    main()
