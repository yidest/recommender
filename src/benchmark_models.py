"""Run fixed ranking benchmarks for recommendation models."""

from __future__ import annotations

import argparse
import heapq
from pathlib import Path

import pandas as pd
import torch

from src.config import (
    MOVIE_TITLES_PATH,
    NEURAL_CF_MODEL_PATH,
    TEST_RATINGS_PATH,
    TRAIN_RATINGS_PATH,
    TRAINED_MODEL_PATH,
)
from src.data_processing import load_movie_titles
from src.deep_learning.evaluate_neural_cf import (
    load_neural_cf_checkpoint,
    ndcg_for_single_positive,
    score_candidate_movies,
)
from src.model_store import load_model
from src.ranking_metrics import load_positive_test_items, load_rated_movies_for_users, top_k_movie_ids


def load_popularity_ranking(
    train_ratings_path: Path,
    rating_threshold: float,
    chunksize: int = 1_000_000,
) -> list[int]:
    movie_stats: dict[int, list[float]] = {}

    for chunk in pd.read_csv(
        train_ratings_path,
        usecols=["movie_id", "rating"],
        chunksize=chunksize,
    ):
        chunk["movie_id"] = pd.to_numeric(chunk["movie_id"], errors="coerce")
        chunk["rating"] = pd.to_numeric(chunk["rating"], errors="coerce")
        chunk = chunk.dropna(subset=["movie_id", "rating"])
        chunk["movie_id"] = chunk["movie_id"].astype(int)
        positive_chunk = chunk.loc[chunk["rating"] >= rating_threshold]

        for movie_id, movie_rows in positive_chunk.groupby("movie_id"):
            stats = movie_stats.setdefault(int(movie_id), [0.0, 0.0])
            stats[0] += float(len(movie_rows))
            stats[1] += float(movie_rows["rating"].sum())

    return [
        movie_id
        for movie_id, _ in sorted(
            movie_stats.items(),
            key=lambda item: (item[1][0], item[1][1] / item[1][0], -item[0]),
            reverse=True,
        )
    ]


def top_popular_movie_ids(
    popularity_ranking: list[int],
    rated_movie_ids: set[int],
    top_k: int,
) -> list[int]:
    return [movie_id for movie_id in popularity_ranking if movie_id not in rated_movie_ids][:top_k]


def summarize_hits(
    top_movie_ids_by_model: dict[str, list[int]],
    held_out_movie_id: int,
    top_ks: list[int],
) -> dict[str, dict[int, tuple[int, float]]]:
    summary: dict[str, dict[int, tuple[int, float]]] = {}
    for model_name, top_movie_ids in top_movie_ids_by_model.items():
        model_summary = {}
        for top_k in top_ks:
            top_slice = top_movie_ids[:top_k]
            rank = top_slice.index(held_out_movie_id) + 1 if held_out_movie_id in top_slice else None
            hit = 1 if rank is not None else 0
            model_summary[top_k] = (hit, ndcg_for_single_positive(rank))
        summary[model_name] = model_summary
    return summary


def benchmark_models(args: argparse.Namespace) -> pd.DataFrame:
    top_ks = sorted(set(args.top_k))
    max_top_k = max(top_ks)

    positives = load_positive_test_items(
        test_ratings_path=TEST_RATINGS_PATH,
        rating_threshold=args.rating_threshold,
        max_users=args.max_users,
    )
    if positives.empty:
        raise RuntimeError("No positive test ratings found for benchmark.")

    positive_user_ids = set(positives["user_id"].astype(int))
    rated_movies_by_user = load_rated_movies_for_users(TRAIN_RATINGS_PATH, positive_user_ids)
    movie_titles = load_movie_titles(MOVIE_TITLES_PATH)
    all_movie_ids = sorted(set(movie_titles["movie_id"].astype(int)))
    all_movie_id_set = set(all_movie_ids)

    print("Loading Popularity baseline...")
    popularity_ranking = load_popularity_ranking(TRAIN_RATINGS_PATH, args.rating_threshold)

    print("Loading TruncatedSVDModel...")
    svd_model = load_model(args.svd_model_path)

    print("Loading NeuralCollaborativeFiltering...")
    device = torch.device(args.device)
    neural_model, neural_metadata = load_neural_cf_checkpoint(args.neural_model_path, device)
    neural_user_to_idx = {int(user_id): int(user_idx) for user_id, user_idx in neural_metadata["user_to_idx"].items()}
    neural_movie_to_idx = {int(movie_id): int(movie_idx) for movie_id, movie_idx in neural_metadata["movie_to_idx"].items()}
    neural_idx_to_movie = {
        int(movie_idx): int(movie_id)
        for movie_idx, movie_id in neural_metadata["idx_to_movie"].items()
    }
    neural_candidate_indices = sorted(neural_idx_to_movie)

    rows_by_model_k: dict[tuple[str, int], dict[str, float | int | str]] = {}
    model_names = [
        "PopularityBaseline",
        "TruncatedSVDModel",
        "NeuralCollaborativeFiltering",
        "NeuralCFWithSVDRecall",
    ]
    for model_name in model_names:
        for top_k in top_ks:
            rows_by_model_k[(model_name, top_k)] = {
                "model": model_name,
                "top_k": top_k,
                "rating_threshold": args.rating_threshold,
                "positive_test_items": int(len(positives)),
                "evaluated_items": 0,
                "skipped_items": 0,
                "hits": 0,
                "ndcg_sum": 0.0,
                "candidate_count_sum": 0,
            }

    evaluated = 0
    skipped = 0

    for row in positives.itertuples(index=False):
        user_id = int(row.user_id)
        held_out_movie_id = int(row.movie_id)
        rated_movie_ids = rated_movies_by_user.get(user_id, set())

        if held_out_movie_id not in all_movie_id_set:
            skipped += 1
            continue
        if held_out_movie_id in rated_movie_ids:
            skipped += 1
            continue
        if svd_model.user_to_idx is not None and user_id not in svd_model.user_to_idx:
            skipped += 1
            continue
        if user_id not in neural_user_to_idx or held_out_movie_id not in neural_movie_to_idx:
            skipped += 1
            continue

        candidate_movie_ids = [movie_id for movie_id in all_movie_ids if movie_id not in rated_movie_ids]
        if not candidate_movie_ids:
            skipped += 1
            continue

        popularity_top_ids = top_popular_movie_ids(popularity_ranking, rated_movie_ids, max_top_k)
        svd_top_ids = top_k_movie_ids(svd_model, user_id, candidate_movie_ids, max_top_k)
        svd_recall_ids = top_k_movie_ids(
            svd_model,
            user_id,
            candidate_movie_ids,
            max(args.svd_recall_size, max_top_k),
        )

        rated_movie_indices = {
            neural_movie_to_idx[movie_id]
            for movie_id in rated_movie_ids
            if movie_id in neural_movie_to_idx
        }
        neural_candidates = [
            movie_idx
            for movie_idx in neural_candidate_indices
            if movie_idx not in rated_movie_indices
        ]
        neural_scored = score_candidate_movies(
            model=neural_model,
            user_idx=neural_user_to_idx[user_id],
            candidate_movie_indices=neural_candidates,
            device=device,
            batch_size=args.batch_size,
        )
        neural_top_indices = [
            movie_idx
            for _, movie_idx in heapq.nlargest(
                max_top_k,
                neural_scored,
                key=lambda item: (item[0], -item[1]),
            )
        ]
        neural_top_ids = [neural_idx_to_movie[movie_idx] for movie_idx in neural_top_indices]

        svd_recall_candidate_indices = [
            neural_movie_to_idx[movie_id]
            for movie_id in svd_recall_ids
            if movie_id in neural_movie_to_idx
        ]
        neural_recall_scored = score_candidate_movies(
            model=neural_model,
            user_idx=neural_user_to_idx[user_id],
            candidate_movie_indices=svd_recall_candidate_indices,
            device=device,
            batch_size=args.batch_size,
        )
        neural_recall_top_indices = [
            movie_idx
            for _, movie_idx in heapq.nlargest(
                max_top_k,
                neural_recall_scored,
                key=lambda item: (item[0], -item[1]),
            )
        ]
        neural_recall_top_ids = [
            neural_idx_to_movie[movie_idx]
            for movie_idx in neural_recall_top_indices
        ]

        hit_summary = summarize_hits(
            {
                "PopularityBaseline": popularity_top_ids,
                "TruncatedSVDModel": svd_top_ids,
                "NeuralCollaborativeFiltering": neural_top_ids,
                "NeuralCFWithSVDRecall": neural_recall_top_ids,
            },
            held_out_movie_id=held_out_movie_id,
            top_ks=top_ks,
        )

        for model_name, model_summary in hit_summary.items():
            for top_k, (hit, ndcg) in model_summary.items():
                output_row = rows_by_model_k[(model_name, top_k)]
                output_row["evaluated_items"] = int(output_row["evaluated_items"]) + 1
                output_row["hits"] = int(output_row["hits"]) + hit
                output_row["ndcg_sum"] = float(output_row["ndcg_sum"]) + ndcg
                candidate_count = (
                    len(svd_recall_candidate_indices)
                    if model_name == "NeuralCFWithSVDRecall"
                    else len(candidate_movie_ids)
                )
                output_row["candidate_count_sum"] = int(output_row["candidate_count_sum"]) + candidate_count

        evaluated += 1

    if evaluated == 0:
        raise RuntimeError("No eligible benchmark rows.")

    output_rows = []
    for (model_name, top_k), row in rows_by_model_k.items():
        evaluated_items = int(row["evaluated_items"])
        hits = int(row["hits"])
        row["skipped_items"] = skipped
        row["hit_rate"] = hits / evaluated_items
        row["precision_at_k"] = hits / (evaluated_items * top_k)
        row["ndcg_at_k"] = float(row["ndcg_sum"]) / evaluated_items
        row["avg_candidate_count"] = int(row["candidate_count_sum"]) / evaluated_items
        del row["ndcg_sum"]
        del row["candidate_count_sum"]
        output_rows.append(row)

    return pd.DataFrame(output_rows).sort_values(["top_k", "model"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed ranking benchmarks.")
    parser.add_argument("--top-k", type=int, nargs="+", default=[10, 50, 100])
    parser.add_argument("--max-users", type=int, default=100)
    parser.add_argument("--rating-threshold", type=float, default=4.0)
    parser.add_argument("--svd-model-path", type=Path, default=TRAINED_MODEL_PATH)
    parser.add_argument("--neural-model-path", type=Path, default=NEURAL_CF_MODEL_PATH)
    parser.add_argument(
        "--svd-recall-size",
        type=int,
        default=1000,
        help="Number of SVD candidates for the NeuralCFWithSVDRecall benchmark row.",
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, help="Optional CSV path for benchmark results.")
    args = parser.parse_args()

    if not args.svd_model_path.exists():
        raise FileNotFoundError(f"SVD model not found: {args.svd_model_path}")
    if not args.neural_model_path.exists():
        raise FileNotFoundError(f"Neural CF checkpoint not found: {args.neural_model_path}")

    benchmark = benchmark_models(args)
    print("Fixed benchmark:")
    for row in benchmark.itertuples(index=False):
        print(
            f"  {row.model}: HitRate@{row.top_k}={row.hit_rate:.4f}, "
            f"Precision@{row.top_k}={row.precision_at_k:.4f}, "
            f"NDCG@{row.top_k}={row.ndcg_at_k:.4f}"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        benchmark.to_csv(args.output, index=False)
        print(f"Saved benchmark to: {args.output}")


if __name__ == "__main__":
    main()
