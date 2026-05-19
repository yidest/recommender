"""Evaluate recommendation lists with ranking metrics."""

import argparse
import heapq
import math
from pathlib import Path

import pandas as pd

from src.config import (
    DEFAULT_TOP_N,
    MOVIE_TITLES_PATH,
    TEST_RATINGS_PATH,
    TRAIN_RATINGS_PATH,
    TRAINED_MODEL_PATH,
)
from src.data_processing import load_movie_titles
from src.model_store import load_model
from src.models import TruncatedSVDModel


def load_positive_test_items(
    test_ratings_path: Path,
    rating_threshold: float,
    max_users: int | None,
) -> pd.DataFrame:
    test_df = pd.read_csv(test_ratings_path)
    test_df["movie_id"] = pd.to_numeric(test_df["movie_id"], errors="coerce")
    test_df["user_id"] = pd.to_numeric(test_df["user_id"], errors="coerce")
    test_df["rating"] = pd.to_numeric(test_df["rating"], errors="coerce")
    test_df = test_df.dropna(subset=["movie_id", "user_id", "rating"])
    test_df["movie_id"] = test_df["movie_id"].astype(int)
    test_df["user_id"] = test_df["user_id"].astype(int)
    positives = test_df.loc[test_df["rating"] >= rating_threshold].copy()

    if max_users is not None:
        positives = positives.head(max_users)
    return positives


def load_rated_movies_for_users(
    train_ratings_path: Path,
    user_ids: set[int],
    chunksize: int = 1_000_000,
) -> dict[int, set[int]]:
    rated_movies = {user_id: set() for user_id in user_ids}

    for chunk in pd.read_csv(
        train_ratings_path,
        usecols=["movie_id", "user_id"],
        chunksize=chunksize,
    ):
        chunk["movie_id"] = pd.to_numeric(chunk["movie_id"], errors="coerce")
        chunk["user_id"] = pd.to_numeric(chunk["user_id"], errors="coerce")
        chunk = chunk.dropna(subset=["movie_id", "user_id"])
        chunk["movie_id"] = chunk["movie_id"].astype(int)
        chunk["user_id"] = chunk["user_id"].astype(int)
        chunk = chunk.loc[chunk["user_id"].isin(user_ids)]

        for user_id, movie_id in zip(chunk["user_id"], chunk["movie_id"]):
            rated_movies[int(user_id)].add(int(movie_id))

    return rated_movies


def top_k_movie_ids(
    model: TruncatedSVDModel,
    user_id: int,
    candidate_movie_ids: list[int],
    top_k: int,
) -> list[int]:
    scored_movies = (
        (model.predict(movie_id, user_id), movie_id)
        for movie_id in candidate_movie_ids
    )
    top_scored = heapq.nlargest(
        top_k,
        scored_movies,
        key=lambda item: (item[0], -item[1]),
    )
    return [movie_id for _, movie_id in top_scored]


def ndcg_for_single_positive(rank: int | None) -> float:
    if rank is None:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def evaluate_ranking_metrics(
    model: TruncatedSVDModel,
    train_ratings_path: Path = TRAIN_RATINGS_PATH,
    test_ratings_path: Path = TEST_RATINGS_PATH,
    movie_titles_path: Path = MOVIE_TITLES_PATH,
    top_k: int = DEFAULT_TOP_N,
    rating_threshold: float = 4.0,
    max_users: int | None = None,
) -> dict[str, float | int]:
    positives = load_positive_test_items(
        test_ratings_path=test_ratings_path,
        rating_threshold=rating_threshold,
        max_users=max_users,
    )
    if positives.empty:
        raise RuntimeError("No positive test ratings found for ranking evaluation.")

    movie_titles = load_movie_titles(movie_titles_path)
    all_movie_ids = sorted(set(movie_titles["movie_id"].astype(int)))
    all_movie_id_set = set(all_movie_ids)
    positive_user_ids = set(positives["user_id"].astype(int))
    rated_movies_by_user = load_rated_movies_for_users(train_ratings_path, positive_user_ids)

    hits = 0
    evaluated = 0
    skipped = 0
    ndcg_sum = 0.0
    candidate_count_sum = 0

    for row in positives.itertuples(index=False):
        user_id = int(row.user_id)
        held_out_movie_id = int(row.movie_id)

        if model.user_to_idx is not None and user_id not in model.user_to_idx:
            skipped += 1
            continue
        if held_out_movie_id not in all_movie_id_set:
            skipped += 1
            continue

        rated_movies = rated_movies_by_user.get(user_id, set())
        if held_out_movie_id in rated_movies:
            skipped += 1
            continue

        candidates = [movie_id for movie_id in all_movie_ids if movie_id not in rated_movies]
        if not candidates:
            skipped += 1
            continue

        top_movie_ids = top_k_movie_ids(
            model=model,
            user_id=user_id,
            candidate_movie_ids=candidates,
            top_k=top_k,
        )

        rank = None
        if held_out_movie_id in top_movie_ids:
            rank = top_movie_ids.index(held_out_movie_id) + 1
            hits += 1

        ndcg_sum += ndcg_for_single_positive(rank)
        evaluated += 1
        candidate_count_sum += len(candidates)

    if evaluated == 0:
        raise RuntimeError("No eligible users/items could be evaluated.")

    hit_rate = hits / evaluated
    precision_at_k = hits / (evaluated * top_k)
    ndcg_at_k = ndcg_sum / evaluated
    avg_candidate_count = candidate_count_sum / evaluated

    return {
        "top_k": top_k,
        "rating_threshold": rating_threshold,
        "positive_test_items": int(len(positives)),
        "evaluated_items": evaluated,
        "skipped_items": skipped,
        "hits": hits,
        "hit_rate": hit_rate,
        "precision_at_k": precision_at_k,
        "ndcg_at_k": ndcg_at_k,
        "avg_candidate_count": avg_candidate_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Top-K recommendation metrics.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--rating-threshold", type=float, default=4.0)
    parser.add_argument(
        "--max-users",
        type=int,
        help="Optional cap on positive test rows for faster evaluation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional CSV path for saving the metric summary.",
    )
    args = parser.parse_args()

    if not TRAINED_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {TRAINED_MODEL_PATH}\n"
            "Run `python train.py` before evaluating ranking metrics."
        )

    model = load_model(TRAINED_MODEL_PATH)
    metrics = evaluate_ranking_metrics(
        model=model,
        top_k=args.top_k,
        rating_threshold=args.rating_threshold,
        max_users=args.max_users,
    )

    print(f"Ranking metrics at K={metrics['top_k']}:")
    print(f"  Rating threshold: {metrics['rating_threshold']}")
    print(f"  Positive test items: {metrics['positive_test_items']}")
    print(f"  Evaluated items: {metrics['evaluated_items']}")
    print(f"  Skipped items: {metrics['skipped_items']}")
    print(f"  Hits: {metrics['hits']}")
    print(f"  HitRate@{metrics['top_k']}: {metrics['hit_rate']:.4f}")
    print(f"  Precision@{metrics['top_k']}: {metrics['precision_at_k']:.4f}")
    print(f"  NDCG@{metrics['top_k']}: {metrics['ndcg_at_k']:.4f}")
    print(f"  Avg candidate movies: {metrics['avg_candidate_count']:.1f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([metrics]).to_csv(args.output, index=False)
        print(f"Saved metric summary to: {args.output}")


if __name__ == "__main__":
    main()
