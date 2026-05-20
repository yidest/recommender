"""Generate Top-N recommendations with a trained Neural CF checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from src.config import DEFAULT_TOP_N, MOVIE_TITLES_PATH, NEURAL_CF_MODEL_PATH, TRAIN_RATINGS_PATH
from src.data_processing import load_movie_titles
from src.deep_learning.evaluate_neural_cf import load_neural_cf_checkpoint, score_candidate_movies


def load_rated_movie_ids_for_user(user_id: int, chunksize: int = 1_000_000) -> set[int]:
    rated_movie_ids: set[int] = set()

    for chunk in pd.read_csv(
        TRAIN_RATINGS_PATH,
        usecols=["movie_id", "user_id"],
        chunksize=chunksize,
    ):
        chunk["movie_id"] = pd.to_numeric(chunk["movie_id"], errors="coerce")
        chunk["user_id"] = pd.to_numeric(chunk["user_id"], errors="coerce")
        chunk = chunk.dropna(subset=["movie_id", "user_id"])
        chunk["movie_id"] = chunk["movie_id"].astype(int)
        chunk["user_id"] = chunk["user_id"].astype(int)
        user_rows = chunk.loc[chunk["user_id"] == user_id]
        rated_movie_ids.update(int(movie_id) for movie_id in user_rows["movie_id"])

    return rated_movie_ids


def recommend_for_user(
    user_id: int,
    model_path: Path,
    top_n: int,
    batch_size: int,
    device: torch.device,
    candidate_movie_ids: list[int] | None = None,
) -> pd.DataFrame:
    model, metadata = load_neural_cf_checkpoint(model_path, device)
    user_to_idx = {int(raw_user_id): int(user_idx) for raw_user_id, user_idx in metadata["user_to_idx"].items()}
    movie_to_idx = {int(movie_id): int(movie_idx) for movie_id, movie_idx in metadata["movie_to_idx"].items()}
    idx_to_movie = {int(movie_idx): int(movie_id) for movie_idx, movie_id in metadata["idx_to_movie"].items()}

    if user_id not in user_to_idx:
        raise ValueError(
            f"User {user_id} was not found in the Neural CF checkpoint. "
            "Cold-start recommendations are not supported yet."
        )

    rated_movie_ids = load_rated_movie_ids_for_user(user_id)
    if not rated_movie_ids:
        raise ValueError(f"User {user_id} was not found in training data.")

    rated_movie_indices = {
        movie_to_idx[movie_id]
        for movie_id in rated_movie_ids
        if movie_id in movie_to_idx
    }
    if candidate_movie_ids is None:
        candidate_movie_indices = [
            movie_idx
            for movie_idx in sorted(idx_to_movie)
            if movie_idx not in rated_movie_indices
        ]
    else:
        candidate_movie_indices = [
            movie_to_idx[movie_id]
            for movie_id in candidate_movie_ids
            if movie_id in movie_to_idx and movie_to_idx[movie_id] not in rated_movie_indices
        ]
    if not candidate_movie_indices:
        raise ValueError("No candidate movies are available for Neural CF recommendation.")

    scored = score_candidate_movies(
        model=model,
        user_idx=user_to_idx[user_id],
        candidate_movie_indices=candidate_movie_indices,
        device=device,
        batch_size=batch_size,
    )
    top_scored = sorted(scored, key=lambda item: (item[0], -item[1]), reverse=True)[:top_n]
    rows = [
        {"movie_id": idx_to_movie[movie_idx], "neural_cf_score": score}
        for score, movie_idx in top_scored
    ]

    recommendations = pd.DataFrame(rows)
    movie_titles = load_movie_titles(MOVIE_TITLES_PATH)
    recommendations = recommendations.merge(movie_titles[["movie_id", "title"]], on="movie_id", how="left")
    recommendations["title"] = recommendations["title"].fillna("(unknown title)")
    return recommendations[["movie_id", "title", "neural_cf_score"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend movies with Neural CF.")
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--model-path", type=Path, default=NEURAL_CF_MODEL_PATH)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, help="Optional CSV path for recommendations.")
    args = parser.parse_args()

    if not args.model_path.exists():
        raise FileNotFoundError(
            f"Neural CF checkpoint not found: {args.model_path}\n"
            "Run `python -m src.deep_learning.train_neural_cf` first."
        )

    recommendations = recommend_for_user(
        user_id=args.user_id,
        model_path=args.model_path,
        top_n=args.top_n,
        batch_size=args.batch_size,
        device=torch.device(args.device),
    )

    output_df = recommendations.copy()
    output_df.insert(0, "rank", range(1, len(output_df) + 1))
    print(f"Top {args.top_n} Neural CF recommendations for user {args.user_id}:")
    for row in output_df.itertuples(index=False):
        print(f"{row.rank:>2}. movie_id={row.movie_id} | score={row.neural_cf_score:.4f} | {row.title}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(args.output, index=False)
        print(f"Saved Neural CF recommendations to: {args.output}")


if __name__ == "__main__":
    main()
