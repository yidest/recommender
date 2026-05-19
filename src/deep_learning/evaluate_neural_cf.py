"""Evaluate a trained Neural CF model with ranking metrics."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch

from src.config import (
    DEFAULT_TOP_N,
    MOVIE_TITLES_PATH,
    NEURAL_CF_MODEL_PATH,
    TEST_RATINGS_PATH,
    TRAIN_RATINGS_PATH,
)
from src.data_processing import load_movie_titles
from src.deep_learning.model import NeuralCollaborativeFiltering
from src.ranking_metrics import load_positive_test_items, load_rated_movies_for_users


def load_neural_cf_checkpoint(
    model_path: Path,
    device: torch.device,
) -> tuple[NeuralCollaborativeFiltering, dict]:
    checkpoint = torch.load(model_path, map_location=device)
    metadata = checkpoint["metadata"]
    model = NeuralCollaborativeFiltering(
        num_users=int(metadata["num_users"]),
        num_movies=int(metadata["num_movies"]),
        embedding_dim=int(metadata["embedding_dim"]),
        hidden_dims=tuple(int(value) for value in metadata["hidden_dims"]),
        dropout=float(metadata["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, metadata


def score_candidate_movies(
    model: NeuralCollaborativeFiltering,
    user_idx: int,
    candidate_movie_indices: list[int],
    device: torch.device,
    batch_size: int,
) -> list[tuple[float, int]]:
    scored: list[tuple[float, int]] = []

    with torch.no_grad():
        for start in range(0, len(candidate_movie_indices), batch_size):
            batch_movie_indices = candidate_movie_indices[start : start + batch_size]
            user_tensor = torch.full(
                (len(batch_movie_indices),),
                user_idx,
                dtype=torch.long,
                device=device,
            )
            movie_tensor = torch.tensor(batch_movie_indices, dtype=torch.long, device=device)
            scores = model(user_tensor, movie_tensor).detach().cpu().tolist()
            scored.extend((float(score), int(movie_idx)) for score, movie_idx in zip(scores, batch_movie_indices))

    return scored


def ndcg_for_single_positive(rank: int | None) -> float:
    if rank is None:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def evaluate_neural_cf_ranking(
    model: NeuralCollaborativeFiltering,
    metadata: dict,
    top_k: int,
    rating_threshold: float,
    max_users: int | None,
    scan_multiplier: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, float | int]:
    positive_row_limit = max_users * scan_multiplier if max_users is not None else None
    positives = load_positive_test_items(
        test_ratings_path=TEST_RATINGS_PATH,
        rating_threshold=rating_threshold,
        max_users=positive_row_limit,
    )
    if positives.empty:
        raise RuntimeError("No positive test ratings found for Neural CF evaluation.")

    user_to_idx = {int(user_id): int(user_idx) for user_id, user_idx in metadata["user_to_idx"].items()}
    movie_to_idx = {int(movie_id): int(movie_idx) for movie_id, movie_idx in metadata["movie_to_idx"].items()}
    idx_to_movie = {int(movie_idx): int(movie_id) for movie_idx, movie_id in metadata["idx_to_movie"].items()}
    candidate_movie_indices = sorted(idx_to_movie)

    positive_user_ids = set(positives["user_id"].astype(int))
    rated_movies_by_user = load_rated_movies_for_users(TRAIN_RATINGS_PATH, positive_user_ids)
    movie_titles = load_movie_titles(MOVIE_TITLES_PATH)
    known_movie_ids = set(movie_titles["movie_id"].astype(int))

    hits = 0
    evaluated = 0
    skipped = 0
    ndcg_sum = 0.0
    candidate_count_sum = 0

    for row in positives.itertuples(index=False):
        user_id = int(row.user_id)
        held_out_movie_id = int(row.movie_id)

        if user_id not in user_to_idx or held_out_movie_id not in movie_to_idx:
            skipped += 1
            continue
        if held_out_movie_id not in known_movie_ids:
            skipped += 1
            continue

        rated_movie_ids = rated_movies_by_user.get(user_id, set())
        if held_out_movie_id in rated_movie_ids:
            skipped += 1
            continue

        rated_movie_indices = {
            movie_to_idx[movie_id]
            for movie_id in rated_movie_ids
            if movie_id in movie_to_idx
        }
        candidates = [
            movie_idx
            for movie_idx in candidate_movie_indices
            if movie_idx not in rated_movie_indices
        ]
        if not candidates:
            skipped += 1
            continue

        scored = score_candidate_movies(
            model=model,
            user_idx=user_to_idx[user_id],
            candidate_movie_indices=candidates,
            device=device,
            batch_size=batch_size,
        )
        top_scored = sorted(scored, key=lambda item: (item[0], -item[1]), reverse=True)[:top_k]
        top_movie_indices = [movie_idx for _, movie_idx in top_scored]
        held_out_movie_idx = movie_to_idx[held_out_movie_id]

        rank = None
        if held_out_movie_idx in top_movie_indices:
            rank = top_movie_indices.index(held_out_movie_idx) + 1
            hits += 1

        ndcg_sum += ndcg_for_single_positive(rank)
        evaluated += 1
        candidate_count_sum += len(candidates)

        if max_users is not None and evaluated >= max_users:
            break

    if evaluated == 0:
        raise RuntimeError("No eligible Neural CF evaluation rows.")

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
    parser = argparse.ArgumentParser(description="Evaluate a Neural CF checkpoint.")
    parser.add_argument("--model-path", type=Path, default=NEURAL_CF_MODEL_PATH)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--rating-threshold", type=float, default=4.0)
    parser.add_argument("--max-users", type=int, help="Optional positive test row cap.")
    parser.add_argument(
        "--scan-multiplier",
        type=int,
        default=20,
        help="Scan extra positive test rows to find eligible users for partial checkpoints.",
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, help="Optional CSV path for metric output.")
    args = parser.parse_args()

    if not args.model_path.exists():
        raise FileNotFoundError(
            f"Neural CF checkpoint not found: {args.model_path}\n"
            "Run `python -m src.deep_learning.train_neural_cf` first."
        )

    device = torch.device(args.device)
    model, metadata = load_neural_cf_checkpoint(args.model_path, device)
    metrics = evaluate_neural_cf_ranking(
        model=model,
        metadata=metadata,
        top_k=args.top_k,
        rating_threshold=args.rating_threshold,
        max_users=args.max_users,
        scan_multiplier=args.scan_multiplier,
        batch_size=args.batch_size,
        device=device,
    )

    print(f"Neural CF ranking metrics at K={metrics['top_k']}:")
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
        print(f"Saved Neural CF metric summary to: {args.output}")


if __name__ == "__main__":
    main()
