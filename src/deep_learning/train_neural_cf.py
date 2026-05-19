"""Train a Neural Collaborative Filtering model with BPR loss."""

from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import NEURAL_CF_DATA_CACHE_PATH, NEURAL_CF_MODEL_PATH, RANDOM_SEED, TRAIN_RATINGS_PATH
from src.deep_learning.dataset import (
    BPRDataset,
    build_movie_sampling_weights,
    build_cache_metadata,
    cache_metadata_matches,
    load_interaction_data,
    load_interaction_data_cache,
    sample_positive_pairs,
    save_interaction_data_cache,
    split_positive_pairs,
)
from src.deep_learning.model import NeuralCollaborativeFiltering, bpr_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_bpr_loss(
    model: NeuralCollaborativeFiltering,
    data_loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for user_idx, positive_idx, negative_idx in data_loader:
            user_idx = user_idx.to(device)
            positive_idx = positive_idx.to(device)
            negative_idx = negative_idx.to(device)

            positive_scores = model(user_idx, positive_idx)
            negative_scores = model(user_idx, negative_idx)
            loss = bpr_loss(positive_scores, negative_scores)
            batch_size = user_idx.size(0)
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size

    if total_count == 0:
        raise RuntimeError("Validation loader is empty.")
    return total_loss / total_count


def save_checkpoint(
    model: NeuralCollaborativeFiltering,
    model_path: Path,
    metadata: dict,
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": metadata,
        },
        model_path,
    )


def train_neural_cf(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device)

    cache_metadata = build_cache_metadata(
        ratings_path=TRAIN_RATINGS_PATH,
        positive_threshold=args.positive_threshold,
        max_rows=args.max_rows,
    )
    interaction_data = None
    uses_default_cache_path = args.data_cache_path == NEURAL_CF_DATA_CACHE_PATH
    use_cache = not args.no_data_cache and (args.max_rows is None or not uses_default_cache_path)
    if not use_cache and args.max_rows is not None and uses_default_cache_path and not args.quiet_data_loading:
        print("Skipping default Neural CF cache for --max-rows smoke training.")

    if use_cache and args.data_cache_path.exists() and not args.rebuild_data_cache:
        cached_data, cached_metadata = load_interaction_data_cache(args.data_cache_path)
        if cache_metadata_matches(cached_metadata, cache_metadata):
            interaction_data = cached_data
            print(f"Loaded Neural CF interaction cache: {args.data_cache_path}")
        else:
            print(f"Ignoring stale Neural CF interaction cache: {args.data_cache_path}")

    if interaction_data is None:
        interaction_data = load_interaction_data(
            ratings_path=TRAIN_RATINGS_PATH,
            positive_threshold=args.positive_threshold,
            max_rows=args.max_rows,
            max_positive_samples=None,
            seed=args.seed,
            chunk_size=args.chunk_size,
            show_progress=not args.quiet_data_loading,
        )
        if use_cache:
            save_interaction_data_cache(interaction_data, args.data_cache_path, cache_metadata)
            print(f"Saved Neural CF interaction cache to: {args.data_cache_path}")

    interaction_data = sample_positive_pairs(
        interaction_data=interaction_data,
        max_positive_samples=args.max_positive_samples,
        seed=args.seed,
    )
    train_pairs, validation_pairs = split_positive_pairs(
        positive_pairs=interaction_data.positive_pairs,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    movie_sampling_weights = None
    if args.negative_sampling_strategy in {"popularity", "mixed"}:
        movie_sampling_weights = build_movie_sampling_weights(
            positive_pairs=train_pairs,
            num_movies=len(interaction_data.movie_to_idx),
            popularity_power=args.negative_popularity_power,
        )

    train_dataset = BPRDataset(
        positive_pairs=train_pairs,
        user_rated_movies=interaction_data.user_rated_movies,
        num_movies=len(interaction_data.movie_to_idx),
        seed=args.seed,
        negative_sampling_strategy=args.negative_sampling_strategy,
        movie_sampling_weights=movie_sampling_weights,
        mixed_negative_probability=args.mixed_negative_probability,
    )
    validation_dataset = BPRDataset(
        positive_pairs=validation_pairs,
        user_rated_movies=interaction_data.user_rated_movies,
        num_movies=len(interaction_data.movie_to_idx),
        seed=args.seed + 1,
        negative_sampling_strategy=args.negative_sampling_strategy,
        movie_sampling_weights=movie_sampling_weights,
        mixed_negative_probability=args.mixed_negative_probability,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)

    model = NeuralCollaborativeFiltering(
        num_users=len(interaction_data.user_to_idx),
        num_movies=len(interaction_data.movie_to_idx),
        embedding_dim=args.embedding_dim,
        hidden_dims=tuple(args.hidden_dims),
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_validation_loss = float("inf")
    best_state_dict = copy.deepcopy(model.state_dict())
    stale_epochs = 0

    print(
        "Training NeuralCollaborativeFiltering "
        f"with {len(train_pairs)} train positives and {len(validation_pairs)} validation positives..."
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for user_idx, positive_idx, negative_idx in train_loader:
            user_idx = user_idx.to(device)
            positive_idx = positive_idx.to(device)
            negative_idx = negative_idx.to(device)

            optimizer.zero_grad()
            positive_scores = model(user_idx, positive_idx)
            negative_scores = model(user_idx, negative_idx)
            loss = bpr_loss(positive_scores, negative_scores)
            loss.backward()
            optimizer.step()

            batch_size = user_idx.size(0)
            train_loss_sum += float(loss.item()) * batch_size
            train_count += batch_size

        train_loss = train_loss_sum / train_count
        validation_loss = evaluate_bpr_loss(model, validation_loader, device)
        print(
            f"Epoch {epoch:02d}: train_bpr_loss={train_loss:.4f}, "
            f"validation_bpr_loss={validation_loss:.4f}"
        )

        if validation_loss < best_validation_loss - args.min_delta:
            best_validation_loss = validation_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    model.load_state_dict(best_state_dict)
    metadata = {
        "num_users": int(len(interaction_data.user_to_idx)),
        "num_movies": int(len(interaction_data.movie_to_idx)),
        "embedding_dim": int(args.embedding_dim),
        "hidden_dims": [int(hidden_dim) for hidden_dim in args.hidden_dims],
        "dropout": float(args.dropout),
        "positive_threshold": float(args.positive_threshold),
        "negative_sampling_strategy": args.negative_sampling_strategy,
        "negative_popularity_power": float(args.negative_popularity_power),
        "mixed_negative_probability": float(args.mixed_negative_probability),
        "user_to_idx": interaction_data.user_to_idx,
        "movie_to_idx": interaction_data.movie_to_idx,
        "idx_to_user": interaction_data.idx_to_user,
        "idx_to_movie": interaction_data.idx_to_movie,
        "best_validation_bpr_loss": float(best_validation_loss),
    }
    save_checkpoint(model, args.output, metadata)
    print(f"Best validation BPR loss: {best_validation_loss:.4f}")
    print(f"Saved Neural CF model to: {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Neural CF with BPR loss.")
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[64, 32])
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--positive-threshold", type=float, default=4.0)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=0.0001)
    parser.add_argument(
        "--negative-sampling-strategy",
        choices=["random", "popularity", "mixed"],
        default="random",
        help="Negative sampler for BPR training.",
    )
    parser.add_argument(
        "--negative-popularity-power",
        type=float,
        default=0.75,
        help="Power applied to movie positive counts for popularity-based negatives.",
    )
    parser.add_argument(
        "--mixed-negative-probability",
        type=float,
        default=0.5,
        help="Probability of popularity-based negatives when using mixed sampling.",
    )
    parser.add_argument("--max-rows", type=int, help="Optional row cap for smoke tests.")
    parser.add_argument(
        "--max-positive-samples",
        type=int,
        help="Optional positive interaction cap sampled after loading ratings.",
    )
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument(
        "--quiet-data-loading",
        action="store_true",
        help="Hide chunk-level data loading progress.",
    )
    parser.add_argument("--data-cache-path", type=Path, default=NEURAL_CF_DATA_CACHE_PATH)
    parser.add_argument(
        "--rebuild-data-cache",
        action="store_true",
        help="Rebuild the Neural CF interaction cache before training.",
    )
    parser.add_argument(
        "--no-data-cache",
        action="store_true",
        help="Read processed ratings directly without loading or saving the interaction cache.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=NEURAL_CF_MODEL_PATH)
    args = parser.parse_args()

    train_neural_cf(args)


if __name__ == "__main__":
    main()
