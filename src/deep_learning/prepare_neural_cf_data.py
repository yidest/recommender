"""Prepare and cache Neural CF interaction data."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import NEURAL_CF_DATA_CACHE_PATH, RANDOM_SEED, TRAIN_RATINGS_PATH
from src.deep_learning.dataset import (
    build_cache_metadata,
    cache_metadata_matches,
    load_interaction_data,
    load_interaction_data_cache,
    save_interaction_data_cache,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare cached Neural CF interaction data.")
    parser.add_argument("--positive-threshold", type=float, default=4.0)
    parser.add_argument("--max-rows", type=int, help="Optional row cap for smoke-test caches.")
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--output", type=Path, default=NEURAL_CF_DATA_CACHE_PATH)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the cache even when a matching cache already exists.",
    )
    args = parser.parse_args()

    if args.max_rows is not None and args.output == NEURAL_CF_DATA_CACHE_PATH:
        raise ValueError(
            "--max-rows builds a partial cache. Pass --output to avoid overwriting "
            f"the default full-data cache at {NEURAL_CF_DATA_CACHE_PATH}."
        )

    expected_metadata = build_cache_metadata(
        ratings_path=TRAIN_RATINGS_PATH,
        positive_threshold=args.positive_threshold,
        max_rows=args.max_rows,
    )

    if args.output.exists() and not args.force:
        _, cached_metadata = load_interaction_data_cache(args.output)
        if cache_metadata_matches(cached_metadata, expected_metadata):
            print(f"Neural CF interaction cache is already current: {args.output}")
            return
        print(f"Existing Neural CF interaction cache is stale: {args.output}")

    interaction_data = load_interaction_data(
        ratings_path=TRAIN_RATINGS_PATH,
        positive_threshold=args.positive_threshold,
        max_rows=args.max_rows,
        max_positive_samples=None,
        seed=args.seed,
        chunk_size=args.chunk_size,
        show_progress=True,
    )
    save_interaction_data_cache(interaction_data, args.output, expected_metadata)
    print(f"Saved Neural CF interaction cache to: {args.output}")


if __name__ == "__main__":
    main()
