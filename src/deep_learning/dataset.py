"""Dataset utilities for Neural Collaborative Filtering."""

from __future__ import annotations

import bisect
import itertools
import pickle
import random
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset


CACHE_VERSION = 1


@dataclass
class InteractionData:
    positive_pairs: list[tuple[int, int]]
    user_to_idx: dict[int, int]
    movie_to_idx: dict[int, int]
    idx_to_user: dict[int, int]
    idx_to_movie: dict[int, int]
    user_rated_movies: dict[int, set[int]]


def build_movie_sampling_weights(
    positive_pairs: list[tuple[int, int]],
    num_movies: int,
    popularity_power: float,
) -> list[float]:
    counts = [1.0] * num_movies
    for _, movie_idx in positive_pairs:
        counts[movie_idx] += 1.0
    return [count**popularity_power for count in counts]


def build_cache_metadata(
    ratings_path: Path,
    positive_threshold: float,
    max_rows: int | None,
) -> dict[str, int | float | str | None]:
    stat = ratings_path.stat()
    return {
        "cache_version": CACHE_VERSION,
        "ratings_path": str(ratings_path.resolve()),
        "ratings_size": int(stat.st_size),
        "ratings_mtime_ns": int(stat.st_mtime_ns),
        "positive_threshold": float(positive_threshold),
        "max_rows": int(max_rows) if max_rows is not None else None,
    }


def cache_metadata_matches(
    cached_metadata: dict,
    expected_metadata: dict,
) -> bool:
    keys = [
        "cache_version",
        "ratings_path",
        "ratings_size",
        "ratings_mtime_ns",
        "positive_threshold",
        "max_rows",
    ]
    return all(cached_metadata.get(key) == expected_metadata.get(key) for key in keys)


def save_interaction_data_cache(
    interaction_data: InteractionData,
    cache_path: Path,
    metadata: dict,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(
            {
                "metadata": metadata,
                "interaction_data": interaction_data,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def load_interaction_data_cache(cache_path: Path) -> tuple[InteractionData, dict]:
    with cache_path.open("rb") as f:
        payload = pickle.load(f)
    return payload["interaction_data"], payload["metadata"]


def sample_positive_pairs(
    interaction_data: InteractionData,
    max_positive_samples: int | None,
    seed: int,
) -> InteractionData:
    if max_positive_samples is None or len(interaction_data.positive_pairs) <= max_positive_samples:
        return interaction_data

    positive_pairs = random.Random(seed).sample(interaction_data.positive_pairs, max_positive_samples)
    return replace(interaction_data, positive_pairs=positive_pairs)


def load_interaction_data(
    ratings_path: Path,
    positive_threshold: float = 4.0,
    max_rows: int | None = None,
    max_positive_samples: int | None = None,
    seed: int = 42,
    chunk_size: int = 1_000_000,
    show_progress: bool = True,
) -> InteractionData:
    """Load ratings and convert them into indexed positive interactions."""
    user_to_idx: dict[int, int] = {}
    movie_to_idx: dict[int, int] = {}
    idx_to_user: dict[int, int] = {}
    idx_to_movie: dict[int, int] = {}
    user_rated_movies: dict[int, set[int]] = {}
    positive_pairs: list[tuple[int, int]] = []
    total_rows = 0

    if show_progress:
        print(f"Building Neural CF interactions from ratings: {ratings_path}", flush=True)

    for chunk_number, chunk in enumerate(
        _read_rating_chunks(ratings_path, max_rows=max_rows, chunk_size=chunk_size),
        start=1,
    ):
        chunk = _clean_ratings_chunk(chunk)
        total_rows += len(chunk)

        for row in chunk.itertuples(index=False):
            user_id = int(row.user_id)
            movie_id = int(row.movie_id)

            user_idx = user_to_idx.get(user_id)
            if user_idx is None:
                user_idx = len(user_to_idx)
                user_to_idx[user_id] = user_idx
                idx_to_user[user_idx] = user_id
                user_rated_movies[user_idx] = set()

            movie_idx = movie_to_idx.get(movie_id)
            if movie_idx is None:
                movie_idx = len(movie_to_idx)
                movie_to_idx[movie_id] = movie_idx
                idx_to_movie[movie_idx] = movie_id

            user_rated_movies[user_idx].add(movie_idx)
            if float(row.rating) >= positive_threshold:
                positive_pairs.append((user_idx, movie_idx))

        if show_progress:
            print(
                f"  chunk {chunk_number}: rows={total_rows:,}, "
                f"users={len(user_to_idx):,}, movies={len(movie_to_idx):,}, "
                f"positive_pairs={len(positive_pairs):,}",
                flush=True,
            )

    if not user_to_idx or not movie_to_idx:
        raise RuntimeError("No valid ratings found for Neural CF training.")

    if not positive_pairs:
        raise RuntimeError("No positive interactions found for Neural CF training.")

    num_movies = len(movie_to_idx)
    positive_pairs = [
        (user_idx, movie_idx)
        for user_idx, movie_idx in positive_pairs
        if len(user_rated_movies[user_idx]) < num_movies
    ]
    if not positive_pairs:
        raise RuntimeError("No positive interactions have available negative samples.")

    if max_positive_samples is not None and len(positive_pairs) > max_positive_samples:
        if show_progress:
            print(
                f"Sampling {max_positive_samples:,} positives from {len(positive_pairs):,}...",
                flush=True,
            )
        positive_pairs = random.Random(seed).sample(positive_pairs, max_positive_samples)

    if show_progress:
        print(
            f"Loaded interactions: users={len(user_to_idx):,}, movies={len(movie_to_idx):,}, "
            f"positive_pairs={len(positive_pairs):,}",
            flush=True,
        )

    return InteractionData(
        positive_pairs=positive_pairs,
        user_to_idx=user_to_idx,
        movie_to_idx=movie_to_idx,
        idx_to_user=idx_to_user,
        idx_to_movie=idx_to_movie,
        user_rated_movies=user_rated_movies,
    )


def _read_rating_chunks(
    ratings_path: Path,
    max_rows: int | None,
    chunk_size: int,
):
    read_kwargs = {
        "usecols": ["movie_id", "user_id", "rating"],
        "chunksize": chunk_size,
    }
    if max_rows is not None:
        read_kwargs["nrows"] = max_rows
    yield from pd.read_csv(ratings_path, **read_kwargs)


def _clean_ratings_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk = chunk.copy()
    chunk["movie_id"] = pd.to_numeric(chunk["movie_id"], errors="coerce")
    chunk["user_id"] = pd.to_numeric(chunk["user_id"], errors="coerce")
    chunk["rating"] = pd.to_numeric(chunk["rating"], errors="coerce")
    chunk = chunk.dropna(subset=["movie_id", "user_id", "rating"])
    chunk["movie_id"] = chunk["movie_id"].astype(int)
    chunk["user_id"] = chunk["user_id"].astype(int)
    return chunk


def split_positive_pairs(
    positive_pairs: list[tuple[int, int]],
    validation_fraction: float,
    seed: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    pairs = positive_pairs.copy()
    random.Random(seed).shuffle(pairs)
    validation_size = max(1, int(len(pairs) * validation_fraction))
    validation_pairs = pairs[:validation_size]
    train_pairs = pairs[validation_size:]
    if not train_pairs:
        raise RuntimeError("Training split is empty. Use more data or a smaller validation fraction.")
    return train_pairs, validation_pairs


class BPRDataset(Dataset):
    """Return user, positive item, and sampled negative item indices."""

    def __init__(
        self,
        positive_pairs: list[tuple[int, int]],
        user_rated_movies: dict[int, set[int]],
        num_movies: int,
        seed: int,
        negative_sampling_strategy: str = "random",
        movie_sampling_weights: list[float] | None = None,
        mixed_negative_probability: float = 0.5,
    ) -> None:
        self.positive_pairs = positive_pairs
        self.user_rated_movies = user_rated_movies
        self.num_movies = num_movies
        self.rng = random.Random(seed)
        self.negative_sampling_strategy = negative_sampling_strategy
        self.movie_sampling_weights = movie_sampling_weights
        self.cumulative_movie_sampling_weights = (
            list(itertools.accumulate(movie_sampling_weights))
            if movie_sampling_weights is not None
            else None
        )
        self.mixed_negative_probability = mixed_negative_probability

        if self.negative_sampling_strategy not in {"random", "popularity", "mixed"}:
            raise ValueError(
                "negative_sampling_strategy must be one of: random, popularity, mixed"
            )
        if self.negative_sampling_strategy in {"popularity", "mixed"} and self.movie_sampling_weights is None:
            raise ValueError("movie_sampling_weights is required for popularity-based negative sampling.")

    def __len__(self) -> int:
        return len(self.positive_pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        user_idx, positive_movie_idx = self.positive_pairs[index]
        negative_movie_idx = self._sample_negative(user_idx)
        return (
            torch.tensor(user_idx, dtype=torch.long),
            torch.tensor(positive_movie_idx, dtype=torch.long),
            torch.tensor(negative_movie_idx, dtype=torch.long),
        )

    def _sample_negative(self, user_idx: int) -> int:
        rated_movies = self.user_rated_movies[user_idx]
        if len(rated_movies) >= self.num_movies:
            raise RuntimeError(f"User index {user_idx} has no available negative movies.")

        if self.negative_sampling_strategy == "popularity":
            return self._sample_popular_negative(rated_movies)
        if (
            self.negative_sampling_strategy == "mixed"
            and self.rng.random() < self.mixed_negative_probability
        ):
            return self._sample_popular_negative(rated_movies)

        while True:
            movie_idx = self.rng.randrange(self.num_movies)
            if movie_idx not in rated_movies:
                return movie_idx

    def _sample_popular_negative(self, rated_movies: set[int]) -> int:
        if self.cumulative_movie_sampling_weights is None:
            raise RuntimeError("Missing movie sampling weights.")

        total_weight = self.cumulative_movie_sampling_weights[-1]
        while True:
            sample = self.rng.random() * total_weight
            movie_idx = bisect.bisect_left(self.cumulative_movie_sampling_weights, sample)
            if movie_idx not in rated_movies:
                return movie_idx
