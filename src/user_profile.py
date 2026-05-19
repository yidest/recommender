"""Inspect an existing user's rating history."""

import argparse

import pandas as pd

from src.config import DEFAULT_TOP_N, MOVIE_TITLES_PATH, TRAIN_RATINGS_PATH
from src.data_processing import load_movie_titles


def get_user_profile(
    user_id: int,
    top_n: int = DEFAULT_TOP_N,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    ratings = pd.read_csv(TRAIN_RATINGS_PATH)
    ratings["movie_id"] = pd.to_numeric(ratings["movie_id"], errors="coerce")
    ratings["user_id"] = pd.to_numeric(ratings["user_id"], errors="coerce")
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    ratings = ratings.dropna(subset=["movie_id", "user_id", "rating"])
    ratings["movie_id"] = ratings["movie_id"].astype(int)
    ratings["user_id"] = ratings["user_id"].astype(int)

    user_ratings = ratings.loc[ratings["user_id"] == user_id].copy()
    if user_ratings.empty:
        raise ValueError(
            f"User {user_id} was not found in training data. "
            "Cold-start users are not supported yet."
        )

    movie_titles = load_movie_titles(MOVIE_TITLES_PATH)
    profile = user_ratings.merge(movie_titles, on="movie_id", how="left")
    profile["title"] = profile["title"].fillna("(unknown title)")
    profile = profile.sort_values(["rating", "movie_id"], ascending=[False, True])

    summary = {
        "rating_count": int(len(user_ratings)),
        "average_rating": float(user_ratings["rating"].mean()),
        "min_rating": float(user_ratings["rating"].min()),
        "max_rating": float(user_ratings["rating"].max()),
    }
    return profile.head(top_n), summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Show an existing user's rating profile.")
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    args = parser.parse_args()

    profile, summary = get_user_profile(user_id=args.user_id, top_n=args.top_n)
    print(f"User {args.user_id} profile:")
    print(f"  Rated movies: {summary['rating_count']}")
    print(f"  Average rating: {summary['average_rating']:.3f}")
    print(f"  Rating range: {summary['min_rating']:.0f}-{summary['max_rating']:.0f}")
    print(f"Top {len(profile)} highest-rated movies:")

    for rank, row in enumerate(profile.itertuples(index=False), start=1):
        print(f"{rank:>2}. rating={row.rating:.0f} | movie_id={row.movie_id} | {row.title}")


if __name__ == "__main__":
    main()
