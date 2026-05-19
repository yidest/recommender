"""Generate Top-N recommendations for existing users."""

import argparse
from pathlib import Path

import pandas as pd
import torch

from src.config import (
    DEFAULT_N_COMPONENTS,
    DEFAULT_TOP_N,
    MOVIE_TITLES_PATH,
    NEURAL_CF_MODEL_PATH,
    RANDOM_SEED,
    TRAIN_RATINGS_PATH,
    TRAINED_MODEL_PATH,
)
from src.data_processing import load_movie_titles
from src.deep_learning.recommend_neural_cf import recommend_for_user as recommend_with_neural_cf
from src.model_store import load_model
from src.models import BiasModel, TruncatedSVDModel


def build_model(train_ratings_path: Path = TRAIN_RATINGS_PATH) -> TruncatedSVDModel:
    model = TruncatedSVDModel(
        bias_model=BiasModel(lambda_movie=25.0, lambda_user=10.0),
        n_components=DEFAULT_N_COMPONENTS,
        random_state=RANDOM_SEED,
    )
    model.fit_from_csv(train_ratings_path)
    return model


def recommend_with_svd(
    user_id: int,
    model: TruncatedSVDModel,
    train_ratings_path: Path,
    movie_titles_path: Path,
    top_n: int = DEFAULT_TOP_N,
) -> pd.DataFrame:
    train_df = pd.read_csv(train_ratings_path, usecols=["movie_id", "user_id"])
    train_df["movie_id"] = pd.to_numeric(train_df["movie_id"], errors="coerce")
    train_df["user_id"] = pd.to_numeric(train_df["user_id"], errors="coerce")
    train_df = train_df.dropna(subset=["movie_id", "user_id"])
    train_df["movie_id"] = train_df["movie_id"].astype(int)
    train_df["user_id"] = train_df["user_id"].astype(int)

    if user_id not in set(train_df["user_id"]):
        raise ValueError(
            f"User {user_id} was not found in training data. "
            "Cold-start recommendations are not supported yet."
        )

    rated_movies = set(train_df.loc[train_df["user_id"] == user_id, "movie_id"])
    movie_titles = load_movie_titles(movie_titles_path)
    movie_titles["movie_id"] = pd.to_numeric(movie_titles["movie_id"], errors="coerce")
    movie_titles = movie_titles.dropna(subset=["movie_id"])
    movie_titles["movie_id"] = movie_titles["movie_id"].astype(int)

    candidates = movie_titles.loc[~movie_titles["movie_id"].isin(rated_movies)].copy()
    candidates["predicted_rating"] = candidates["movie_id"].apply(
        lambda movie_id: model.predict(int(movie_id), user_id)
    )

    recommendations = candidates.sort_values(
        ["predicted_rating", "movie_id"],
        ascending=[False, True],
    ).head(top_n)

    return recommendations[["movie_id", "title", "predicted_rating"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend movies for an existing user.")
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument(
        "--model",
        choices=["neural-cf", "svd"],
        default="neural-cf",
        help="Recommendation model to use. Defaults to the Neural CF main model.",
    )
    parser.add_argument("--model-path", type=Path, default=NEURAL_CF_MODEL_PATH)
    parser.add_argument("--svd-model-path", type=Path, default=TRAINED_MODEL_PATH)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional CSV path for saving recommendations.",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Retrain the SVD baseline instead of loading its saved model artifact.",
    )
    args = parser.parse_args()

    if args.model == "neural-cf":
        if not args.model_path.exists():
            raise FileNotFoundError(
                f"Neural CF checkpoint not found: {args.model_path}\n"
                "Run `python -m src.deep_learning.train_neural_cf` first."
            )
        recommendations = recommend_with_neural_cf(
            user_id=args.user_id,
            model_path=args.model_path,
            top_n=args.top_n,
            batch_size=args.batch_size,
            device=torch.device(args.device),
        )
        score_column = "neural_cf_score"
        model_label = "Neural CF"
    else:
        if args.svd_model_path.exists() and not args.retrain:
            model = load_model(args.svd_model_path)
        else:
            model = build_model()

        recommendations = recommend_with_svd(
            user_id=args.user_id,
            model=model,
            train_ratings_path=TRAIN_RATINGS_PATH,
            movie_titles_path=MOVIE_TITLES_PATH,
            top_n=args.top_n,
        )
        score_column = "predicted_rating"
        model_label = "TruncatedSVD"

    print(f"Top {args.top_n} {model_label} recommendations for user {args.user_id}:")
    output_df = recommendations.copy()
    output_df.insert(0, "rank", range(1, len(output_df) + 1))

    for row in output_df.itertuples(index=False):
        score = getattr(row, score_column)
        print(
            f"{row.rank:>2}. movie_id={row.movie_id} | "
            f"{score_column}={score:.4f} | {row.title}"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(args.output, index=False)
        print(f"Saved recommendations to: {args.output}")


if __name__ == "__main__":
    main()
