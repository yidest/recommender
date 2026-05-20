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
from src.ranking_metrics import top_k_movie_ids


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


def load_svd_model(model_path: Path, retrain: bool) -> TruncatedSVDModel:
    if model_path.exists() and not retrain:
        return load_model(model_path)
    return build_model()


def load_rated_movie_ids(user_id: int, train_ratings_path: Path) -> set[int]:
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
    return set(train_df.loc[train_df["user_id"] == user_id, "movie_id"])


def load_all_movie_ids(movie_titles_path: Path) -> list[int]:
    movie_titles = load_movie_titles(movie_titles_path)
    movie_titles["movie_id"] = pd.to_numeric(movie_titles["movie_id"], errors="coerce")
    movie_titles = movie_titles.dropna(subset=["movie_id"])
    movie_titles["movie_id"] = movie_titles["movie_id"].astype(int)
    return sorted(set(movie_titles["movie_id"]))


def svd_recall_candidates(
    user_id: int,
    svd_model: TruncatedSVDModel,
    candidate_count: int,
) -> list[int]:
    rated_movie_ids = load_rated_movie_ids(user_id, TRAIN_RATINGS_PATH)
    all_movie_ids = load_all_movie_ids(MOVIE_TITLES_PATH)
    candidate_movie_ids = [movie_id for movie_id in all_movie_ids if movie_id not in rated_movie_ids]
    return top_k_movie_ids(
        model=svd_model,
        user_id=user_id,
        candidate_movie_ids=candidate_movie_ids,
        top_k=candidate_count,
    )


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
    parser.add_argument(
        "--candidate-source",
        choices=["svd-recall", "full"],
        default="svd-recall",
        help="Candidate source for Neural CF reranking. Defaults to SVD recall.",
    )
    parser.add_argument(
        "--candidate-count",
        type=int,
        default=1000,
        help="Number of SVD candidates to recall before Neural CF reranking.",
    )
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
        candidate_movie_ids = None
        if args.candidate_source == "svd-recall":
            svd_model = load_svd_model(args.svd_model_path, args.retrain)
            candidate_movie_ids = svd_recall_candidates(
                user_id=args.user_id,
                svd_model=svd_model,
                candidate_count=args.candidate_count,
            )
        recommendations = recommend_with_neural_cf(
            user_id=args.user_id,
            model_path=args.model_path,
            top_n=args.top_n,
            batch_size=args.batch_size,
            device=torch.device(args.device),
            candidate_movie_ids=candidate_movie_ids,
        )
        score_column = "neural_cf_score"
        model_label = "Neural CF"
        candidate_label = (
            f"SVD recall candidates: {len(candidate_movie_ids)}"
            if candidate_movie_ids is not None
            else "Candidate source: full catalog"
        )
    else:
        model = load_svd_model(args.svd_model_path, args.retrain)

        recommendations = recommend_with_svd(
            user_id=args.user_id,
            model=model,
            train_ratings_path=TRAIN_RATINGS_PATH,
            movie_titles_path=MOVIE_TITLES_PATH,
            top_n=args.top_n,
        )
        score_column = "predicted_rating"
        model_label = "TruncatedSVD"
        candidate_label = None

    print(f"Top {args.top_n} {model_label} recommendations for user {args.user_id}:")
    if candidate_label is not None:
        print(candidate_label)
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
