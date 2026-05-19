"""Rating prediction models."""

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD


class GlobalMeanModel:
    """Predict every rating with the global training-set mean."""

    def __init__(self) -> None:
        self.global_mean = 0.0

    def fit_from_csv(self, train_csv_path: Path) -> None:
        total_sum = 0.0
        total_count = 0

        with train_csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    rating = float(row["rating"])
                except (KeyError, ValueError):
                    continue
                total_sum += rating
                total_count += 1

        if total_count == 0:
            raise RuntimeError("Training data is empty or invalid.")
        self.global_mean = total_sum / total_count

    def predict(self, movie_id: int, user_id: int) -> float:
        return float(np.clip(self.global_mean, 1.0, 5.0))


class BiasModel:
    """Global mean plus regularized movie and user bias."""

    def __init__(self, lambda_movie: float = 25.0, lambda_user: float = 10.0):
        self.global_mean = 0.0
        self.movie_bias: dict[int, float] = {}
        self.user_bias: dict[int, float] = {}
        self.lambda_movie = lambda_movie
        self.lambda_user = lambda_user

    def fit_from_csv(self, train_csv_path: Path) -> None:
        total_sum = 0.0
        total_count = 0
        movie_sum: defaultdict[int, float] = defaultdict(float)
        movie_count: defaultdict[int, int] = defaultdict(int)

        with train_csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    movie_id = int(row["movie_id"])
                    rating = float(row["rating"])
                except (KeyError, ValueError):
                    continue
                total_sum += rating
                total_count += 1
                movie_sum[movie_id] += rating
                movie_count[movie_id] += 1

        if total_count == 0:
            raise RuntimeError("Training data is empty or invalid.")
        self.global_mean = total_sum / total_count

        for movie_id, rating_sum in movie_sum.items():
            count = movie_count[movie_id]
            avg_movie_rating = rating_sum / count
            raw_deviation = avg_movie_rating - self.global_mean
            shrink = count / (count + self.lambda_movie)
            self.movie_bias[movie_id] = shrink * raw_deviation

        user_sum: defaultdict[int, float] = defaultdict(float)
        user_count: defaultdict[int, int] = defaultdict(int)
        with train_csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    movie_id = int(row["movie_id"])
                    user_id = int(row["user_id"])
                    rating = float(row["rating"])
                except (KeyError, ValueError):
                    continue

                movie_bias = self.movie_bias.get(movie_id, 0.0)
                residual = rating - self.global_mean - movie_bias
                user_sum[user_id] += residual
                user_count[user_id] += 1

        for user_id, residual_sum in user_sum.items():
            count = user_count[user_id]
            raw_bias = residual_sum / count
            shrink = count / (count + self.lambda_user)
            self.user_bias[user_id] = shrink * raw_bias

    def predict(self, movie_id: int, user_id: int) -> float:
        movie_bias = self.movie_bias.get(movie_id, 0.0)
        user_bias = self.user_bias.get(user_id, 0.0)
        prediction = self.global_mean + movie_bias + user_bias
        return float(np.clip(prediction, 1.0, 5.0))


@dataclass
class TruncatedSVDModel:
    """Bias model plus TruncatedSVD factors learned from residual ratings."""

    bias_model: BiasModel
    n_components: int = 20
    random_state: int = 42
    svd: TruncatedSVD | None = None
    user_factors: np.ndarray | None = None
    movie_factors: np.ndarray | None = None
    user_to_idx: dict[int, int] | None = None
    movie_to_idx: dict[int, int] | None = None

    def fit_from_csv(self, train_csv_path: Path) -> None:
        self.bias_model.fit_from_csv(train_csv_path)

        train_df = pd.read_csv(train_csv_path)
        train_df["user_id"] = pd.to_numeric(train_df["user_id"], errors="coerce")
        train_df["movie_id"] = pd.to_numeric(train_df["movie_id"], errors="coerce")
        train_df["rating"] = pd.to_numeric(train_df["rating"], errors="coerce")
        train_df = train_df.dropna(subset=["user_id", "movie_id", "rating"])
        train_df["user_id"] = train_df["user_id"].astype(int)
        train_df["movie_id"] = train_df["movie_id"].astype(int)

        users = train_df["user_id"].unique()
        movies = train_df["movie_id"].unique()
        self.user_to_idx = {user_id: i for i, user_id in enumerate(users)}
        self.movie_to_idx = {movie_id: j for j, movie_id in enumerate(movies)}

        user_indices = train_df["user_id"].map(self.user_to_idx).to_numpy()
        movie_indices = train_df["movie_id"].map(self.movie_to_idx).to_numpy()

        base_prediction = np.array(
            [
                self.bias_model.predict(movie_id, user_id)
                for user_id, movie_id in zip(train_df["user_id"], train_df["movie_id"])
            ]
        )
        residual = train_df["rating"].to_numpy() - base_prediction
        residual_matrix = coo_matrix(
            (residual, (user_indices, movie_indices)),
            shape=(len(users), len(movies)),
        ).tocsr()

        self.svd = TruncatedSVD(
            n_components=self.n_components,
            random_state=self.random_state,
        )
        self.user_factors = self.svd.fit_transform(residual_matrix)
        self.movie_factors = self.svd.components_.T

    def predict(self, movie_id: int, user_id: int) -> float:
        prediction = self.bias_model.predict(movie_id, user_id)

        if (
            self.user_to_idx is None
            or self.movie_to_idx is None
            or self.user_factors is None
            or self.movie_factors is None
        ):
            return float(np.clip(prediction, 1.0, 5.0))

        if user_id not in self.user_to_idx or movie_id not in self.movie_to_idx:
            return float(np.clip(prediction, 1.0, 5.0))

        user_index = self.user_to_idx[user_id]
        movie_index = self.movie_to_idx[movie_id]
        prediction += float(self.user_factors[user_index].dot(self.movie_factors[movie_index]))
        return float(np.clip(prediction, 1.0, 5.0))


def evaluate_rmse(
    model: TruncatedSVDModel | BiasModel | GlobalMeanModel,
    test_csv_path: Path,
) -> float:
    """Evaluate a model with RMSE on a test ratings CSV."""
    squared_error_sum = 0.0
    count = 0

    with test_csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                movie_id = int(row["movie_id"])
                user_id = int(row["user_id"])
                rating = float(row["rating"])
            except (KeyError, ValueError):
                continue

            prediction = model.predict(movie_id, user_id)
            squared_error_sum += (prediction - rating) ** 2
            count += 1

    if count == 0:
        raise RuntimeError("Test data is empty or invalid.")
    return math.sqrt(squared_error_sum / count)
