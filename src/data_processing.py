"""Data loading and preprocessing utilities."""

import csv
import random
from pathlib import Path

import pandas as pd


def parse_and_split_dataset(
    input_path: Path,
    train_out_path: Path,
    test_out_path: Path,
    seed: int,
) -> None:
    """Parse the raw ratings file and split one rating per movie into test."""
    rng = random.Random(seed)
    train_out_path.parent.mkdir(parents=True, exist_ok=True)
    test_out_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8", errors="ignore") as fin, (
        train_out_path.open("w", newline="", encoding="utf-8")
    ) as ftrain, test_out_path.open("w", newline="", encoding="utf-8") as ftest:
        train_writer = csv.writer(ftrain)
        test_writer = csv.writer(ftest)
        header = ["movie_id", "user_id", "rating", "date"]
        train_writer.writerow(header)
        test_writer.writerow(header)
        current_movie_id = None
        current_ratings: list[tuple[int, int, str]] = []

        def flush_current_movie() -> None:
            nonlocal current_movie_id
            if current_movie_id is None or not current_ratings:
                return

            test_idx = rng.randrange(len(current_ratings))
            for i, (user_id, rating, date_str) in enumerate(current_ratings):
                row = [current_movie_id, user_id, rating, date_str]
                if i == test_idx:
                    test_writer.writerow(row)
                else:
                    train_writer.writerow(row)
            current_ratings.clear()

        for line in fin:
            line = line.strip()
            if not line:
                continue

            if line.endswith(":"):
                flush_current_movie()
                movie_id_str = line[:-1]
                try:
                    current_movie_id = int(movie_id_str)
                except ValueError:
                    current_movie_id = None
                continue

            parts = line.split(",")
            if len(parts) != 3:
                continue

            user_str, rating_str, date_str = parts
            try:
                user_id = int(user_str)
                rating = int(rating_str)
            except ValueError:
                continue

            if current_movie_id is not None:
                current_ratings.append((user_id, rating, date_str))

        flush_current_movie()


def load_movie_titles(movie_titles_path: Path) -> pd.DataFrame:
    """Load movie titles with stable column names."""
    records = []
    with movie_titles_path.open("r", encoding="latin1", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            try:
                movie_id = int(row[0])
            except ValueError:
                continue

            year = row[1] or None
            title = ",".join(row[2:]).strip().strip(",").strip()
            records.append({"movie_id": movie_id, "year": year, "title": title})

    return pd.DataFrame(records, columns=["movie_id", "year", "title"])
