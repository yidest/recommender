"""Shared project paths and constants."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

RAW_RATINGS_PATH = RAW_DATA_DIR / "data.txt"
MOVIE_TITLES_PATH = RAW_DATA_DIR / "movieTitles.csv"

TRAIN_RATINGS_PATH = PROCESSED_DATA_DIR / "train_rating.csv"
TEST_RATINGS_PATH = PROCESSED_DATA_DIR / "test_rating.csv"
NEURAL_CF_DATA_CACHE_PATH = PROCESSED_DATA_DIR / "neural_cf_interactions.pkl"
TRAINED_MODEL_PATH = MODEL_DIR / "truncated_svd_model.pkl"
NEURAL_CF_MODEL_PATH = MODEL_DIR / "neural_cf_model.pt"
DEFAULT_EXPERIMENT_RESULTS_PATH = REPORTS_DIR / "svd_experiments.csv"

RANDOM_SEED = 12554347

DEFAULT_N_COMPONENTS = 50
DEFAULT_TOP_N = 10
