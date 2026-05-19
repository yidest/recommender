"""Train and evaluate the movie recommendation model."""

import argparse

from src.config import (
    DEFAULT_N_COMPONENTS,
    RANDOM_SEED,
    RAW_RATINGS_PATH,
    TEST_RATINGS_PATH,
    TRAIN_RATINGS_PATH,
    TRAINED_MODEL_PATH,
)
from src.data_processing import parse_and_split_dataset
from src.model_store import save_model
from src.models import BiasModel, GlobalMeanModel, TruncatedSVDModel, evaluate_rmse


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the recommendation model.")
    parser.add_argument(
        "--force-process",
        action="store_true",
        help="Rebuild train/test CSV files even when processed files already exist.",
    )
    args = parser.parse_args()

    if not RAW_RATINGS_PATH.exists():
        raise FileNotFoundError(
            f"Raw ratings file not found: {RAW_RATINGS_PATH}\n"
            "Place data.txt at data/raw/data.txt before running training."
        )

    processed_files_exist = TRAIN_RATINGS_PATH.exists() and TEST_RATINGS_PATH.exists()
    if args.force_process or not processed_files_exist:
        print("Preparing train/test rating CSV files...")
        parse_and_split_dataset(
            input_path=RAW_RATINGS_PATH,
            train_out_path=TRAIN_RATINGS_PATH,
            test_out_path=TEST_RATINGS_PATH,
            seed=RANDOM_SEED,
        )
    else:
        print("Using existing processed train/test CSV files.")

    print("Training baseline models...")
    global_mean_model = GlobalMeanModel()
    global_mean_model.fit_from_csv(TRAIN_RATINGS_PATH)

    bias_model = BiasModel(lambda_movie=25.0, lambda_user=10.0)
    bias_model.fit_from_csv(TRAIN_RATINGS_PATH)

    global_mean_rmse = evaluate_rmse(global_mean_model, TEST_RATINGS_PATH)
    bias_rmse = evaluate_rmse(bias_model, TEST_RATINGS_PATH)

    print("Training TruncatedSVD recommendation model...")
    model = TruncatedSVDModel(
        bias_model=BiasModel(lambda_movie=25.0, lambda_user=10.0),
        n_components=DEFAULT_N_COMPONENTS,
        random_state=RANDOM_SEED,
    )
    model.fit_from_csv(TRAIN_RATINGS_PATH)

    rmse = evaluate_rmse(model, TEST_RATINGS_PATH)
    print("Model comparison:")
    print(f"  GlobalMeanModel:   RMSE {global_mean_rmse:.4f}")
    print(f"  BiasModel:         RMSE {bias_rmse:.4f}")
    print(f"  TruncatedSVDModel: RMSE {rmse:.4f}")

    save_model(model, TRAINED_MODEL_PATH)
    print(f"Saved trained model to: {TRAINED_MODEL_PATH}")


if __name__ == "__main__":
    main()
