# Project Requirements

## Goal

Extend the existing movie rating prediction project into a movie recommendation system that can run on macOS, use relative paths, and be prepared for a clean GitHub upload.

## Confirmed Scope

- Keep the existing rating prediction logic, but rename the PCA-based model concept to `TruncatedSVD`.
- Support recommendations for existing users only in the first version.
- Leave cold-start recommendations for a later phase.
- Use English for the public-facing README, with optional short Chinese notes.
- Do not commit raw or processed data files to GitHub.

## Data Assumptions

Raw data files are expected at:

- `data/raw/data.txt`
- `data/raw/movieTitles.csv`

Processed files should be generated at:

- `data/processed/train_rating.csv`
- `data/processed/test_rating.csv`

## Functional Requirements

- Parse the raw rating dataset into train and test CSV files.
- Train a baseline bias model and a TruncatedSVD residual model.
- Evaluate prediction quality with RMSE.
- Generate Top-N movie recommendations for a user already present in the training data.
- Display recommendation output with movie IDs, movie titles, and predicted ratings.

## Non-Functional Requirements

- Use `pathlib.Path` for all project paths.
- Centralize paths and key constants in `src/config.py`.
- Keep code modular and readable for a beginner-friendly machine learning project.
- Prefer simple command-line workflows before adding extra tools or interfaces.
- Keep implementation changes small enough to test after each step.

