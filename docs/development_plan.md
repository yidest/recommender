# Development Plan

## Phase 1: Project Foundation

- Create the project folder structure.
- Add `src/config.py` with relative paths and constants.
- Add `.gitignore`.
- Add `requirements.txt`.
- Move the existing code into modular files without changing behavior.

Exit criteria:

- The project imports correctly.
- No absolute Desktop paths remain in source code.

## Phase 2: Data Processing and Training

- Move raw dataset parsing into `src/data_processing.py`.
- Move model classes into `src/models.py`.
- Move RMSE evaluation into `src/evaluate.py`.
- Update `train.py` to run parsing, training, and evaluation.

Exit criteria:

- `python train.py` creates processed CSV files and prints RMSE.

## Phase 3: Recommendation Feature

- Load movie titles from `data/raw/movieTitles.csv`.
- Add recommendation logic for existing users.
- Exclude movies the user already rated in the training data.
- Return Top-N predicted movies.

Exit criteria:

- A command can produce readable Top-N recommendations for an existing user.

## Phase 4: Documentation and GitHub Readiness

- Write `README.md` in English.
- Explain setup, data placement, training, evaluation, and recommendation commands.
- Keep data out of Git with `.gitignore`.
- Add a short project summary suitable for GitHub.

Exit criteria:

- A new user can clone the repository, place data files, install dependencies, and run the project.

## Near-Term Next Step

Start with Phase 1 and Phase 2 only. Do not add cold-start logic, web apps, notebooks, or advanced evaluation until the basic command-line project is stable.

