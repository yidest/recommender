# Technical Standards

## Python

- Target macOS development with a standard Python virtual environment.
- Use relative project paths based on the repository root.
- Avoid hard-coded absolute paths.
- Keep functions focused and named by behavior.
- Use type hints where they improve clarity.

## Project Layout

Planned structure:

```text
.
├── data/
│   ├── raw/
│   │   ├── data.txt
│   │   └── movieTitles.csv
│   └── processed/
│       ├── train_rating.csv
│       └── test_rating.csv
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_processing.py
│   ├── models.py
│   ├── evaluate.py
│   └── recommend.py
├── train.py
├── requirements.txt
├── README.md
├── docs/
└── dev_logs/
```

## Path Rules

- `src/config.py` owns reusable paths and constants.
- Runtime code imports paths from `src.config`.
- No local Desktop paths should appear in project code.
- Data directories should be created when needed by processing scripts.

## Data and Git

- `data/raw/` should be ignored by Git.
- `data/processed/` should be ignored by Git.
- README should explain where to place raw data files.
- Generated caches and virtual environments should be ignored by Git.

## Model Naming

- Do not call the matrix factorization model `PCA`.
- Use `TruncatedSVDModel` or similar naming.
- Keep `BiasModel` as the baseline model name unless a clearer name is needed later.

## Verification Standard

Each implementation step should be verified with the smallest useful command, such as:

```bash
python train.py
```

For recommendation behavior:

```bash
python -m src.recommend --user-id USER_ID --top-n 10
```

