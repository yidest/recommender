# Codex Workflow Notes

This file is for Codex working sessions. Use it to stay aligned with the project standards and avoid re-discovering decisions.

## Standard Documents

- Requirements: `docs/requirements.md`
- Technical standards: `docs/technical_standards.md`
- Development plan: `docs/development_plan.md`
- Development workflow and logs: `docs/development_workflow.md`
- Design standards: `docs/design_standards.md`
- Daily logs: `dev_logs/`

## Current Project Direction

The project extends an existing movie rating prediction assignment into a GitHub-ready movie recommendation system.

Key constraints:

- Use `pathlib.Path`.
- Put shared paths and constants in `src/config.py`.
- Assume raw files are located at `data/raw/data.txt` and `data/raw/movieTitles.csv`.
- Save processed CSVs to `data/processed/`.
- Rename PCA-related model naming to TruncatedSVD.
- First recommendation version supports existing users only.
- Do not commit raw or processed data.

## Working Instructions

Before code changes:

1. Read the relevant standard document in `docs/`.
2. Update today's log in `dev_logs/YYYY-MM-DD.md`.
3. Make a small, testable change.
4. Record verification commands and results in the daily log.

Implementation order:

1. Project structure and config.
2. Behavior-preserving refactor from the old script.
3. Training command.
4. Existing-user Top-N recommendation command.
5. README and GitHub cleanup.

Avoid adding cold-start logic, web UI, notebooks, or extra model families until the basic command-line workflow is stable.

