# Design Standards

## User Experience

This is a command-line machine learning project. The first usable experience should be simple terminal commands, not a web interface.

## Command Design

Prefer commands that are easy to remember:

```bash
python train.py
python -m src.recommend --user-id USER_ID --top-n 10
```

## Output Design

Training output should show:

- Whether data processing ran
- Number of records if practical
- Model name
- RMSE

Recommendation output should show:

- User ID
- Rank
- Movie ID
- Movie title
- Predicted rating

## Documentation Style

- Public README should be in English.
- Keep explanations beginner-friendly.
- Explain the difference between rating prediction and recommendation.
- Mention that raw data is not included in the repository.

