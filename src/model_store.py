"""Save and load trained recommendation models."""

import pickle
from pathlib import Path

from src.models import TruncatedSVDModel


def save_model(model: TruncatedSVDModel, model_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as f:
        pickle.dump(model, f)


def load_model(model_path: Path) -> TruncatedSVDModel:
    with model_path.open("rb") as f:
        model = pickle.load(f)

    if not isinstance(model, TruncatedSVDModel):
        raise TypeError(f"Stored object is not a TruncatedSVDModel: {model_path}")
    return model
