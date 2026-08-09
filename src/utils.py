"""Shared utility helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np


def setup_logging() -> None:
    """Configure concise console logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def ensure_directories(paths: list[Path]) -> None:
    """Create project directories if they do not exist."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def to_builtin(value: Any) -> Any:
    """Convert numpy/pandas scalar objects into JSON-serializable values."""
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return [to_builtin(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    return value


def save_json(data: dict[str, Any], path: Path) -> None:
    """Save a dictionary as indented JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(to_builtin(data), file, indent=2)


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file with a helpful error."""
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def format_model_reason(metadata: dict[str, Any]) -> str:
    """Generate a deterministic best-model selection sentence."""
    best_name = metadata["best_model"]
    other_name = "CatBoost" if best_name == "XGBoost" else "XGBoost"
    best = metadata[best_name.lower()]
    other = metadata[other_name.lower()]
    if best["r2"] > other["r2"]:
        return f"{best_name} was selected because it achieved the highest test R²."
    if best["rmse"] < other["rmse"]:
        return f"{best_name} was selected because test R² was tied and it achieved lower RMSE."
    return f"{best_name} was selected because test R² and RMSE were tied and it achieved lower MAE."

