"""Preprocessing and train/test splitting."""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import FEATURES, RANDOM_STATE, TARGET, TEST_SIZE


def prepare_model_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Validate numeric columns and return model features and target."""
    model_data = data[FEATURES + [TARGET]].copy()
    for column in FEATURES + [TARGET]:
        model_data[column] = pd.to_numeric(model_data[column], errors="coerce")

    missing_before = model_data.isna().sum()
    if missing_before.any():
        logging.warning(
            "Missing/non-numeric values found after parsing. Rows with missing selected values will be removed: %s",
            missing_before[missing_before > 0].to_dict(),
        )
        model_data = model_data.dropna(subset=FEATURES + [TARGET]).copy()

    duplicate_count = int(model_data.duplicated().sum())
    if duplicate_count:
        logging.info("Removing %s duplicate records from selected modeling data.", duplicate_count)
        model_data = model_data.drop_duplicates().copy()

    if model_data.empty:
        raise ValueError("No valid records remain after preprocessing.")

    return model_data[FEATURES], model_data[TARGET]


def split_data(X: pd.DataFrame, y: pd.Series):
    """Create reproducible 80/20 train/test split."""
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def compute_feature_ranges(X_train: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Return training-domain min/max values for each feature."""
    return {
        feature: {"min": float(X_train[feature].min()), "max": float(X_train[feature].max())}
        for feature in FEATURES
    }

