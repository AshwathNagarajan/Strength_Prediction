"""Data loading and validation for capacity/deflection prediction."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

from .performance_config import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGETS


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


ALIASES = {
    "Recycled Aggregate (%)": ["Recycled Aggregate (%)", "Recycled Agg. (%)", "Recycled Agg (%)"],
    "Target: Deflection δ (mm)": ["Target: Deflection δ (mm)", "Target: Deflection Î´ (mm)", "Target Deflection (mm)"],
}


def read_dataset(path: Path) -> pd.DataFrame:
    """Read the project dataset from Excel or CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported dataset format: {path.suffix}")


def resolve_column(df: pd.DataFrame, canonical: str) -> str:
    """Resolve a canonical column name with a small alias map."""
    normalized = {_normalize(column): column for column in df.columns}
    candidates = [canonical] + ALIASES.get(canonical, [])
    for candidate in candidates:
        actual = normalized.get(_normalize(candidate))
        if actual:
            return actual
    raise ValueError(f"Required column not found: {canonical}. Available columns: {', '.join(df.columns)}")


def load_performance_dataset(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Load and canonicalize required feature and target columns."""
    raw = read_dataset(path)
    required = FEATURES + [target["column"] for target in TARGETS.values()]
    mapping = {canonical: resolve_column(raw, canonical) for canonical in required}
    data = raw[[mapping[column] for column in required]].rename(
        columns={actual: canonical for canonical, actual in mapping.items()}
    )
    return data, mapping


def validate_performance_dataset(data: pd.DataFrame, mapping: dict[str, str]) -> dict[str, Any]:
    """Collect validation diagnostics for the structural-performance dataset."""
    report = {
        "rows": int(data.shape[0]),
        "columns": int(data.shape[1]),
        "selected_columns": list(data.columns),
        "source_column_mapping": mapping,
        "dtypes": {column: str(dtype) for column, dtype in data.dtypes.items()},
        "missing_values": data.isna().sum().astype(int).to_dict(),
        "duplicate_records": int(data.duplicated().sum()),
        "numeric_statistics": data[NUMERIC_FEATURES + [target["column"] for target in TARGETS.values()]]
        .describe()
        .T.to_dict(orient="index"),
        "categorical_values": {
            feature: sorted(data[feature].dropna().astype(str).unique().tolist()) for feature in CATEGORICAL_FEATURES
        },
    }
    logging.info("Performance dataset rows: %s", report["rows"])
    logging.info("Performance selected columns: %s", ", ".join(data.columns))
    logging.info("Performance missing values: %s", report["missing_values"])
    logging.info("Performance duplicate records: %s", report["duplicate_records"])
    return report


def clean_performance_data(data: pd.DataFrame) -> pd.DataFrame:
    """Parse selected features/targets and remove invalid rows."""
    cleaned = data.copy()
    target_columns = [target["column"] for target in TARGETS.values()]
    for column in NUMERIC_FEATURES + target_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    for column in CATEGORICAL_FEATURES:
        cleaned[column] = cleaned[column].astype(str).str.strip()
        cleaned.loc[cleaned[column].isin(["", "nan", "None"]), column] = pd.NA

    missing = cleaned[FEATURES + target_columns].isna().sum()
    if missing.any():
        logging.warning("Dropping rows with missing selected values: %s", missing[missing > 0].to_dict())
        cleaned = cleaned.dropna(subset=FEATURES + target_columns).copy()

    duplicates = int(cleaned.duplicated(subset=FEATURES + target_columns).sum())
    if duplicates:
        logging.info("Removing %s duplicate selected records.", duplicates)
        cleaned = cleaned.drop_duplicates(subset=FEATURES + target_columns).copy()
    return cleaned


def feature_profile(data: pd.DataFrame) -> dict[str, Any]:
    """Store numeric ranges and categorical domains from the training data."""
    return {
        "numeric_ranges": {
            feature: {"min": float(data[feature].min()), "max": float(data[feature].max())}
            for feature in NUMERIC_FEATURES
        },
        "categorical_values": {
            feature: sorted(data[feature].dropna().astype(str).unique().tolist()) for feature in CATEGORICAL_FEATURES
        },
    }

