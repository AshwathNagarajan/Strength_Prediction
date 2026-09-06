"""Dataset loading, safe column mapping, and validation."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

from .config import FEATURES, TARGET


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


EXPECTED_ALIASES = {
    "w/b Ratio": ["w/b Ratio", "wb ratio", "water binder ratio", "water-to-binder ratio"],
    "Fly Ash (%)": ["Fly Ash (%)", "fly ash", "fly ash percent", "fly ash (%)"],
    "GGBS (%)": ["GGBS (%)", "ggbs", "ggbs percent", "ggbs (%)"],
    "Recycled Aggregate (%)": [
        "Recycled Aggregate (%)",
        "Recycled Agg. (%)",
        "recycled aggregate",
        "recycled agg",
        "recycled aggregate percent",
    ],
    TARGET: [
        TARGET,
        "compressive strength",
        "compressive strength fc",
        "compressive strength fc mpa",
    ],
}


def resolve_columns(df: pd.DataFrame) -> dict[str, str]:
    """Map canonical feature/target names to actual CSV columns."""
    normalized_columns = {_normalize(column): column for column in df.columns}
    mapping: dict[str, str] = {}
    missing: list[str] = []

    for canonical, aliases in EXPECTED_ALIASES.items():
        found = None
        for alias in aliases:
            found = normalized_columns.get(_normalize(alias))
            if found:
                break
        if found is None:
            missing.append(canonical)
        else:
            mapping[canonical] = found

    if missing:
        available = ", ".join(df.columns)
        raise ValueError(
            "Could not locate required column(s): "
            + ", ".join(missing)
            + f". Available columns: {available}"
        )
    return mapping


def load_dataset(dataset_path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Load tabular data and rename required columns to canonical names."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    suffix = dataset_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(dataset_path)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(dataset_path)
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_path.suffix}. Use CSV or Excel.")
    mapping = resolve_columns(df)
    selected_actual = [mapping[name] for name in FEATURES + [TARGET]]
    data = df[selected_actual].rename(columns={actual: canonical for canonical, actual in mapping.items()})
    return data, mapping


def validate_dataset(data: pd.DataFrame, column_mapping: dict[str, str]) -> dict[str, Any]:
    """Collect validation diagnostics for selected model columns."""
    required = FEATURES + [TARGET]
    report: dict[str, Any] = {
        "rows": int(data.shape[0]),
        "columns": int(data.shape[1]),
        "selected_columns": required,
        "source_column_mapping": column_mapping,
        "dtypes": {column: str(dtype) for column, dtype in data.dtypes.items()},
        "missing_values": data.isna().sum().astype(int).to_dict(),
        "duplicate_records": int(data.duplicated().sum()),
        "statistics": data.describe().T.to_dict(orient="index"),
        "target_distribution": data[TARGET].describe().to_dict(),
    }

    logging.info("Dataset rows: %s", report["rows"])
    logging.info("Dataset columns: %s", report["columns"])
    logging.info("Selected columns: %s", ", ".join(required))
    logging.info("Missing values: %s", report["missing_values"])
    logging.info("Duplicate selected records: %s", report["duplicate_records"])
    return report
