from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError("Only CSV and XLSX datasets are supported.")


def preview_table(data: pd.DataFrame) -> dict:
    numeric = data.select_dtypes(include="number")
    return {
        "rows": int(len(data)),
        "columns": list(data.columns),
        "dtypes": {name: str(dtype) for name, dtype in data.dtypes.items()},
        "missing": {name: int(value) for name, value in data.isna().sum().items()},
        "numeric_statistics": numeric.describe().replace({float("nan"): None}).to_dict(),
        "preview": data.head(20).where(pd.notna(data.head(20)), None).to_dict(orient="records"),
    }
