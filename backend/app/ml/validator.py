from __future__ import annotations

import numpy as np
import pandas as pd


def quality_report(data: pd.DataFrame) -> dict:
    numeric = data.select_dtypes(include="number")
    outliers: dict[str, int] = {}
    suspicious: list[str] = []
    for column in numeric.columns:
        series = numeric[column].dropna()
        if series.empty:
            outliers[column] = 0
            continue
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers[column] = int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())
        lowered = column.lower()
        if any(word in lowered for word in ("width", "depth", "span", "thickness", "strength", "diameter")) and (series < 0).any():
            suspicious.append(f"{column} contains negative engineering values.")
        if any(word in lowered for word in ("percent", "pct", "%")) and ((series < 0) | (series > 100)).any():
            suspicious.append(f"{column} contains values outside 0-100%.")
    correlation = numeric.corr()
    strong_correlations = []
    names = list(correlation.columns)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            value = correlation.loc[left, right]
            if pd.notna(value) and abs(value) > 0.9:
                strong_correlations.append({"feature_a": left, "feature_b": right, "correlation": float(value)})
    return {
        "rows": int(len(data)),
        "columns": int(len(data.columns)),
        "missing_values": int(data.isna().sum().sum()),
        "duplicate_rows": int(data.duplicated().sum()),
        "zero_variance_features": [name for name in data.columns if data[name].nunique(dropna=True) <= 1],
        "potential_outliers": outliers,
        "suspicious_ranges": suspicious,
        "correlation": correlation.replace([np.inf, -np.inf, np.nan], None).to_dict(),
        "strong_correlation_warnings": strong_correlations,
    }


def domain_metadata(data: pd.DataFrame, features: list[str]) -> dict:
    result: dict = {}
    for name in features:
        series = data[name]
        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna().astype(float)
            result[name] = {
                "kind": "numeric", "min": float(clean.min()), "max": float(clean.max()),
                "mean": float(clean.mean()), "median": float(clean.median()), "std": float(clean.std(ddof=0)),
                "q05": float(clean.quantile(0.05)), "q25": float(clean.quantile(0.25)),
                "q75": float(clean.quantile(0.75)), "q95": float(clean.quantile(0.95)),
                "unique_values": sorted(clean.unique().tolist()) if clean.nunique() <= 30 else [],
            }
        else:
            result[name] = {"kind": "categorical", "unique_values": sorted(series.dropna().astype(str).unique().tolist())}
    return result
