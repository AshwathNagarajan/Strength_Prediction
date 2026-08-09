"""SHAP explainability helpers for global and local explanations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from .config import FEATURES


def _as_array(values: Any) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 0]
    return values


def build_explainer(model, X_background: pd.DataFrame):
    """Build a SHAP explainer for a tree-based model."""
    return shap.TreeExplainer(model)


def calculate_shap_values(model, X: pd.DataFrame) -> tuple[Any, np.ndarray]:
    """Return explainer and SHAP value matrix."""
    explainer = build_explainer(model, X)
    shap_values = _as_array(explainer.shap_values(X))
    return explainer, shap_values


def generate_global_shap_plots(model, X: pd.DataFrame, model_name: str, output_dir: Path, max_samples: int = 500) -> dict[str, str]:
    """Generate summary and bar SHAP plots for a model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    X_sample = X.sample(n=min(max_samples, len(X)), random_state=42) if len(X) > max_samples else X.copy()
    _, shap_values = calculate_shap_values(model, X_sample)

    summary_path = output_dir / f"{model_name.lower()}_shap_summary.png"
    bar_path = output_dir / f"{model_name.lower()}_shap_feature_importance.png"

    plt.figure(figsize=(8, 5), dpi=150)
    shap.summary_plot(shap_values, X_sample, feature_names=FEATURES, show=False)
    plt.tight_layout()
    plt.savefig(summary_path, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5), dpi=150)
    shap.summary_plot(shap_values, X_sample, feature_names=FEATURES, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(bar_path, bbox_inches="tight")
    plt.close()

    importance = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({"feature": FEATURES, "mean_abs_shap": importance}).sort_values("mean_abs_shap", ascending=False)
    importance_df.to_csv(output_dir / f"{model_name.lower()}_shap_importance.csv", index=False)

    return {"summary": str(summary_path), "bar": str(bar_path)}


def explain_single_prediction(model, input_df: pd.DataFrame) -> dict[str, Any]:
    """Calculate local SHAP contributions and deterministic explanation text."""
    explainer = build_explainer(model, input_df)
    raw_values = _as_array(explainer.shap_values(input_df))
    contributions = raw_values[0].astype(float)
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = float(np.asarray(expected_value).ravel()[0])
    else:
        expected_value = float(expected_value)

    predicted = float(model.predict(input_df)[0])
    rows = []
    for feature, value, contribution in zip(FEATURES, input_df.iloc[0].values, contributions):
        rows.append(
            {
                "Feature": feature,
                "Input": float(value),
                "SHAP Contribution": float(contribution),
                "Effect": "Increased" if contribution >= 0 else "Reduced",
            }
        )

    sorted_rows = sorted(rows, key=lambda row: abs(row["SHAP Contribution"]), reverse=True)
    top = sorted_rows[0]
    second = sorted_rows[1] if len(sorted_rows) > 1 else None
    reduced = [row["Feature"] for row in sorted_rows if row["SHAP Contribution"] < 0]
    increased = [row["Feature"] for row in sorted_rows if row["SHAP Contribution"] >= 0]

    text = (
        f"The model predicts a compressive strength of {predicted:.2f} MPa. "
        f"For this mixture, {top['Feature']} had the strongest "
        f"{'positive' if top['SHAP Contribution'] >= 0 else 'negative'} influence relative to the model baseline."
    )
    if second:
        text += (
            f" {second['Feature']} was the second most influential parameter and "
            f"{'increased' if second['SHAP Contribution'] >= 0 else 'reduced'} the prediction."
        )
    if reduced:
        text += " Reduced contributors: " + ", ".join(reduced) + "."
    if increased:
        text += " Positive contributors: " + ", ".join(increased) + "."

    return {
        "prediction": predicted,
        "baseline": expected_value,
        "contributions": rows,
        "explanation": text,
    }


def create_waterfall_figure(local_explanation: dict[str, Any]):
    """Create a matplotlib waterfall plot for one prediction."""
    values = np.array([row["SHAP Contribution"] for row in local_explanation["contributions"]], dtype=float)
    data = np.array([row["Input"] for row in local_explanation["contributions"]], dtype=float)
    explanation = shap.Explanation(
        values=values,
        base_values=local_explanation["baseline"],
        data=data,
        feature_names=FEATURES,
    )
    shap.plots.waterfall(explanation, show=False)
    fig = plt.gcf()
    fig.set_size_inches(8, 4.8)
    fig.tight_layout()
    return fig

