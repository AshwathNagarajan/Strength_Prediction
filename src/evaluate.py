"""Model evaluation, comparison, and plotting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score


def calculate_metrics(y_true, y_pred) -> dict[str, float]:
    """Calculate R², MAE, and RMSE."""
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    rmse = float(np.sqrt(np.mean((y_true_array - y_pred_array) ** 2)))
    return {
        "r2": float(r2_score(y_true_array, y_pred_array)),
        "mae": float(mean_absolute_error(y_true_array, y_pred_array)),
        "rmse": rmse,
    }


def select_best_model(metrics: dict[str, dict[str, float]]) -> str:
    """Select best model by R², then RMSE, then MAE."""
    return sorted(
        metrics,
        key=lambda name: (-metrics[name]["r2"], metrics[name]["rmse"], metrics[name]["mae"]),
    )[0]


def plot_actual_vs_predicted(y_true, y_pred, model_name: str, metrics: dict[str, float], output_path: Path) -> None:
    """Save actual-vs-predicted scatter plot."""
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    ax.scatter(y_true, y_pred, alpha=0.72, edgecolor="white", linewidth=0.5, color="#2563eb")
    low = min(float(np.min(y_true)), float(np.min(y_pred)))
    high = max(float(np.max(y_true)), float(np.max(y_pred)))
    ax.plot([low, high], [low, high], color="#dc2626", linestyle="--", linewidth=1.5, label="Ideal fit")
    ax.set_title(f"Actual vs Predicted - {model_name}")
    ax.set_xlabel("Actual Compressive Strength (MPa)")
    ax.set_ylabel("Predicted Compressive Strength (MPa)")
    ax.text(0.05, 0.95, f"R² = {metrics['r2']:.3f}", transform=ax.transAxes, va="top", bbox={"facecolor": "white", "alpha": 0.85})
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_residuals(y_true, y_pred, model_name: str, output_path: Path) -> None:
    """Save predicted-vs-residual plot."""
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    ax.scatter(y_pred, residuals, alpha=0.72, edgecolor="white", linewidth=0.5, color="#0f766e")
    ax.axhline(0, color="#dc2626", linestyle="--", linewidth=1.5)
    ax.set_title(f"Residual Plot - {model_name}")
    ax.set_xlabel("Predicted Compressive Strength (MPa)")
    ax.set_ylabel("Residual (Actual - Predicted) MPa")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_model_comparison(metrics: dict[str, dict[str, float]], output_path: Path) -> None:
    """Save grouped comparison chart for R², MAE, and RMSE."""
    df = pd.DataFrame(metrics).T[["r2", "mae", "rmse"]]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150)
    colors = ["#2563eb", "#16a34a"]
    for ax, metric, title in zip(axes, ["r2", "mae", "rmse"], ["R² (higher is better)", "MAE (lower is better)", "RMSE (lower is better)"]):
        ax.bar(df.index, df[metric], color=colors[: len(df.index)])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        for idx, value in enumerate(df[metric]):
            ax.text(idx, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("XGBoost vs CatBoost Model Comparison")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_metrics_table(metrics: dict[str, dict[str, float]], cv_results: dict[str, Any], output_dir: Path) -> None:
    """Save test and cross-validation metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).T.to_csv(output_dir / "test_metrics.csv", index_label="model")
    pd.DataFrame(cv_results).T.to_csv(output_dir / "cross_validation_metrics.csv", index_label="model")

