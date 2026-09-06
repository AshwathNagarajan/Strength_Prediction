"""Train prediction and explanation models for ultimate load and deflection."""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.performance_config import (
    CATEGORICAL_FEATURES,
    FEATURE_PROFILE_FILE,
    FEATURES,
    METADATA_FILE,
    NUMERIC_FEATURES,
    PERFORMANCE_FIGURES_DIR,
    PERFORMANCE_METRICS_DIR,
    PERFORMANCE_MODEL_DIR,
    PERFORMANCE_OUTPUT_DIR,
    PERFORMANCE_SHAP_DIR,
    TARGETS,
    DATASET_PATH,
)
from src.performance_data import clean_performance_data, feature_profile, load_performance_dataset, validate_performance_dataset
from src.performance_models import train_one_target
from src.utils import ensure_directories, save_json, setup_logging


def generate_eda(data: pd.DataFrame) -> None:
    """Generate compact EDA plots for selected structural-performance data."""
    sns.set_theme(style="whitegrid")
    PERFORMANCE_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for target_key, target in TARGETS.items():
        target_column = target["column"]
        fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
        sns.histplot(data[target_column], kde=True, ax=ax, color="#2563eb")
        ax.set_title(f"{target['label']} Distribution")
        ax.set_xlabel(f"{target['label']} ({target['unit']})")
        fig.tight_layout()
        fig.savefig(PERFORMANCE_FIGURES_DIR / f"{target_key}_distribution.png")
        plt.close(fig)

        for feature in ["Compressive Strength fc' (MPa)", "Yield Strength fy (MPa)", "Span L (mm)", "Beam Depth h (mm)", "Fly Ash (%)"]:
            fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
            sns.scatterplot(data=data, x=feature, y=target_column, ax=ax, alpha=0.65)
            ax.set_title(f"{feature} vs {target['label']}")
            fig.tight_layout()
            safe_feature = feature.lower().replace(" ", "_").replace("/", "_").replace("'", "").replace("(", "").replace(")", "").replace("%", "pct")
            fig.savefig(PERFORMANCE_FIGURES_DIR / f"{target_key}_{safe_feature}.png")
            plt.close(fig)

    corr_columns = NUMERIC_FEATURES + [target["column"] for target in TARGETS.values()]
    fig, ax = plt.subplots(figsize=(11, 9), dpi=150)
    sns.heatmap(data[corr_columns].corr(numeric_only=True), cmap="vlag", center=0, annot=False, ax=ax)
    ax.set_title("Correlation Heatmap: Inputs and Prediction Targets")
    fig.tight_layout()
    fig.savefig(PERFORMANCE_FIGURES_DIR / "performance_correlation_heatmap.png")
    plt.close(fig)


def main() -> None:
    """Run the full structural-performance training workflow."""
    setup_logging()
    ensure_directories([PERFORMANCE_MODEL_DIR, PERFORMANCE_OUTPUT_DIR, PERFORMANCE_FIGURES_DIR, PERFORMANCE_METRICS_DIR, PERFORMANCE_SHAP_DIR])

    logging.info("Loading performance dataset from %s", DATASET_PATH)
    selected, mapping = load_performance_dataset(DATASET_PATH)
    validation = validate_performance_dataset(selected, mapping)
    save_json(validation, PERFORMANCE_METRICS_DIR / "performance_dataset_validation.json")

    data = clean_performance_data(selected)
    save_json(feature_profile(data[FEATURES]), FEATURE_PROFILE_FILE)
    generate_eda(data)

    metadata = {
        "dataset": {
            "path": str(DATASET_PATH),
            "rows_used_after_preprocessing": int(len(data)),
            "features": FEATURES,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "targets": TARGETS,
            "source_column_mapping": mapping,
            "excluded_leakage_columns": [
                "Target: Capacity Pu (kN)",
                "Target: Deflection δ (mm)",
                "Target: Shear Qu (kN)",
                "Target: Failure Mode",
                "Concrete GWP (kg CO2e/m3)",
            ],
        },
        "targets": {},
    }

    X = data[FEATURES]
    for target_key, target in TARGETS.items():
        result = train_one_target(X, data[target["column"]], target_key, target["label"], target["unit"])
        metadata["targets"][target_key] = result
        logging.info("%s best model: %s", target["label"], result["best_model"])
        logging.info("%s metrics: %s", target["label"], result["metrics"])

    save_json(metadata, METADATA_FILE)
    logging.info("Performance training complete.")


if __name__ == "__main__":
    main()

