"""Train Review 1 compressive-strength prediction models."""

from __future__ import annotations

import logging

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import (
    BEST_MODEL_FILE,
    DATASET_PATH,
    FEATURES,
    FIGURES_DIR,
    METADATA_FILE,
    METRICS_DIR,
    MODEL_DIR,
    MODEL_FILES,
    OUTPUT_DIR,
    SHAP_DIR,
    TARGET,
    FEATURE_RANGES_FILE,
)
from src.data_loader import load_dataset, validate_dataset
from src.evaluate import (
    calculate_metrics,
    plot_actual_vs_predicted,
    plot_model_comparison,
    plot_residuals,
    save_metrics_table,
    select_best_model,
)
from src.explainability import generate_global_shap_plots
from src.preprocessing import compute_feature_ranges, prepare_model_data, split_data
from src.train_models import tune_catboost, tune_xgboost
from src.utils import ensure_directories, save_json, setup_logging


def generate_eda_plots(data: pd.DataFrame) -> None:
    """Generate required exploratory data-analysis figures."""
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    sns.histplot(data[TARGET], kde=True, ax=ax, color="#2563eb")
    ax.set_title("Compressive Strength Distribution")
    ax.set_xlabel("Compressive Strength fc' (MPa)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_strength_distribution.png")
    plt.close(fig)

    plot_names = {
        "w/b Ratio": "eda_wb_ratio_vs_strength.png",
        "Fly Ash (%)": "eda_fly_ash_vs_strength.png",
        "GGBS (%)": "eda_ggbs_vs_strength.png",
        "Recycled Aggregate (%)": "eda_recycled_aggregate_vs_strength.png",
    }
    for feature, filename in plot_names.items():
        fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
        sns.regplot(data=data, x=feature, y=TARGET, ax=ax, scatter_kws={"alpha": 0.65}, line_kws={"color": "#dc2626"})
        ax.set_title(f"{feature} vs Compressive Strength")
        ax.set_ylabel("Compressive Strength fc' (MPa)")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / filename)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    corr = data[FEATURES + [TARGET]].corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap="vlag", center=0, fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_correlation_heatmap.png")
    plt.close(fig)


def main() -> None:
    """Run the full training workflow."""
    setup_logging()
    ensure_directories([MODEL_DIR, OUTPUT_DIR, FIGURES_DIR, METRICS_DIR, SHAP_DIR])

    logging.info("Loading dataset from %s", DATASET_PATH)
    raw_selected_data, column_mapping = load_dataset(DATASET_PATH)
    validation_report = validate_dataset(raw_selected_data, column_mapping)
    save_json(validation_report, METRICS_DIR / "dataset_validation.json")

    X, y = prepare_model_data(raw_selected_data)
    model_data = pd.concat([X, y], axis=1)
    generate_eda_plots(model_data)

    X_train, X_test, y_train, y_test = split_data(X, y)
    feature_ranges = compute_feature_ranges(X_train)
    save_json(feature_ranges, FEATURE_RANGES_FILE)

    xgb_model, xgb_cv, xgb_params = tune_xgboost(X_train, y_train)
    cat_model, cat_cv, cat_params = tune_catboost(X_train, y_train)

    models = {"XGBoost": xgb_model, "CatBoost": cat_model}
    metrics = {}
    predictions = {}
    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        predictions[model_name] = y_pred
        metrics[model_name] = calculate_metrics(y_test, y_pred)
        safe_name = model_name.lower()
        plot_actual_vs_predicted(
            y_test,
            y_pred,
            model_name,
            metrics[model_name],
            FIGURES_DIR / f"{safe_name}_actual_vs_predicted.png",
        )
        plot_residuals(y_test, y_pred, model_name, FIGURES_DIR / f"{safe_name}_residual_plot.png")
        joblib.dump(model, MODEL_FILES[model_name])

    plot_model_comparison(metrics, FIGURES_DIR / "model_comparison_metrics.png")
    best_model_name = select_best_model(metrics)
    joblib.dump(models[best_model_name], BEST_MODEL_FILE)

    cv_results = {"XGBoost": xgb_cv, "CatBoost": cat_cv}
    save_metrics_table(metrics, cv_results, METRICS_DIR)
    for model_name, model in models.items():
        generate_global_shap_plots(model, X_train, model_name, SHAP_DIR)

    metadata = {
        "best_model": best_model_name,
        "xgboost": metrics["XGBoost"],
        "catboost": metrics["CatBoost"],
        "cross_validation": cv_results,
        "best_params": {"XGBoost": xgb_params, "CatBoost": cat_params},
        "dataset": {
            "path": str(DATASET_PATH),
            "rows_used_after_preprocessing": int(len(X)),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "features": FEATURES,
            "target": TARGET,
            "source_column_mapping": column_mapping,
        },
    }
    save_json(metadata, METADATA_FILE)

    logging.info("Training complete.")
    logging.info("XGBoost metrics: %s", metrics["XGBoost"])
    logging.info("CatBoost metrics: %s", metrics["CatBoost"])
    logging.info("Best model: %s", best_model_name)


if __name__ == "__main__":
    main()

