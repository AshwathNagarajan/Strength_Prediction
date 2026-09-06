"""Training, evaluation, and SHAP utilities for structural performance models."""

from __future__ import annotations

import logging
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostRegressor
from scipy.stats import randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from .performance_config import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    PERFORMANCE_FIGURES_DIR,
    PERFORMANCE_MODEL_DIR,
    PERFORMANCE_SHAP_DIR,
    RANDOM_STATE,
)


def build_preprocessor() -> ColumnTransformer:
    """Create preprocessing for numeric and categorical inputs."""
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def transformed_frame(preprocessor: ColumnTransformer, X: pd.DataFrame) -> pd.DataFrame:
    """Transform features and return a dataframe with generated feature names."""
    transformed = preprocessor.transform(X)
    names = preprocessor.get_feature_names_out()
    return pd.DataFrame(transformed, columns=names, index=X.index)


def calculate_metrics(y_true, y_pred) -> dict[str, float]:
    """Calculate regression metrics."""
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    return {
        "r2": float(r2_score(y_true_array, y_pred_array)),
        "mae": float(mean_absolute_error(y_true_array, y_pred_array)),
        "rmse": float(np.sqrt(np.mean((y_true_array - y_pred_array) ** 2))),
    }


def select_best(metrics: dict[str, dict[str, float]]) -> str:
    """Select best model by highest R², lower RMSE, then lower MAE."""
    return sorted(metrics, key=lambda name: (-metrics[name]["r2"], metrics[name]["rmse"], metrics[name]["mae"]))[0]


def _cv_summary(model, X, y) -> dict[str, float]:
    scores = cross_validate(
        model,
        X,
        y,
        cv=5,
        scoring={"r2": "r2", "mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error"},
        n_jobs=-1,
    )
    return {
        "cv_r2_mean": float(scores["test_r2"].mean()),
        "cv_r2_std": float(scores["test_r2"].std()),
        "cv_mae_mean": float((-scores["test_mae"]).mean()),
        "cv_rmse_mean": float((-scores["test_rmse"]).mean()),
    }


def train_one_target(X: pd.DataFrame, y: pd.Series, target_key: str, target_label: str, unit: str) -> dict[str, Any]:
    """Train XGBoost and CatBoost for one target."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE)
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)
    X_train_t = transformed_frame(preprocessor, X_train)
    X_test_t = transformed_frame(preprocessor, X_test)

    xgb_search = RandomizedSearchCV(
        XGBRegressor(objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=-1, eval_metric="rmse"),
        {
            "n_estimators": randint(150, 550),
            "max_depth": randint(2, 7),
            "learning_rate": uniform(0.015, 0.12),
            "subsample": uniform(0.70, 0.30),
            "colsample_bytree": uniform(0.70, 0.30),
            "min_child_weight": randint(1, 8),
            "reg_alpha": uniform(0.0, 0.5),
            "reg_lambda": uniform(0.5, 2.0),
        },
        n_iter=14,
        scoring="neg_root_mean_squared_error",
        cv=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    logging.info("Tuning XGBoost for %s.", target_label)
    xgb_search.fit(X_train_t, y_train)
    xgb = xgb_search.best_estimator_

    cat_search = RandomizedSearchCV(
        CatBoostRegressor(loss_function="RMSE", random_seed=RANDOM_STATE, verbose=False, allow_writing_files=False),
        {
            "iterations": randint(200, 650),
            "depth": randint(2, 8),
            "learning_rate": uniform(0.015, 0.12),
            "l2_leaf_reg": uniform(1.0, 8.0),
            "random_strength": uniform(0.0, 2.0),
            "bagging_temperature": uniform(0.0, 1.0),
        },
        n_iter=14,
        scoring="neg_root_mean_squared_error",
        cv=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    logging.info("Tuning CatBoost for %s.", target_label)
    cat_search.fit(X_train_t, y_train)
    cat = CatBoostRegressor(
        loss_function="RMSE",
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
        **{key: (int(value) if key in {"iterations", "depth"} else float(value)) for key, value in cat_search.best_params_.items()},
    )
    cat.fit(X_train_t, y_train)

    models = {"XGBoost": xgb, "CatBoost": cat}
    metrics: dict[str, dict[str, float]] = {}
    cv_results: dict[str, dict[str, float]] = {}
    predictions: dict[str, np.ndarray] = {}
    for name, model in models.items():
        pred = model.predict(X_test_t)
        predictions[name] = pred
        metrics[name] = calculate_metrics(y_test, pred)
        cv_results[name] = _cv_summary(model, X_train_t, y_train)
        _plot_actual_vs_predicted(y_test, pred, name, target_label, unit, metrics[name], PERFORMANCE_FIGURES_DIR / f"{target_key}_{name.lower()}_actual_vs_predicted.png")
        _plot_residuals(y_test, pred, name, target_label, unit, PERFORMANCE_FIGURES_DIR / f"{target_key}_{name.lower()}_residual_plot.png")
        _global_shap(model, X_train_t, target_key, name)

    best = select_best(metrics)
    artifact = {
        "preprocessor": preprocessor,
        "models": models,
        "best_model": best,
        "features": list(X.columns),
        "transformed_features": list(X_train_t.columns),
        "target_key": target_key,
        "target_label": target_label,
        "unit": unit,
    }
    PERFORMANCE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, PERFORMANCE_MODEL_DIR / f"{target_key}_model_bundle.pkl")

    pd.DataFrame(metrics).T.to_csv(PERFORMANCE_FIGURES_DIR.parent / "metrics" / f"{target_key}_test_metrics.csv", index_label="model")

    return {
        "best_model": best,
        "metrics": metrics,
        "cross_validation": cv_results,
        "best_params": {"XGBoost": xgb_search.best_params_, "CatBoost": cat_search.best_params_},
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }


def _plot_actual_vs_predicted(y_true, y_pred, model_name: str, target_label: str, unit: str, metrics: dict[str, float], output_path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    ax.scatter(y_true, y_pred, alpha=0.7, edgecolor="white", linewidth=0.5, color="#2563eb")
    low = min(float(np.min(y_true)), float(np.min(y_pred)))
    high = max(float(np.max(y_true)), float(np.max(y_pred)))
    ax.plot([low, high], [low, high], color="#dc2626", linestyle="--", label="Ideal fit")
    ax.set_title(f"{target_label}: Actual vs Predicted - {model_name}")
    ax.set_xlabel(f"Actual {target_label} ({unit})")
    ax.set_ylabel(f"Predicted {target_label} ({unit})")
    ax.text(0.05, 0.95, f"R² = {metrics['r2']:.3f}", transform=ax.transAxes, va="top", bbox={"facecolor": "white", "alpha": 0.85})
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_residuals(y_true, y_pred, model_name: str, target_label: str, unit: str, output_path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    ax.scatter(y_pred, residuals, alpha=0.7, edgecolor="white", linewidth=0.5, color="#0f766e")
    ax.axhline(0, color="#dc2626", linestyle="--")
    ax.set_title(f"{target_label}: Residual Plot - {model_name}")
    ax.set_xlabel(f"Predicted {target_label} ({unit})")
    ax.set_ylabel(f"Residual ({unit})")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _global_shap(model, X_train_t: pd.DataFrame, target_key: str, model_name: str) -> None:
    PERFORMANCE_SHAP_DIR.mkdir(parents=True, exist_ok=True)
    sample = X_train_t.sample(n=min(500, len(X_train_t)), random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(sample)
    if np.asarray(values).ndim == 3:
        values = np.asarray(values)[:, :, 0]

    plt.figure(figsize=(9, 6), dpi=150)
    shap.summary_plot(values, sample, show=False)
    plt.tight_layout()
    plt.savefig(PERFORMANCE_SHAP_DIR / f"{target_key}_{model_name.lower()}_shap_summary.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 6), dpi=150)
    shap.summary_plot(values, sample, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(PERFORMANCE_SHAP_DIR / f"{target_key}_{model_name.lower()}_shap_importance.png", bbox_inches="tight")
    plt.close()

    importance = pd.DataFrame({"feature": sample.columns, "mean_abs_shap": np.abs(values).mean(axis=0)})
    importance.sort_values("mean_abs_shap", ascending=False).to_csv(
        PERFORMANCE_SHAP_DIR / f"{target_key}_{model_name.lower()}_shap_importance.csv", index=False
    )

