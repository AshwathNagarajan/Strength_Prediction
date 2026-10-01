from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.ml.model_registry import ModelRegistry
from app.ml.validator import domain_metadata


def _model_factories() -> dict[str, Callable[[], object]]:
    models: dict[str, Callable[[], object]] = {
        "LinearRegression": LinearRegression,
        "Ridge": lambda: Ridge(alpha=1.0),
        "RandomForest": lambda: RandomForestRegressor(n_estimators=160, random_state=42, n_jobs=-1),
        "ExtraTrees": lambda: ExtraTreesRegressor(n_estimators=160, random_state=42, n_jobs=-1),
        "GradientBoosting": lambda: GradientBoostingRegressor(random_state=42),
        "MLP": lambda: MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=700, random_state=42),
    }
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = lambda: XGBRegressor(n_estimators=250, max_depth=5, learning_rate=0.05, random_state=42, n_jobs=-1)
    except ImportError:
        pass
    try:
        from lightgbm import LGBMRegressor
        models["LightGBM"] = lambda: LGBMRegressor(n_estimators=250, random_state=42, verbosity=-1)
    except ImportError:
        pass
    try:
        from catboost import CatBoostRegressor
        models["CatBoost"] = lambda: CatBoostRegressor(iterations=250, depth=6, verbose=False, random_seed=42)
    except ImportError:
        pass
    return models


def _preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    numeric = frame.select_dtypes(include="number").columns.tolist()
    categorical = [name for name in frame.columns if name not in numeric]
    return ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])


def train_models(data: pd.DataFrame, inputs: list[str], targets: list[str], test_size: float, cv_folds: int, mode: str = "quick", selected_models: list[str] | None = None, status_callback=None) -> dict:
    clean = data.dropna(subset=targets).copy()
    if len(clean) < 12:
        raise ValueError("At least 12 complete rows are required for model training.")
    x, y = clean[inputs], clean[targets]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
    folds = min(cv_folds, max(2, len(x_train) // 3))
    cv = KFold(n_splits=folds, shuffle=True, random_state=42)
    results, fitted = {}, {}
    all_model_names = {"LinearRegression", "Ridge", "RandomForest", "ExtraTrees", "GradientBoosting", "MLP", "XGBoost", "LightGBM", "CatBoost"}
    factories = _model_factories()
    if selected_models:
        unknown = sorted(set(selected_models) - all_model_names)
        if unknown: raise ValueError("Unknown models: " + ", ".join(unknown))
        requested = set(selected_models)
    else: requested = all_model_names
    unavailable = requested - set(factories)
    for name in unavailable: results[name] = {"status": "unavailable", "reason": "Optional dependency is not installed.", "training_time": 0.0}
    factories = {name: factory for name, factory in factories.items() if name in requested}
    tuning_spaces = {
        "Ridge": {"model__alpha": np.logspace(-3, 2, 20)},
        "RandomForest": {"model__n_estimators": [120, 200, 300], "model__max_depth": [None, 8, 14], "model__min_samples_leaf": [1, 2, 4]},
        "ExtraTrees": {"model__n_estimators": [120, 200, 300], "model__max_depth": [None, 10, 18], "model__min_samples_leaf": [1, 2, 4]},
    }
    for name, factory in factories.items():
        started = time.perf_counter()
        estimator = factory()
        if len(targets) > 1 and name in {"GradientBoosting", "XGBoost", "LightGBM", "CatBoost"}:
            estimator = MultiOutputRegressor(estimator)
        pipeline = Pipeline([("preprocessor", _preprocessor(x_train)), ("model", estimator)])
        try:
            fit_y = y_train if len(targets) > 1 else y_train.iloc[:, 0]
            best_parameters = {}
            if mode == "standard" and name in tuning_spaces:
                search = RandomizedSearchCV(pipeline, tuning_spaces[name], n_iter=int(min(8, np.prod([len(v) for v in tuning_spaces[name].values()]))), scoring="r2", cv=cv, random_state=42, n_jobs=-1)
                search.fit(x_train, fit_y)
                pipeline, best_parameters = search.best_estimator_, search.best_params_
            else:
                pipeline.fit(x_train, fit_y)
            predicted = np.asarray(pipeline.predict(x_test))
            actual = y_test.to_numpy()
            train_predicted = np.asarray(pipeline.predict(x_train))
            train_actual = y_train.to_numpy()
            if predicted.ndim == 1:
                predicted, actual = predicted[:, None], actual[:, :1]
            if train_predicted.ndim == 1:
                train_predicted, train_actual = train_predicted[:, None], train_actual[:, :1]
            metrics = {}
            for index, target in enumerate(targets):
                test_values = actual[:, index]
                metrics[target] = {
                    "training_r2": float(r2_score(train_actual[:, index], train_predicted[:, index])),
                    "r2": float(r2_score(test_values, predicted[:, index])),
                    "mae": float(mean_absolute_error(test_values, predicted[:, index])),
                    "rmse": float(mean_squared_error(test_values, predicted[:, index]) ** 0.5),
                    "mape": float(mean_absolute_percentage_error(test_values, predicted[:, index]) * 100) if np.all(test_values != 0) else None,
                }
                metrics[target]["overfitting_warning"] = metrics[target]["training_r2"] - metrics[target]["r2"] > 0.15
            cv_score = cross_val_score(pipeline, x_train, y_train if len(targets) > 1 else y_train.iloc[:, 0], cv=cv, scoring="r2").mean()
            results[name] = {"status": "trained", "metrics": metrics, "cv_r2": float(cv_score), "training_time": time.perf_counter() - started, "best_parameters": best_parameters}
            fitted[name] = pipeline
        except Exception as exc:
            results[name] = {"status": "unavailable", "reason": str(exc), "training_time": time.perf_counter() - started}
    usable = [name for name, value in results.items() if value["status"] == "trained"]
    if not usable:
        raise RuntimeError("Every training algorithm failed.")
    if status_callback: status_callback("evaluating", "Selecting the best validated model")
    def rank(name: str) -> tuple[float, float, float]:
        values = results[name]
        mean_r2 = float(np.mean([row["r2"] for row in values["metrics"].values()]))
        mean_rmse = float(np.mean([row["rmse"] for row in values["metrics"].values()]))
        return (0.65 * values["cv_r2"] + 0.35 * mean_r2, -mean_rmse, -values["training_time"])
    best = max(usable, key=rank)
    best_predictions = np.asarray(fitted[best].predict(x_test))
    if best_predictions.ndim == 1: best_predictions = best_predictions[:, None]
    sample_size = min(300, len(x_test))
    evaluation = {
        target: [
            {"actual": float(y_test.iloc[row_index, target_index]), "predicted": float(best_predictions[row_index, target_index]), "residual": float(y_test.iloc[row_index, target_index] - best_predictions[row_index, target_index])}
            for row_index in range(sample_size)
        ] for target_index, target in enumerate(targets)
    }
    metadata = {
        "active_model": best, "targets": targets, "inputs": inputs, "metrics": results,
        "trained_at": datetime.now(timezone.utc).isoformat(), "dataset_rows": len(clean), "cv_folds": folds, "tuning_mode": mode, "evaluation": evaluation,
    }
    feature_domain = domain_metadata(clean, inputs)
    metadata["feature_domain"] = feature_domain
    metadata["random_state"] = 42
    metadata["training_samples"] = len(x_train)
    metadata["test_samples"] = len(x_test)
    metadata["hyperparameters"] = fitted[best].named_steps["model"].get_params(deep=False)
    if status_callback: status_callback("saving", "Saving versioned model artifacts")
    artifact_dir = ModelRegistry().save({"pipeline": fitted[best], "models": fitted, "metadata": metadata}, metadata)
    return {**metadata, "artifact_dir": str(artifact_dir.resolve()), "feature_domain": feature_domain}
