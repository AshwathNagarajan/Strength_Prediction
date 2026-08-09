"""Training and tuning for XGBoost and CatBoost regressors."""

from __future__ import annotations

import logging
from typing import Any

from catboost import CatBoostRegressor
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV, cross_validate
from xgboost import XGBRegressor

from .config import CV_FOLDS, RANDOM_STATE


def _clean_params(params: dict[str, Any]) -> dict[str, Any]:
    """Convert scipy/numpy sampled params to plain Python scalars."""
    cleaned: dict[str, Any] = {}
    integer_params = {"n_estimators", "max_depth", "min_child_weight", "iterations", "depth"}
    for key, value in params.items():
        if key in integer_params:
            cleaned[key] = int(value)
        else:
            cleaned[key] = float(value) if hasattr(value, "__float__") else value
    return cleaned


def _cv_summary(estimator, X, y) -> dict[str, float]:
    scores = cross_validate(
        estimator,
        X,
        y,
        cv=CV_FOLDS,
        scoring={"r2": "r2", "mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error"},
        n_jobs=-1,
    )
    return {
        "cv_r2_mean": float(scores["test_r2"].mean()),
        "cv_r2_std": float(scores["test_r2"].std()),
        "cv_mae_mean": float((-scores["test_mae"]).mean()),
        "cv_rmse_mean": float((-scores["test_rmse"]).mean()),
    }


def tune_xgboost(X_train, y_train) -> tuple[XGBRegressor, dict[str, float], dict]:
    """Tune and fit XGBoost using randomized search."""
    logging.info("Tuning XGBoost with %s-fold CV.", CV_FOLDS)
    estimator = XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="rmse",
    )
    params = {
        "n_estimators": randint(150, 550),
        "max_depth": randint(2, 7),
        "learning_rate": uniform(0.015, 0.12),
        "subsample": uniform(0.70, 0.30),
        "colsample_bytree": uniform(0.70, 0.30),
        "min_child_weight": randint(1, 8),
        "reg_alpha": uniform(0.0, 0.5),
        "reg_lambda": uniform(0.5, 2.0),
    }
    search = RandomizedSearchCV(
        estimator,
        params,
        n_iter=18,
        scoring="neg_root_mean_squared_error",
        cv=CV_FOLDS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    best_params = _clean_params(search.best_params_)
    best_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="rmse",
        **best_params,
    )
    best_model.fit(X_train, y_train)
    cv_summary = _cv_summary(best_model, X_train, y_train)
    return best_model, cv_summary, best_params


def tune_catboost(X_train, y_train) -> tuple[CatBoostRegressor, dict[str, float], dict]:
    """Tune and fit CatBoost using randomized search."""
    logging.info("Tuning CatBoost with %s-fold CV.", CV_FOLDS)
    estimator = CatBoostRegressor(
        loss_function="RMSE",
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
    )
    params = {
        "iterations": randint(200, 700),
        "depth": randint(2, 8),
        "learning_rate": uniform(0.015, 0.12),
        "l2_leaf_reg": uniform(1.0, 8.0),
        "random_strength": uniform(0.0, 2.0),
        "bagging_temperature": uniform(0.0, 1.0),
    }
    search = RandomizedSearchCV(
        estimator,
        params,
        n_iter=18,
        scoring="neg_root_mean_squared_error",
        cv=CV_FOLDS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    best_params = _clean_params(search.best_params_)
    best_model = CatBoostRegressor(
        loss_function="RMSE",
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
        **best_params,
    )
    best_model.fit(X_train, y_train)
    cv_summary = _cv_summary(best_model, X_train, y_train)
    return best_model, cv_summary, best_params
