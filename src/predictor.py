"""Reusable prediction API for trained compressive-strength models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .config import BEST_MODEL_FILE, FEATURE_RANGES_FILE, FEATURES, METADATA_FILE, MODEL_FILES
from .explainability import explain_single_prediction
from .utils import load_json


class StrengthPredictor:
    """Load saved models once and serve predictions with SHAP explanations."""

    def __init__(self) -> None:
        self.metadata = load_json(METADATA_FILE)
        self.feature_ranges = load_json(FEATURE_RANGES_FILE)
        missing = [str(path) for path in [*MODEL_FILES.values(), BEST_MODEL_FILE] if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing trained model files: " + ", ".join(missing))
        self.models = {name: joblib.load(path) for name, path in MODEL_FILES.items()}
        self.best_model_name = self.metadata["best_model"]
        self.best_model = joblib.load(BEST_MODEL_FILE)

    def _input_frame(self, wb_ratio: float, fly_ash: float, ggbs: float, recycled_aggregate: float) -> pd.DataFrame:
        values = [wb_ratio, fly_ash, ggbs, recycled_aggregate]
        for feature, value in zip(FEATURES, values):
            if value is None:
                raise ValueError(f"{feature} is required.")
            try:
                float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{feature} must be numeric.") from exc
        return pd.DataFrame([values], columns=FEATURES, dtype=float)

    def validate_domain(self, input_df: pd.DataFrame) -> list[str]:
        """Return warning messages for values outside training ranges."""
        warnings = []
        for feature in FEATURES:
            value = float(input_df.iloc[0][feature])
            min_value = float(self.feature_ranges[feature]["min"])
            max_value = float(self.feature_ranges[feature]["max"])
            if value < min_value or value > max_value:
                warnings.append(
                    f"{feature}={value:g} is outside the training range [{min_value:g}, {max_value:g}]. "
                    "The prediction is an extrapolation and may be less reliable."
                )
        return warnings

    def predict_strength(self, wb_ratio: float, fly_ash: float, ggbs: float, recycled_aggregate: float) -> dict[str, Any]:
        """Predict compressive strength from four sustainable concrete inputs."""
        input_df = self._input_frame(wb_ratio, fly_ash, ggbs, recycled_aggregate)
        predictions = {
            "XGBoost": float(self.models["XGBoost"].predict(input_df)[0]),
            "CatBoost": float(self.models["CatBoost"].predict(input_df)[0]),
        }
        best_prediction = predictions[self.best_model_name]
        local_explanation = explain_single_prediction(self.best_model, input_df)
        difference = abs(predictions["XGBoost"] - predictions["CatBoost"])
        agreement_threshold = max(
            float(self.metadata["xgboost"]["rmse"]),
            float(self.metadata["catboost"]["rmse"]),
        )

        return {
            "xgboost_prediction": predictions["XGBoost"],
            "catboost_prediction": predictions["CatBoost"],
            "best_model": self.best_model_name,
            "best_prediction": best_prediction,
            "difference_between_models": difference,
            "percentage_difference_between_models": difference / max(abs(best_prediction), 1e-9) * 100,
            "agreement_threshold_mpa": agreement_threshold,
            "agreement_warning": difference > agreement_threshold,
            "domain_warnings": self.validate_domain(input_df),
            "shap_values": local_explanation["contributions"],
            "baseline_prediction": local_explanation["baseline"],
            "explanation": local_explanation["explanation"],
            "input_data": input_df,
        }


def predict_strength(wb_ratio: float, fly_ash: float, ggbs: float, recycled_aggregate: float) -> dict[str, Any]:
    """Convenience function for one-off predictions."""
    return StrengthPredictor().predict_strength(wb_ratio, fly_ash, ggbs, recycled_aggregate)

