"""Prediction and SHAP explanation API for ultimate load and deflection."""

from __future__ import annotations

from typing import Any
import re

import joblib
import numpy as np
import pandas as pd
import shap

from .performance_config import (
    CATEGORICAL_FEATURES,
    FEATURE_PROFILE_FILE,
    FEATURES,
    METADATA_FILE,
    NUMERIC_FEATURES,
    PERFORMANCE_MODEL_DIR,
    TARGETS,
)
from .performance_models import transformed_frame
from .utils import load_json


class PerformancePredictor:
    """Load saved structural-performance models and explain predictions."""

    def __init__(self) -> None:
        if not METADATA_FILE.exists():
            raise FileNotFoundError("Performance models are missing. Run `python train_performance.py` first.")
        self.metadata = load_json(METADATA_FILE)
        self.profile = load_json(FEATURE_PROFILE_FILE)
        self.bundles = {
            target_key: joblib.load(PERFORMANCE_MODEL_DIR / f"{target_key}_model_bundle.pkl")
            for target_key in TARGETS
        }

    def default_inputs(self) -> dict[str, Any]:
        """Return midpoint numeric defaults and first categorical defaults."""
        values: dict[str, Any] = {}
        for feature in NUMERIC_FEATURES:
            bounds = self.profile["numeric_ranges"][feature]
            values[feature] = (float(bounds["min"]) + float(bounds["max"])) / 2
        for feature in CATEGORICAL_FEATURES:
            categories = self.profile["categorical_values"][feature]
            values[feature] = categories[0] if categories else ""
        return values

    def input_frame(self, values: dict[str, Any]) -> pd.DataFrame:
        """Validate and construct a one-row input frame."""
        row: dict[str, Any] = {}
        for feature in NUMERIC_FEATURES:
            try:
                row[feature] = self._coerce_numeric(values[feature])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{feature} must be numeric.") from exc
        for feature in CATEGORICAL_FEATURES:
            value = str(values.get(feature, "")).strip()
            if not value:
                raise ValueError(f"{feature} is required.")
            row[feature] = value
        return pd.DataFrame([row], columns=FEATURES)

    @staticmethod
    def _coerce_numeric(value: Any) -> float:
        """Convert user-entered numeric values, including simple bracketed scientific notation."""
        if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
            if len(value) != 1:
                raise ValueError("Expected a single numeric value.")
            value = list(value)[0]
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("[") and text.endswith("]"):
                text = text[1:-1].strip()
            text = text.replace(",", "")
            match = re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text)
            if not match:
                raise ValueError(f"Could not parse numeric value: {value!r}")
            return float(text)
        return float(value)

    def domain_warnings(self, input_df: pd.DataFrame) -> list[str]:
        """Warn when inputs are outside observed dataset domains."""
        warnings = []
        for feature in NUMERIC_FEATURES:
            value = float(input_df.iloc[0][feature])
            bounds = self.profile["numeric_ranges"][feature]
            min_value = float(bounds["min"])
            max_value = float(bounds["max"])
            if value < min_value or value > max_value:
                warnings.append(f"{feature}={value:g} is outside the training range [{min_value:g}, {max_value:g}].")
        for feature in CATEGORICAL_FEATURES:
            value = str(input_df.iloc[0][feature])
            allowed = self.profile["categorical_values"][feature]
            if value not in allowed:
                warnings.append(f"{feature}={value} was not present in the training dataset.")
        return warnings

    def predict(self, values: dict[str, Any]) -> dict[str, Any]:
        """Predict ultimate load and deflection, including local SHAP explanations."""
        input_df = self.input_frame(values)
        output: dict[str, Any] = {"inputs": input_df, "domain_warnings": self.domain_warnings(input_df), "targets": {}}
        for target_key, target in TARGETS.items():
            bundle = self.bundles[target_key]
            transformed = transformed_frame(bundle["preprocessor"], input_df)
            predictions = {
                name: float(model.predict(transformed)[0])
                for name, model in bundle["models"].items()
            }
            best_name = bundle["best_model"]
            best_model = bundle["models"][best_name]
            explanation = self._explain(best_model, transformed, input_df, target["label"], target["unit"])
            output["targets"][target_key] = {
                "label": target["label"],
                "unit": target["unit"],
                "xgboost_prediction": predictions["XGBoost"],
                "catboost_prediction": predictions["CatBoost"],
                "best_model": best_name,
                "best_prediction": predictions[best_name],
                "difference_between_models": abs(predictions["XGBoost"] - predictions["CatBoost"]),
                "explanation": explanation,
            }
        return output

    def _explain(self, model, transformed: pd.DataFrame, original_input: pd.DataFrame, target_label: str, unit: str) -> dict[str, Any]:
        explainer = shap.TreeExplainer(model)
        values = np.asarray(explainer.shap_values(transformed))
        if values.ndim == 3:
            values = values[:, :, 0]
        contributions = values[0].astype(float)
        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = float(np.asarray(expected_value).ravel()[0])
        else:
            expected_value = float(expected_value)

        rows = []
        for feature in NUMERIC_FEATURES:
            if feature in transformed.columns:
                contribution = float(contributions[list(transformed.columns).index(feature)])
            else:
                contribution = 0.0
            rows.append(
                {
                    "Feature": feature,
                    "Input": float(original_input.iloc[0][feature]),
                    "SHAP Contribution": contribution,
                    "Effect": "Increased" if contribution >= 0 else "Reduced",
                }
            )

        for feature in CATEGORICAL_FEATURES:
            prefix = f"{feature}_"
            contribution = float(
                sum(contributions[index] for index, name in enumerate(transformed.columns) if name.startswith(prefix))
            )
            rows.append(
                {
                    "Feature": feature,
                    "Input": str(original_input.iloc[0][feature]),
                    "SHAP Contribution": contribution,
                    "Effect": "Increased" if contribution >= 0 else "Reduced",
                }
            )

        sorted_rows = sorted(rows, key=lambda row: abs(row["SHAP Contribution"]), reverse=True)
        top = sorted_rows[0]
        second = sorted_rows[1]
        prediction = float(model.predict(transformed)[0])
        text = (
            f"The best model predicts {target_label.lower()} as {prediction:.2f} {unit}. "
            f"{top['Feature']} had the strongest "
            f"{'positive' if top['SHAP Contribution'] >= 0 else 'negative'} influence relative to the model baseline. "
            f"{second['Feature']} was the second strongest contributor and "
            f"{'increased' if second['SHAP Contribution'] >= 0 else 'reduced'} the prediction."
        )
        return {
            "baseline": expected_value,
            "prediction": prediction,
            "contributions": rows,
            "text": text,
        }
