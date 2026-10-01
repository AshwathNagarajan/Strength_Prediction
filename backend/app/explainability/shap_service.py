from __future__ import annotations

import numpy as np
import pandas as pd

from app.ml.model_registry import ModelRegistry


def _dense(value):
    return value.toarray() if hasattr(value, "toarray") else np.asarray(value)


def _target_models(model, target_count: int):
    if hasattr(model, "estimators_"): return list(model.estimators_)
    return [model] * target_count


def _values(model, transformed):
    import shap
    try:
        explainer = shap.TreeExplainer(model)
        values = _dense(explainer.shap_values(transformed))
        base = np.asarray(explainer.expected_value).reshape(-1)
    except Exception:
        background = _dense(transformed)
        explainer = shap.Explainer(model.predict, background)
        explanation = explainer(background)
        values = np.asarray(explanation.values)
        base = np.asarray(explanation.base_values).reshape(-1)
    if values.ndim == 1: values = values[None, :]
    return values, base


def explain(features: dict) -> dict:
    bundle = ModelRegistry().load()
    pipeline, metadata = bundle["pipeline"], bundle["metadata"]
    frame = pd.DataFrame([{name: features[name] for name in metadata["inputs"]}])
    transformed = pipeline.named_steps["preprocessor"].transform(frame)
    names = pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
    contributions, bases = {}, {}
    for index, (target, model) in enumerate(zip(metadata["targets"], _target_models(pipeline.named_steps["model"], len(metadata["targets"])))):
        values, base = _values(model, transformed)
        vector = values[0] if values.shape[0] == 1 else values.reshape(-1, values.shape[-1])[0]
        rows = [{"feature": names[i], "value": float(vector[i])} for i in range(min(len(names), len(vector)))]
        contributions[target] = sorted(rows, key=lambda row: abs(row["value"]), reverse=True)
        bases[target] = float(base[0]) if base.size else None
    return {"base_values": bases, "feature_contributions": contributions}


def global_explain(frame: pd.DataFrame) -> dict:
    bundle = ModelRegistry().load()
    pipeline, metadata = bundle["pipeline"], bundle["metadata"]
    transformed = pipeline.named_steps["preprocessor"].transform(frame[metadata["inputs"]])
    names = pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
    per_target = {}
    aggregate = np.zeros(len(names), dtype=float)
    for target, model in zip(metadata["targets"], _target_models(pipeline.named_steps["model"], len(metadata["targets"]))):
        values, _ = _values(model, transformed)
        matrix = values.reshape(-1, values.shape[-1])
        scores = np.mean(np.abs(matrix), axis=0)
        aggregate[:len(scores)] += scores
        per_target[target] = sorted([{"feature": names[i], "mean_abs_shap": float(scores[i])} for i in range(min(len(names), len(scores)))], key=lambda row: row["mean_abs_shap"], reverse=True)
    aggregate /= max(1, len(metadata["targets"]))
    combined = sorted([{"feature": names[i], "mean_abs_shap": float(aggregate[i])} for i in range(len(names))], key=lambda row: row["mean_abs_shap"], reverse=True)
    return {"importance": combined, "targets": per_target, "caution": "SHAP values describe how the trained model used each feature. They should not be interpreted as proof of physical causality."}
