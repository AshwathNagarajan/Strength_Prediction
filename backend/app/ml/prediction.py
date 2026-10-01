from __future__ import annotations

import numpy as np
import pandas as pd

from app.ml.model_registry import ModelRegistry


def analyze_domain(features: dict, domain: dict) -> dict:
    details, overall, similarities = {}, "IN_DOMAIN", []
    for name, meta in domain.items():
        value = features.get(name)
        if meta["kind"] == "numeric":
            numeric = float(value)
            if numeric < meta["min"] or numeric > meta["max"]:
                status, overall = "OUTSIDE_TRAINING_RANGE", "OUTSIDE_TRAINING_RANGE"
            elif numeric < meta["q05"] or numeric > meta["q95"]:
                status = "NEAR_BOUNDARY"
                if overall == "IN_DOMAIN": overall = status
            else:
                status = "IN_DOMAIN"
            details[name] = {"status": status, "value": numeric, "min": meta["min"], "max": meta["max"]}
            span = meta["max"] - meta["min"]
            similarities.append(max(0.0, min(1.0, 2 * min(numeric - meta["min"], meta["max"] - numeric) / span)) if span else 1.0)
        else:
            status = "IN_DOMAIN" if str(value) in meta["unique_values"] else "OUTSIDE_TRAINING_RANGE"
            if status != "IN_DOMAIN": overall = status
            details[name] = {"status": status, "value": value, "allowed": meta["unique_values"]}
            similarities.append(1.0 if status == "IN_DOMAIN" else 0.0)
    score = float(np.mean(similarities)) if similarities else 0.0
    label = "High domain similarity" if score >= 0.4 and overall == "IN_DOMAIN" else "Moderate domain similarity" if score >= 0.15 and overall != "OUTSIDE_TRAINING_RANGE" else "Near experimental boundary" if overall != "OUTSIDE_TRAINING_RANGE" else "Outside experimental domain"
    return {"overall": overall, "features": details, "similarity_score": score, "reliability_label": label}


def predict(features: dict, domain: dict, include_ensemble: bool = True) -> dict:
    bundle = ModelRegistry().load()
    metadata = bundle["metadata"]
    missing = [name for name in metadata["inputs"] if name not in features]
    if missing:
        raise ValueError("Missing input features: " + ", ".join(missing))
    unknown = sorted(set(features) - set(metadata["inputs"]))
    if unknown: raise ValueError("Unknown input features: " + ", ".join(unknown))
    for name in metadata["inputs"]:
        meta, value = domain[name], features[name]
        if meta["kind"] == "numeric":
            try: numeric = float(value)
            except (TypeError, ValueError) as exc: raise ValueError(f"{name} must be numeric.") from exc
            if not np.isfinite(numeric): raise ValueError(f"{name} must be finite.")
            if any(token in name.lower() for token in ("percent", "pct", "%")) and not 0 <= numeric <= 100:
                raise ValueError(f"{name} must be between 0 and 100 percent.")
        elif str(value) not in set(map(str, meta["unique_values"])):
            raise ValueError(f"Invalid category for {name}: {value}")
    frame = pd.DataFrame([{name: features[name] for name in metadata["inputs"]}])
    raw = np.asarray(bundle["pipeline"].predict(frame)).reshape(-1)
    predictions = {target: float(raw[index]) for index, target in enumerate(metadata["targets"])}
    ensemble_values = []
    for candidate in bundle.get("models", {}).values() if include_ensemble else []:
        try: ensemble_values.append(np.asarray(candidate.predict(frame)).reshape(-1))
        except Exception: continue
    model_spread = {}
    if len(ensemble_values) >= 2:
        matrix = np.vstack(ensemble_values)
        model_spread = {target: float(np.std(matrix[:, index])) for index, target in enumerate(metadata["targets"])}
    domain_result = analyze_domain(features, domain)
    warnings = [] if domain_result["overall"] == "IN_DOMAIN" else ["One or more values are near or outside the experimental domain."]
    return {"predictions": predictions, "model_name": metadata["active_model"], "model_metrics": metadata["metrics"][metadata["active_model"]], "domain_analysis": domain_result, "ensemble_prediction_spread": model_spread, "warnings": warnings}
