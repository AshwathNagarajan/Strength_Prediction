from __future__ import annotations

import csv
import io
import json

from fastapi import APIRouter, HTTPException, Response

from app.core.config import GLOBAL_SHAP_FILE, LATEST_OPTIMIZATION_FILE, LATEST_PREDICTION_FILE
from app.ml.model_registry import ModelRegistry
from app.utils.files import read_json

router = APIRouter(prefix="/api/reports", tags=["reports"])

@router.get("/model-comparison")
def comparison(format: str = "json"):
    metadata = ModelRegistry().metadata()
    if not metadata: raise HTTPException(404, "No model metrics are available.")
    rows = []
    for model, result in metadata["metrics"].items():
        if result["status"] == "trained":
            rows.append({"model": model, "cv_r2": result["cv_r2"], "training_time": result["training_time"], "metrics": result["metrics"]})
    if format == "json": return rows
    if format != "csv": raise HTTPException(400, "Format must be json or csv.")
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["model", "target", "r2", "mae", "rmse", "cv_r2", "training_time"])
    for row in rows:
        for target, metrics in row["metrics"].items(): writer.writerow([row["model"], target, metrics["r2"], metrics["mae"], metrics["rmse"], row["cv_r2"], row["training_time"]])
    return Response(buffer.getvalue(), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=model-comparison.csv"})


def _json_download(filename: str, value):
    return Response(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False), media_type="application/json", headers={"Content-Disposition": f"attachment; filename={filename}"})


@router.get("/prediction")
def prediction_report(format: str = "json"):
    value = read_json(LATEST_PREDICTION_FILE)
    if not value: raise HTTPException(404, "No prediction result is available.")
    if format != "json": raise HTTPException(400, "Prediction export supports JSON.")
    return _json_download("prediction-result.json", value)


@router.get("/optimization")
def optimization_report(format: str = "json"):
    value = read_json(LATEST_OPTIMIZATION_FILE)
    if not value: raise HTTPException(404, "No optimization result is available.")
    if format == "json": return _json_download("optimization-result.json", value)
    if format != "csv": raise HTTPException(400, "Format must be json or csv.")
    rows = [value.get("best_solution"), *value.get("alternatives", [])]
    rows = [row for row in rows if row]
    if not rows: raise HTTPException(404, "No feasible optimization rows are available.")
    feature_names = sorted({key for row in rows for key in row["features"]})
    target_names = sorted({key for row in rows for key in row["predictions"]})
    buffer = io.StringIO(); writer = csv.writer(buffer)
    writer.writerow([*feature_names, *target_names, "feasible", "domain_reliability"])
    for row in rows: writer.writerow([*[row["features"].get(key) for key in feature_names], *[row["predictions"].get(key) for key in target_names], row["feasible"], row["domain_reliability"]])
    return Response(buffer.getvalue(), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=feasible-solutions.csv"})


@router.get("/feature-importance")
def feature_importance(format: str = "csv"):
    value = read_json(GLOBAL_SHAP_FILE)
    if not value: raise HTTPException(404, "Global SHAP data is unavailable. Generate it first.")
    if format == "json": return _json_download("shap-feature-importance.json", value)
    if format != "csv": raise HTTPException(400, "Format must be json or csv.")
    buffer = io.StringIO(); writer = csv.writer(buffer); writer.writerow(["feature", "mean_abs_shap"])
    for row in value["importance"]: writer.writerow([row["feature"], row["mean_abs_shap"]])
    return Response(buffer.getvalue(), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=shap-feature-importance.csv"})
