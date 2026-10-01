"""FastAPI backend for the Vite React frontend."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.optimizer import MaterialOptimizer, OptimizationRequest
from src.performance_config import PERFORMANCE_SHAP_DIR, TARGETS
from src.performance_predictor import PerformancePredictor


app = FastAPI(title="Steel-Concrete Performance AI API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = PerformancePredictor()
optimizer = MaterialOptimizer(predictor)


class PredictionRequest(BaseModel):
    inputs: dict[str, Any]
    explain: bool = True


class OptimizeRequest(BaseModel):
    target_ultimate_load: float = Field(gt=0)
    target_deflection: float | None = Field(default=None, gt=0)
    max_deflection: float | None = Field(default=None, gt=0)
    base_inputs: dict[str, Any] | None = None
    max_results: int = Field(default=10, ge=1, le=25)
    candidate_count: int = Field(default=800, ge=50, le=3000)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    return {
        "metadata": predictor.metadata,
        "profile": predictor.profile,
        "targets": TARGETS,
        "default_inputs": predictor.default_inputs(),
        "display_names": {
            "Slab Thickness hc (mm)": "Beam Width (mm)",
        },
    }


@app.get("/global-importance")
def global_importance() -> dict[str, Any]:
    """Return saved global SHAP importance values for each target/model."""
    output: dict[str, Any] = {}
    for target_key in TARGETS:
        output[target_key] = {}
        for model_name in ["xgboost", "catboost"]:
            path = PERFORMANCE_SHAP_DIR / f"{target_key}_{model_name}_shap_importance.csv"
            rows = []
            if path.exists():
                with path.open("r", encoding="utf-8") as file:
                    header = file.readline().strip().split(",")
                    for line in file:
                        values = line.strip().split(",")
                        if len(values) != len(header):
                            continue
                        row = dict(zip(header, values))
                        rows.append({"feature": row["feature"], "mean_abs_shap": float(row["mean_abs_shap"])})
            output[target_key][model_name] = rows
    return output


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    try:
        if request.explain:
            result = predictor.predict(request.inputs)
        else:
            result = predictor.predict_without_explanation(request.inputs)
        result.pop("inputs", None)
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/optimize")
def optimize(request: OptimizeRequest) -> dict[str, Any]:
    try:
        return optimizer.optimize(
            OptimizationRequest(
                target_ultimate_load=request.target_ultimate_load,
                target_deflection=request.target_deflection,
                max_deflection=request.max_deflection,
                base_inputs=request.base_inputs,
                max_results=request.max_results,
                candidate_count=request.candidate_count,
            )
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
