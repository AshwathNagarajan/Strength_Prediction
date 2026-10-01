from datetime import datetime, timezone
import logging

from fastapi import APIRouter, HTTPException

from app.core.config import DOMAIN_FILE, GLOBAL_SHAP_FILE, LATEST_PREDICTION_FILE, PREDICTION_HISTORY_FILE
from app.explainability.shap_service import explain, global_explain
from app.hf.explanation_service import deterministic_prediction_explanation, explain_verified
from app.ml.prediction import predict
from app.schemas.prediction import PredictionRequest
from app.utils.files import atomic_json, read_json
from app.services.dataset_service import current_dataset_path, load_schema
from app.ml.data_loader import load_table

router = APIRouter(prefix="/api", tags=["prediction"])
logger = logging.getLogger(__name__)

@router.post("/predict")
def run(request: PredictionRequest):
    try:
        domain = read_json(DOMAIN_FILE)
        if not domain: raise FileNotFoundError("Feature metadata is missing. Train models first.")
        result = predict(request.features, domain)
        shap_result = explain(request.features) if request.explain else None
        fallback = deterministic_prediction_explanation(result, shap_result)
        output = {**result, "explanation": shap_result, "natural_language_explanation": explain_verified({"prediction": result, "shap": shap_result}, fallback), "uncertainty": {"calibrated": False, "ensemble_prediction_spread": result.get("ensemble_prediction_spread", {}), "message": "Prediction uncertainty has not been calibrated. Ensemble spread is descriptive only."}}
        entry = {"timestamp": datetime.now(timezone.utc).isoformat(), "inputs": request.features, "model": result["model_name"], "output": result["predictions"], "domain_warnings": result["warnings"]}
        history = read_json(PREDICTION_HISTORY_FILE, [])
        atomic_json(PREDICTION_HISTORY_FILE, [*history[-199:], entry])
        atomic_json(LATEST_PREDICTION_FILE, output)
        logger.info("Prediction completed with model=%s domain=%s", result["model_name"], result["domain_analysis"]["overall"])
        return output
    except Exception as exc: raise HTTPException(400, str(exc)) from exc

@router.post("/explain/local")
def local(request: PredictionRequest):
    try: return explain(request.features)
    except Exception as exc: raise HTTPException(400, str(exc)) from exc

@router.get("/explain/global")
def global_explanation():
    try:
        schema = load_schema()
        data = load_table(current_dataset_path())
        sample = data[schema.inputs].dropna().sample(n=min(100, len(data)), random_state=42)
        output = global_explain(sample)
        atomic_json(GLOBAL_SHAP_FILE, output)
        return output
    except Exception as exc: raise HTTPException(400, f"Global SHAP failed: {exc}") from exc


@router.get("/prediction/history")
def history():
    return read_json(PREDICTION_HISTORY_FILE, [])


@router.delete("/prediction/history")
def clear_history():
    atomic_json(PREDICTION_HISTORY_FILE, [])
    return {"cleared": True}
