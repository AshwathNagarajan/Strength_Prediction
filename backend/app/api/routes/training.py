import logging
from fastapi import APIRouter, HTTPException

from app.core.config import DOMAIN_FILE, TRAINING_STATUS_FILE
from app.ml.data_loader import load_table
from app.ml.training import train_models
from app.schemas.dataset import TrainingRequest
from app.services.dataset_service import current_dataset_path, load_schema
from app.utils.files import atomic_json, json_safe, read_json

router = APIRouter(prefix="/api/training", tags=["training"])
logger = logging.getLogger(__name__)

@router.post("/train")
@router.post("/start")
def start(request: TrainingRequest):
    try:
        atomic_json(TRAINING_STATUS_FILE, {"state": "preprocessing", "message": "Loading and validating dataset"})
        logger.info("Training started mode=%s models=%s", request.mode, request.models or "all available")
        schema = load_schema().validate_roles()
        atomic_json(TRAINING_STATUS_FILE, {"state": "training", "message": "Training candidate models"})
        def update_status(state: str, message: str): atomic_json(TRAINING_STATUS_FILE, {"state": state, "message": message})
        result = train_models(load_table(current_dataset_path()), schema.inputs, schema.targets, request.test_size, request.cv_folds, request.mode, request.models, update_status)
        atomic_json(DOMAIN_FILE, result.pop("feature_domain"))
        atomic_json(TRAINING_STATUS_FILE, {"state": "completed", "message": "Training completed", "result": result})
        logger.info("Training completed active_model=%s", result["active_model"])
        return json_safe(result)
    except Exception as exc:
        logger.exception("Training failed")
        atomic_json(TRAINING_STATUS_FILE, {"state": "failed", "message": str(exc)})
        raise HTTPException(400, str(exc)) from exc

@router.get("/status")
def status():
    return read_json(TRAINING_STATUS_FILE, {"state": "idle", "message": "No training run started"})

@router.get("/results")
def results():
    from app.ml.model_registry import ModelRegistry
    value = ModelRegistry().metadata()
    if not value: raise HTTPException(404, "No trained model is available.")
    return json_safe(value)
