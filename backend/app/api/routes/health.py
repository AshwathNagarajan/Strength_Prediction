from fastapi import APIRouter

from app.core.config import DATASET_STATE_FILE, MODEL_METADATA_FILE, SCHEMA_FILE

router = APIRouter(tags=["health"])

@router.get("/api/health")
def health():
    return {
        "status": "ok",
        "dataset_configured": DATASET_STATE_FILE.exists(),
        "schema_configured": SCHEMA_FILE.exists(),
        "model_trained": MODEL_METADATA_FILE.exists(),
        "python": "3.12 supported",
    }
