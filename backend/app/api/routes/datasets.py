from fastapi import APIRouter, File, HTTPException, UploadFile

from app.ml.data_loader import load_table, preview_table
from app.ml.validator import quality_report
from app.schemas.dataset import DatasetSchema
from app.services.dataset_service import current_dataset_path, load_schema, save_schema, save_upload

router = APIRouter(prefix="/api/dataset", tags=["dataset"])

@router.post("/upload")
def upload(file: UploadFile = File(...)):
    try: return save_upload(file)
    except Exception as exc: raise HTTPException(400, str(exc)) from exc

@router.post("/schema")
def configure(schema: DatasetSchema):
    try: return {"schema": save_schema(schema)}
    except Exception as exc: raise HTTPException(400, str(exc)) from exc

@router.get("/schema")
def schema():
    try: return load_schema().model_dump()
    except Exception as exc: raise HTTPException(404, str(exc)) from exc

@router.get("/quality")
def quality():
    try:
        data = load_table(current_dataset_path())
        return {**preview_table(data), "quality": quality_report(data)}
    except Exception as exc: raise HTTPException(404, str(exc)) from exc

@router.get("/info")
def info():
    try: return preview_table(load_table(current_dataset_path()))
    except Exception as exc: raise HTTPException(404, str(exc)) from exc

@router.get("/ranges")
def ranges():
    from app.core.config import DOMAIN_FILE
    from app.utils.files import read_json
    value = read_json(DOMAIN_FILE)
    if not value: raise HTTPException(404, "Feature ranges are unavailable. Train models first.")
    return value
