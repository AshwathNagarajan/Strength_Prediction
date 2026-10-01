from fastapi import APIRouter, HTTPException

from app.ml.model_registry import ModelRegistry
from app.utils.files import json_safe

router = APIRouter(prefix="/api/models", tags=["models"])

@router.get("")
def versions():
    return json_safe(ModelRegistry().list_versions())

@router.get("/best")
def best():
    value = ModelRegistry().metadata()
    if not value: raise HTTPException(404, "No trained model is available.")
    return json_safe(value)
