from __future__ import annotations

import shutil
import logging
from pathlib import Path

from fastapi import UploadFile

from app.core.config import DATASET_STATE_FILE, RAW_DIR, SCHEMA_FILE
from app.ml.data_loader import load_table, preview_table
from app.ml.validator import quality_report
from app.schemas.dataset import DatasetSchema
from app.utils.files import atomic_json, read_json

logger = logging.getLogger(__name__)


def current_dataset_path() -> Path:
    state = read_json(DATASET_STATE_FILE, {})
    path = Path(state.get("path", ""))
    if not path.exists(): raise FileNotFoundError("No dataset configured.")
    return path


def save_upload(file: UploadFile) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".csv", ".xlsx", ".xls"}: raise ValueError("Only CSV and XLSX files are supported.")
    destination = RAW_DIR / Path(file.filename or f"dataset{suffix}").name
    with destination.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)
    data = load_table(destination)
    atomic_json(DATASET_STATE_FILE, {"path": str(destination.resolve()), "filename": destination.name})
    logger.info("Dataset uploaded filename=%s rows=%s columns=%s", destination.name, len(data), len(data.columns))
    return {**preview_table(data), "quality": quality_report(data)}


def save_schema(schema: DatasetSchema) -> dict:
    schema.validate_roles()
    data = load_table(current_dataset_path())
    missing = sorted(set(item.name for item in schema.columns) - set(data.columns))
    if missing: raise ValueError("Schema columns not found in dataset: " + ", ".join(missing))
    atomic_json(SCHEMA_FILE, schema.model_dump())
    logger.info("Dataset schema saved inputs=%s targets=%s", len(schema.inputs), len(schema.targets))
    return schema.model_dump()


def load_schema() -> DatasetSchema:
    value = read_json(SCHEMA_FILE)
    if not value: raise FileNotFoundError("No dataset schema configured.")
    return DatasetSchema.model_validate(value)
