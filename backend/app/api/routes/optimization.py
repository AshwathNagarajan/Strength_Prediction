import logging
from fastapi import APIRouter, HTTPException

from app.core.config import DOMAIN_FILE
from app.core.config import LATEST_OPTIMIZATION_FILE
from app.hf.explanation_service import explain_verified
from app.optimization.optimization_service import optimize
from app.schemas.optimization import OptimizationRequest
from app.services.dataset_service import load_schema
from app.utils.files import atomic_json, read_json
from app.hf.constraint_parser import parse_request
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/optimization", tags=["optimization"])
logger = logging.getLogger(__name__)

@router.post("")
@router.post("/run")
def run(request: OptimizationRequest):
    try:
        logger.info("Optimization started algorithm=%s candidates=%s", request.algorithm, request.candidate_count)
        result = optimize(request, load_schema(), read_json(DOMAIN_FILE, {}))
        fallback = (
            f"The {request.algorithm} search evaluated {result['evaluated_candidates']} candidates and found "
            f"{result['feasible_candidates']} feasible designs within the experimental feature ranges."
        )
        result["natural_language_explanation"] = explain_verified(result, fallback)
        atomic_json(LATEST_OPTIMIZATION_FILE, result)
        logger.info("Optimization completed algorithm=%s evaluated=%s feasible=%s", request.algorithm, result["evaluated_candidates"], result["feasible_candidates"])
        return result
    except Exception as exc: raise HTTPException(400, str(exc)) from exc


class NaturalLanguageRequest(BaseModel):
    text: str = Field(min_length=5, max_length=4000)


@router.post("/parse-request")
def parse_natural(request: NaturalLanguageRequest):
    try:
        schema = load_schema()
        parsed = parse_request(request.text, schema.targets, schema.inputs)
        unresolved = [name for name, value in parsed["request"]["fixed_features"].items() if value is None]
        if unresolved: parsed["warnings"].append("Provide current values for fixed features: " + ", ".join(unresolved))
        return parsed
    except Exception as exc: raise HTTPException(400, str(exc)) from exc


@router.get("/config")
def config():
    try:
        schema, domain = load_schema(), read_json(DOMAIN_FILE, {})
        return {"features": [item.model_dump() | {"domain": domain.get(item.name)} for item in schema.columns if item.role == "input"], "targets": [item.model_dump() for item in schema.columns if item.role == "target"], "algorithms": ["grid", "genetic", "nsga2"]}
    except Exception as exc: raise HTTPException(404, str(exc)) from exc
