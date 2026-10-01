from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from app.ml.data_loader import load_table
from app.ml.prediction import analyze_domain, predict
from app.ml.validator import domain_metadata, quality_report
from app.schemas.dataset import ColumnConfig, DatasetSchema
from app.schemas.optimization import OptimizationRequest, TargetConstraint
from app.optimization.optimization_service import CandidateSpace, _pareto, evaluate
from app.hf.constraint_parser import parse_request
from app.main import app
from app.utils.files import json_safe
from fastapi.testclient import TestClient
from app.ml.training import train_models
from app.ml.model_registry import ModelRegistry


def test_csv_loading_and_quality(tmp_path):
    path = tmp_path / "beam.csv"
    pd.DataFrame({"strength": [20, 30, 30], "load": [100, 150, 150]}).to_csv(path, index=False)
    data = load_table(path)
    report = quality_report(data)
    assert data.shape == (3, 2)
    assert report["duplicate_rows"] == 1


def test_xlsx_loading(tmp_path):
    path = tmp_path / "beam.xlsx"
    pd.DataFrame({"strength": [20, 30], "load": [100, 150]}).to_excel(path, index=False)
    assert load_table(path).shape == (2, 2)


def test_schema_requires_inputs_and_targets():
    schema = DatasetSchema(
        dataset_name="test",
        columns=[
            ColumnConfig(name="strength", role="input", optimizable=True),
            ColumnConfig(name="load", role="target"),
        ],
    )
    assert schema.validate_roles().inputs == ["strength"]
    assert schema.targets == ["load"]


def test_domain_classification():
    data = pd.DataFrame({"strength": np.arange(20.0, 41.0)})
    domain = domain_metadata(data, ["strength"])
    assert analyze_domain({"strength": 30}, domain)["overall"] == "IN_DOMAIN"
    assert analyze_domain({"strength": 50}, domain)["overall"] == "OUTSIDE_TRAINING_RANGE"


def test_optimization_request_validation():
    request = OptimizationRequest(
        targets=[TargetConstraint(target="load", operator=">=", value=100)],
        candidate_count=50,
    )
    assert request.targets[0].operator == ">="


def test_candidate_space_respects_fixed_and_discrete_values():
    schema = DatasetSchema(dataset_name="beam", columns=[
        ColumnConfig(name="span", role="input", optimizable=False),
        ColumnConfig(name="thickness", role="input", optimizable=True, variable_type="discrete", allowed_values=[6, 8, 10]),
        ColumnConfig(name="load", role="target"),
    ])
    domain = {
        "span": {"kind": "numeric", "min": 1000, "max": 3000, "median": 2000, "unique_values": []},
        "thickness": {"kind": "numeric", "min": 6, "max": 10, "median": 8, "unique_values": [6, 8, 10]},
    }
    request = OptimizationRequest(targets=[TargetConstraint(target="load", operator=">=", value=100)], fixed_features={"span": 2500}, candidate_count=50)
    sample = CandidateSpace(schema, domain, request).sample(np.random.default_rng(42))
    assert sample["span"] == 2500
    assert sample["thickness"] in {6, 8, 10}


def test_pareto_filter_keeps_non_dominated_rows():
    rows = [
        {"feasible": True, "objectives": [1, 2]},
        {"feasible": True, "objectives": [2, 1]},
        {"feasible": True, "objectives": [3, 3]},
    ]
    assert len(_pareto(rows)) == 2


def test_natural_language_parser_has_validated_fallback(monkeypatch):
    monkeypatch.setenv("HF_PROVIDER", "disabled")
    parsed = parse_request(
        "Find at least 500 ultimate load and maximize sustainable replacement.",
        ["ultimate_load"],
        ["sustainable_replacement"],
    )
    assert parsed["parser"] == "deterministic"
    assert parsed["request"]["targets"][0]["value"] == 500


def test_natural_language_parser_separates_multiple_target_clauses(monkeypatch):
    monkeypatch.setenv("HF_PROVIDER", "disabled")
    parsed = parse_request("ultimate load at least 500 kN and deflection below 12 mm", ["ultimate_load", "deflection"], ["fly_ash_pct"])
    targets = {item["target"]: item for item in parsed["request"]["targets"]}
    assert targets["ultimate_load"]["value"] == 500
    assert targets["ultimate_load"]["operator"] == ">="
    assert targets["deflection"]["value"] == 12
    assert targets["deflection"]["operator"] == "<="


def test_natural_language_parser_understands_engineering_target_aliases(monkeypatch):
    monkeypatch.setenv("HF_PROVIDER", "disabled")
    parsed = parse_request("ultimate load at least 500 kN and deflection below 12 mm", ["Target: Capacity Pu (kN)", "Target: Deflection δ (mm)"], ["Fly Ash (%)"])
    targets = {item["target"]: item for item in parsed["request"]["targets"]}
    assert targets["Target: Capacity Pu (kN)"]["value"] == 500
    assert targets["Target: Deflection δ (mm)"]["value"] == 12


def test_api_uses_consistent_response_envelope():
    response = TestClient(app).get("/api/health")
    assert response.status_code == 200
    assert response.json()["success"] is True
    assert response.json()["data"]["status"] == "ok"


def test_json_safe_replaces_non_finite_values():
    assert json_safe({"nan": float("nan"), "infinity": float("inf")}) == {"nan": None, "infinity": None}


def test_tiny_multi_target_training_pipeline(monkeypatch, tmp_path):
    monkeypatch.setattr(ModelRegistry, "save", lambda self, bundle, metadata: tmp_path)
    x = np.linspace(20, 60, 30)
    data = pd.DataFrame({"strength": x, "category": ["A", "B"] * 15, "load": 4*x + 10, "deflection": 20 - 0.1*x})
    result = train_models(data, ["strength", "category"], ["load", "deflection"], 0.2, 3, "quick", ["LinearRegression"])
    assert result["active_model"] == "LinearRegression"
    assert result["metrics"]["LinearRegression"]["status"] == "trained"


def test_prediction_uses_loaded_pipeline_and_domain_guard(monkeypatch):
    pipeline = Pipeline([("model", LinearRegression())]).fit(pd.DataFrame({"strength": [20, 30, 40]}), [100, 150, 200])
    bundle = {"pipeline": pipeline, "metadata": {"inputs": ["strength"], "targets": ["load"], "active_model": "LinearRegression", "metrics": {"LinearRegression": {"status": "trained"}}}}
    monkeypatch.setattr(ModelRegistry, "load", lambda self: bundle)
    domain = {"strength": {"kind": "numeric", "min": 20, "max": 40, "q05": 21, "q95": 39}}
    result = predict({"strength": 30}, domain)
    assert result["predictions"]["load"] == pytest.approx(150)
    assert result["domain_analysis"]["overall"] == "IN_DOMAIN"


def test_constraint_evaluator_reports_infeasible_candidate(monkeypatch):
    monkeypatch.setattr("app.optimization.optimization_service.predict", lambda features, domain, **kwargs: {"predictions": {"load": 90}, "domain_analysis": {"overall": "IN_DOMAIN", "reliability_label": "High domain similarity", "similarity_score": 1.0}})
    request = OptimizationRequest(targets=[TargetConstraint(target="load", operator=">=", value=100)], candidate_count=50)
    result = evaluate({"strength": 30}, request, {})
    assert result["feasible"] is False
    assert result["violations"]


def test_non_finite_optimization_target_is_rejected():
    with pytest.raises(ValidationError):
        OptimizationRequest(targets=[TargetConstraint(target="load", operator=">=", value=float("nan"))])


def test_optimization_request_repairs_common_llm_json_shape():
    request = OptimizationRequest.model_validate({"targets": {"load": {"operator": ">=", "value": 500}}, "fixed_features": ["span"]})
    assert request.targets[0].target == "load"
    assert request.fixed_features == {"span": None}

    scalar_request = OptimizationRequest.model_validate({"targets": {"load": 500}})
    assert scalar_request.targets[0].value == 500


def test_invalid_hf_json_uses_deterministic_fallback(monkeypatch):
    class Client:
        def __init__(self, **kwargs):
            self.chat = self
            self.completions = self

        def create(self, **kwargs):
            message = type("Message", (), {"content": "not valid json"})()
            choice = type("Choice", (), {"message": message})()
            return type("Response", (), {"choices": [choice]})()

    monkeypatch.setenv("HF_PROVIDER", "api"); monkeypatch.setenv("HF_MODEL_ID", "test-model")
    monkeypatch.setattr("huggingface_hub.InferenceClient", Client)
    parsed = parse_request("ultimate load at least 500", ["ultimate_load"], ["fly_ash_pct"])
    assert parsed["parser"] == "deterministic"
    assert parsed["request"]["targets"][0]["value"] == 500


def test_hf_parser_canonicalizes_schema_names_and_rejects_invented_values(monkeypatch):
    class Client:
        def __init__(self, **kwargs):
            self.chat = self
            self.completions = self

        def create(self, **kwargs):
            content = '{"targets":[{"target":"ultimate load","operator":">","value":500}],"fixed_features":{},"bounds":{},"objectives":[],"algorithm":"grid"}'
            message = type("Message", (), {"content": content})()
            choice = type("Choice", (), {"message": message})()
            return type("Response", (), {"choices": [choice]})()

    monkeypatch.setenv("HF_PROVIDER", "api"); monkeypatch.setenv("HF_MODEL_ID", "test-model")
    monkeypatch.setattr("huggingface_hub.InferenceClient", Client)
    parsed = parse_request("ultimate load at least 500", ["ultimate_load"], ["span"])
    assert parsed["parser"] == "hugging_face"
    assert parsed["request"]["targets"][0]["target"] == "ultimate_load"
