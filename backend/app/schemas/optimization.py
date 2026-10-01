from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TargetConstraint(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    target: str
    operator: Literal[">=", "<=", "=="]
    value: float
    tolerance: float = Field(default=0.0, ge=0)


class Objective(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    field: str
    source: Literal["feature", "target"] = "feature"
    direction: Literal["maximize", "minimize"] = "maximize"
    weight: float = Field(default=1.0, gt=0)


class OptimizationRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    targets: list[TargetConstraint]
    fixed_features: dict[str, Any] = Field(default_factory=dict)
    bounds: dict[str, tuple[float, float]] = Field(default_factory=dict)
    objectives: list[Objective] = Field(default_factory=list)
    algorithm: Literal["grid", "genetic", "nsga2"] = "grid"
    candidate_count: int = Field(default=1000, ge=50, le=10000)
    max_results: int = Field(default=10, ge=1, le=100)
    population_size: int = Field(default=60, ge=10, le=500)
    generations: int = Field(default=30, ge=1, le=300)
    mutation_rate: float = Field(default=0.15, ge=0, le=1)
    random_seed: int = 42

    @model_validator(mode="before")
    @classmethod
    def repair_common_llm_shapes(cls, value):
        if not isinstance(value, dict): return value
        repaired = dict(value)
        if isinstance(repaired.get("targets"), dict):
            normalized = []
            for name, constraint in repaired["targets"].items():
                if isinstance(constraint, dict):
                    normalized.append({"target": name, **constraint})
                else:
                    normalized.append({"target": name, "operator": ">=", "value": constraint})
            repaired["targets"] = normalized
        for constraint in repaired.get("targets", []):
            if isinstance(constraint, dict):
                constraint["operator"] = {">": ">=", "<": "<=", "=": "=="}.get(
                    constraint.get("operator"), constraint.get("operator")
                )
        if isinstance(repaired.get("fixed_features"), list):
            repaired["fixed_features"] = {name: None for name in repaired["fixed_features"]}
        return repaired

    @model_validator(mode="after")
    def require_target(self) -> "OptimizationRequest":
        if not self.targets:
            raise ValueError("At least one structural target constraint is required.")
        return self
