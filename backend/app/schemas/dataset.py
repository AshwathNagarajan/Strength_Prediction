from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ColumnConfig(BaseModel):
    name: str
    role: Literal["input", "target", "identifier", "ignore"]
    display_name: str | None = None
    unit: str = ""
    category: Literal["Material", "Geometry", "Connection", "Other"] = "Other"
    optimizable: bool = False
    discrete: bool = False
    variable_type: Literal["continuous", "discrete", "categorical"] | None = None
    step: float | None = Field(default=None, gt=0)
    allowed_values: list[str | float] = Field(default_factory=list)
    editable: bool = True
    sustainability_direction: Literal["maximize", "minimize"] | None = None


class DatasetSchema(BaseModel):
    dataset_name: str
    columns: list[ColumnConfig]

    @property
    def inputs(self) -> list[str]:
        return [item.name for item in self.columns if item.role == "input"]

    @property
    def targets(self) -> list[str]:
        return [item.name for item in self.columns if item.role == "target"]

    def validate_roles(self) -> "DatasetSchema":
        if not self.inputs:
            raise ValueError("At least one input feature is required.")
        if not self.targets:
            raise ValueError("At least one target output is required.")
        names = [item.name for item in self.columns]
        if len(names) != len(set(names)):
            raise ValueError("Column names in the schema must be unique.")
        return self


class TrainingRequest(BaseModel):
    test_size: float = Field(default=0.2, gt=0.05, lt=0.5)
    cv_folds: int = Field(default=5, ge=2, le=10)
    mode: Literal["quick", "standard"] = "quick"
    models: list[str] | None = None
