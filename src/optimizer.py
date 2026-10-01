"""Constrained material optimization using saved performance models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .performance_config import NUMERIC_FEATURES
from .performance_predictor import PerformancePredictor


@dataclass
class OptimizationRequest:
    """User-controlled optimization settings."""

    target_ultimate_load: float
    target_deflection: float | None = None
    max_deflection: float | None = None
    base_inputs: dict[str, Any] | None = None
    max_results: int = 10
    random_state: int = 42
    candidate_count: int = 800


class MaterialOptimizer:
    """Search feasible dataset-bounded combinations for target performance."""

    def __init__(self, predictor: PerformancePredictor | None = None) -> None:
        self.predictor = predictor or PerformancePredictor()
        self.profile = self.predictor.profile

    def optimize(self, request: OptimizationRequest) -> dict[str, Any]:
        """Return feasible high-sustainability combinations."""
        if request.target_ultimate_load <= 0:
            raise ValueError("Target ultimate load must be positive.")
        rng = np.random.default_rng(request.random_state)
        base = self.predictor.default_inputs()
        if request.base_inputs:
            base.update(request.base_inputs)

        candidates = []
        for candidate in self._seed_candidates(base):
            self._evaluate_candidate(candidate, request, candidates)

        for _ in range(max(50, request.candidate_count)):
            candidate = dict(base)
            self._sample_materials(candidate, rng)
            self._evaluate_candidate(candidate, request, candidates)

        candidates.sort(
            key=lambda item: (
                item["reverse_design_error"],
                -item["sustainable_score"],
                item["predicted_deflection"],
            )
        )
        selected = candidates[: max(1, request.max_results)]
        best_gap = abs(selected[0]["load_difference"]) if selected else None
        target_gap_warning = (
            best_gap is not None
            and best_gap > max(0.10 * request.target_ultimate_load, 100.0)
        )
        return {
            "target_ultimate_load": request.target_ultimate_load,
            "target_deflection": request.target_deflection,
            "max_deflection": request.max_deflection,
            "candidate_count": request.candidate_count,
            "feasible_count": len(candidates),
            "recommended": selected[0] if selected else None,
            "results": selected,
            "target_gap_warning": target_gap_warning,
            "constraints": {
                "numeric_ranges": self.profile["numeric_ranges"],
                "categorical_values": self.profile["categorical_values"],
            },
            "method": (
                "Random constrained search inside observed training ranges. "
                "The search also checks common fly-ash replacement levels of 10%, 15%, 20%, and 25% when those values "
                "are supported by the dataset range. "
                "Feasible candidates must meet the target load and optional deflection limit. "
                "They are ranked by closeness to the requested output target, then sustainable material percentage."
            ),
        }

    def _seed_candidates(self, base: dict[str, Any]) -> list[dict[str, Any]]:
        """Create deterministic candidates around review-friendly replacement levels."""
        ranges = self.profile["numeric_ranges"]
        seeds = []
        for fly_ash in [10.0, 15.0, 20.0, 25.0]:
            if not self._inside_range("Fly Ash (%)", fly_ash):
                continue
            candidate = dict(base)
            candidate["Fly Ash (%)"] = fly_ash
            for feature in ["GGBS (%)", "Recycled Aggregate (%)"]:
                bounds = ranges[feature]
                candidate[feature] = float(np.clip(candidate.get(feature, bounds["min"]), bounds["min"], bounds["max"]))
            seeds.append(candidate)
        return seeds

    def _evaluate_candidate(
        self,
        candidate: dict[str, Any],
        request: OptimizationRequest,
        candidates: list[dict[str, Any]],
    ) -> None:
        """Predict and append the candidate when it satisfies reverse-design constraints."""
        result = self.predictor.predict_without_explanation(candidate)
        load = result["targets"]["ultimate_load"]["best_prediction"]
        deflection = result["targets"]["deflection"]["best_prediction"]
        load_error = abs(load - request.target_ultimate_load)
        deflection_error = abs(deflection - request.target_deflection) if request.target_deflection is not None else 0.0
        meets_load = load >= request.target_ultimate_load
        meets_deflection = request.max_deflection is None or deflection <= request.max_deflection
        material_quantities = {
            "Fly Ash (%)": float(candidate["Fly Ash (%)"]),
            "GGBS (%)": float(candidate["GGBS (%)"]),
            "Recycled Aggregate (%)": float(candidate["Recycled Aggregate (%)"]),
        }
        sustainable_score = sum(material_quantities.values())
        material_quantities["Sustainable Material Total (%)"] = sustainable_score

        if meets_load and meets_deflection:
            candidates.append(
                {
                    "inputs": candidate,
                    "material_quantities": material_quantities,
                    "predicted_ultimate_load": load,
                    "predicted_deflection": deflection,
                    "target_ultimate_load": request.target_ultimate_load,
                    "target_deflection": request.target_deflection,
                    "load_difference": load - request.target_ultimate_load,
                    "deflection_difference": (
                        deflection - request.target_deflection if request.target_deflection is not None else None
                    ),
                    "reverse_design_error": self._normalized_error(
                        load_error,
                        request.target_ultimate_load,
                    )
                    + (
                        self._normalized_error(deflection_error, request.target_deflection)
                        if request.target_deflection is not None
                        else 0.0
                    ),
                    "sustainable_score": sustainable_score,
                    "warnings": result["domain_warnings"],
                }
            )

    def _sample_materials(self, candidate: dict[str, Any], rng: np.random.Generator) -> None:
        """Sample material variables within observed domains."""
        numeric_ranges = self.profile["numeric_ranges"]
        categorical_values = self.profile["categorical_values"]

        concrete_grade = str(rng.choice(categorical_values["Concrete Grade"]))
        candidate["Concrete Grade"] = concrete_grade
        grade_strength = self._strength_from_grade(concrete_grade)
        if grade_strength is not None:
            candidate["Compressive Strength fc' (MPa)"] = float(
                np.clip(
                    grade_strength,
                    numeric_ranges["Compressive Strength fc' (MPa)"]["min"],
                    numeric_ranges["Compressive Strength fc' (MPa)"]["max"],
                )
            )

        for feature in ["Fly Ash (%)", "GGBS (%)", "Recycled Aggregate (%)", "w/b Ratio"]:
            bounds = numeric_ranges[feature]
            candidate[feature] = float(rng.uniform(bounds["min"], bounds["max"]))

        # Keep the common steel and connector controls searchable while leaving geometry from base inputs.
        candidate["Steel Grade"] = str(rng.choice(categorical_values["Steel Grade"]))
        candidate["Shear Connector Type"] = str(rng.choice(categorical_values["Shear Connector Type"]))
        if "Yield Strength fy (MPa)" in NUMERIC_FEATURES:
            bounds = numeric_ranges["Yield Strength fy (MPa)"]
            candidate["Yield Strength fy (MPa)"] = float(rng.choice([250, 275, 355, 460, 500, 550, 690]))
            candidate["Yield Strength fy (MPa)"] = float(
                np.clip(candidate["Yield Strength fy (MPa)"], bounds["min"], bounds["max"])
            )

    def _inside_range(self, feature: str, value: float) -> bool:
        bounds = self.profile["numeric_ranges"][feature]
        return bounds["min"] <= value <= bounds["max"]

    @staticmethod
    def _strength_from_grade(grade: str) -> float | None:
        digits = "".join(ch for ch in grade if ch.isdigit())
        return float(digits) if digits else None

    @staticmethod
    def _normalized_error(error: float, target: float | None) -> float:
        if target is None:
            return 0.0
        return abs(error) / max(abs(target), 1e-9)
