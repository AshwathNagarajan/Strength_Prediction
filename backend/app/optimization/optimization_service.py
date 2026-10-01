from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from app.ml.prediction import predict
from app.schemas.dataset import DatasetSchema
from app.schemas.optimization import OptimizationRequest


def _satisfies(value: float, operator: str, target: float, tolerance: float) -> bool:
    if operator == ">=": return value + tolerance >= target
    if operator == "<=": return value - tolerance <= target
    return abs(value - target) <= tolerance


@dataclass
class CandidateSpace:
    schema: DatasetSchema
    domain: dict
    request: OptimizationRequest

    def __post_init__(self):
        self.columns = {item.name: item for item in self.schema.columns}
        unknown = (set(self.request.fixed_features) | set(self.request.bounds)) - set(self.schema.inputs)
        if unknown: raise ValueError("Unknown optimization features: " + ", ".join(sorted(unknown)))
        self.warnings: list[str] = []

    def bounds(self, name: str) -> tuple[float, float]:
        meta = self.domain[name]
        requested = self.request.bounds.get(name, (meta["min"], meta["max"]))
        low, high = max(float(requested[0]), meta["min"]), min(float(requested[1]), meta["max"])
        if tuple(map(float, requested)) != (low, high): self.warnings.append(f"Requested limits for {name} exceeded the experimental domain and were clipped.")
        if low > high: raise ValueError(f"Invalid bounds for {name} after applying dataset limits.")
        return low, high

    def validate_value(self, name: str, value: Any) -> None:
        meta = self.domain[name]
        if meta["kind"] == "categorical":
            if str(value) not in set(map(str, meta["unique_values"])): raise ValueError(f"Invalid category for {name}: {value}")
        elif not np.isfinite(float(value)) or not meta["min"] <= float(value) <= meta["max"]:
            raise ValueError(f"Fixed value for {name} is outside its experimental range.")

    def sample_value(self, name: str, rng: np.random.Generator) -> Any:
        meta, config = self.domain[name], self.columns[name]
        if name in self.request.fixed_features:
            value = self.request.fixed_features[name]; self.validate_value(name, value); return value
        if not config.optimizable: return meta["unique_values"][0] if meta["kind"] == "categorical" else meta["median"]
        allowed = config.allowed_values or meta.get("unique_values", [])
        if meta["kind"] == "categorical" or config.variable_type in {"categorical", "discrete"} or config.discrete:
            if not allowed: raise ValueError(f"No allowed values configured for discrete feature {name}.")
            return rng.choice(allowed).item()
        low, high = self.bounds(name)
        value = float(rng.uniform(low, high))
        if config.step: value = low + round((value - low) / config.step) * config.step
        return float(np.clip(value, low, high))

    def sample(self, rng: np.random.Generator) -> dict[str, Any]:
        return {name: self.sample_value(name, rng) for name in self.schema.inputs}

    def mutate(self, row: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
        result = dict(row)
        for name in self.schema.inputs:
            if name not in self.request.fixed_features and self.columns[name].optimizable and rng.random() < self.request.mutation_rate:
                result[name] = self.sample_value(name, rng)
        return result


def evaluate(features: dict[str, Any], request: OptimizationRequest, domain: dict) -> dict:
    result = predict(features, domain, include_ensemble=False)
    checks, violations = [], []
    for target in request.targets:
        if target.target not in result["predictions"]: raise ValueError(f"Unknown target: {target.target}")
        actual = result["predictions"][target.target]
        satisfied = _satisfies(actual, target.operator, target.value, target.tolerance)
        checks.append({"target": target.target, "operator": target.operator, "required": target.value, "predicted": actual, "satisfied": satisfied})
        if not satisfied: violations.append(f"Predicted {target.target} does not satisfy {target.operator} {target.value}.")
    objective_values = []
    for objective in request.objectives:
        source = features if objective.source == "feature" else result["predictions"]
        if objective.field not in source: raise ValueError(f"Unknown objective field: {objective.field}")
        objective_values.append(float(source[objective.field]) * objective.weight * (-1 if objective.direction == "maximize" else 1))
    deviation = sum(abs(item["predicted"] - item["required"]) for item in checks)
    if not objective_values: objective_values = [deviation]
    elif request.algorithm == "nsga2" and len(objective_values) == 1: objective_values.append(deviation)
    similarity = result["domain_analysis"]["reliability_label"]
    return {"features": features, "predictions": result["predictions"], "feasible": not violations, "violations": violations, "constraints": checks, "objectives": objective_values, "score": -sum(objective_values), "target_deviation": deviation, "domain_reliability": similarity, "domain_similarity_score": result["domain_analysis"]["similarity_score"]}


def _baseline(space, request, rng):
    rows = [evaluate(space.sample(rng), request, space.domain) for _ in range(request.candidate_count)]
    return rows, len(rows)


def _genetic(space, request, rng):
    population = [space.sample(rng) for _ in range(request.population_size)]
    archive, evaluated = [], 0
    for _ in range(request.generations):
        scored = [evaluate(row, request, space.domain) for row in population]
        archive.extend(scored); evaluated += len(scored)
        ranked = sorted(scored, key=lambda row: (not row["feasible"], -row["score"]))
        parents = ranked[:max(2, len(ranked) // 3)]
        population = [dict(row["features"]) for row in parents[:2]]
        while len(population) < request.population_size:
            first, second = rng.choice(parents, 2, replace=True)
            child = {name: (first["features"][name] if rng.random() < .5 else second["features"][name]) for name in space.schema.inputs}
            population.append(space.mutate(child, rng))
    return archive, evaluated


def _dominates(a, b):
    return all(x <= y for x, y in zip(a["objectives"], b["objectives"])) and any(x < y for x, y in zip(a["objectives"], b["objectives"]))


def _pareto(rows):
    feasible = [row for row in rows if row["feasible"]]
    return [row for row in feasible if not any(_dominates(other, row) for other in feasible if other is not row)]


def _fronts(rows):
    remaining, fronts = list(rows), []
    while remaining:
        front = [row for row in remaining if not any(_dominates(other, row) for other in remaining if other is not row)]
        fronts.append(front)
        front_ids = {id(row) for row in front}
        remaining = [row for row in remaining if id(row) not in front_ids]
    return fronts


def _crowding(front):
    distances = {id(row): 0.0 for row in front}
    if len(front) <= 2:
        return {id(row): float("inf") for row in front}
    for objective_index in range(len(front[0]["objectives"])):
        ordered = sorted(front, key=lambda row: row["objectives"][objective_index])
        distances[id(ordered[0])] = distances[id(ordered[-1])] = float("inf")
        low, high = ordered[0]["objectives"][objective_index], ordered[-1]["objectives"][objective_index]
        if high == low: continue
        for index in range(1, len(ordered) - 1):
            distances[id(ordered[index])] += (ordered[index + 1]["objectives"][objective_index] - ordered[index - 1]["objectives"][objective_index]) / (high - low)
    return distances


def _nsga2(space, request, rng):
    population = [space.sample(rng) for _ in range(request.population_size)]
    scored = [evaluate(row, request, space.domain) for row in population]
    archive, evaluated = list(scored), len(scored)
    for _ in range(request.generations):
        parents = _pareto(scored) or scored
        offspring = []
        while len(offspring) < request.population_size:
            first, second = rng.choice(parents, 2, replace=True)
            child = {name: (first["features"][name] if rng.random() < .5 else second["features"][name]) for name in space.schema.inputs}
            offspring.append(space.mutate(child, rng))
        children = [evaluate(row, request, space.domain) for row in offspring]
        archive.extend(children); evaluated += len(children)
        combined, next_generation = scored + children, []
        for front in _fronts(combined):
            if len(next_generation) + len(front) <= request.population_size: next_generation.extend(front)
            else:
                distances = _crowding(front)
                next_generation.extend(sorted(front, key=lambda row: distances[id(row)], reverse=True)[:request.population_size-len(next_generation)])
                break
        scored = next_generation
    return archive, evaluated


def _diverse(rows, limit):
    selected = []
    for row in rows:
        def near(existing):
            for name, value in row["features"].items():
                other = existing["features"][name]
                if isinstance(value, (int, float)) and isinstance(other, (int, float)):
                    if abs(float(value)-float(other)) / max(abs(float(value)), abs(float(other)), 1.0) > 0.01: return False
                elif value != other: return False
            return True
        if not any(near(existing) for existing in selected): selected.append(row)
        if len(selected) >= limit: break
    return selected


def optimize(request: OptimizationRequest, schema: DatasetSchema, domain: dict) -> dict:
    rng, space = np.random.default_rng(request.random_seed), CandidateSpace(schema, domain, request)
    if request.algorithm == "grid": rows, evaluated = _baseline(space, request, rng)
    elif request.algorithm == "genetic": rows, evaluated = _genetic(space, request, rng)
    else: rows, evaluated = _nsga2(space, request, rng)
    feasible = sorted([row for row in rows if row["feasible"]], key=lambda row: tuple(row["objectives"]))
    pareto = _pareto(rows) if request.algorithm == "nsga2" else []
    selected = _diverse(sorted(pareto or feasible, key=lambda row: tuple(row["objectives"])), request.max_results + 1)
    warnings = list(space.warnings)
    if any(row["domain_reliability"] == "Near experimental boundary" for row in selected): warnings.append("One or more recommendations are near an experimental boundary.")
    return {"algorithm": request.algorithm, "objective": [item.model_dump() for item in request.objectives], "best_solution": selected[0] if selected else None, "alternatives": selected[1:], "pareto_front": _diverse(pareto, 100), "selection_rule": "Suggested balanced solution based on lexicographic objective order and Pareto rank." if request.algorithm == "nsga2" else "Ranked by configured objective, then target deviation.", "evaluated_candidates": evaluated, "feasible_candidates": len(feasible), "warnings": sorted(set(warnings)), "progress": {"iterations": request.generations if request.algorithm != "grid" else 1, "candidates_evaluated": evaluated, "feasible_candidates": len(feasible)}}
