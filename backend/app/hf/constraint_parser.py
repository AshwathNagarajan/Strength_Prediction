from __future__ import annotations

import json
import re
import logging
from typing import Any

from app.core.config import hf_settings
from app.schemas.optimization import Objective, OptimizationRequest, TargetConstraint

logger = logging.getLogger(__name__)


def _canonical_name(value: str, allowed: list[str]) -> str:
    normalized = re.sub(r"[^a-z0-9]", "", value.lower())
    matches = [name for name in allowed if re.sub(r"[^a-z0-9]", "", name.lower()) == normalized]
    if len(matches) != 1:
        raise ValueError(f"Unknown or ambiguous schema field: {value}")
    return matches[0]


def _validate_hf_request(request: OptimizationRequest, text: str, targets: list[str], inputs: list[str]) -> OptimizationRequest:
    supplied_numbers = [float(value) for value in re.findall(r"[-+]?[0-9]+(?:\.[0-9]+)?", text)]
    for constraint in request.targets:
        constraint.target = _canonical_name(constraint.target, targets)
        if not any(abs(constraint.value - value) <= 1e-9 for value in supplied_numbers):
            raise ValueError("Hugging Face introduced a numerical value that was not supplied by the user.")
    request.fixed_features = {_canonical_name(name, inputs): value for name, value in request.fixed_features.items()}
    request.bounds = {_canonical_name(name, inputs): value for name, value in request.bounds.items()}
    for objective in request.objectives:
        objective.field = _canonical_name(objective.field, targets if objective.source == "target" else inputs)
    return request


def _deterministic_parse(text: str, targets: list[str], inputs: list[str]) -> OptimizationRequest:
    lowered = text.lower()
    constraints: list[TargetConstraint] = []
    clauses = [part.strip() for part in re.split(r"[,;]|\band\b", lowered) if part.strip()]
    for target in targets:
        words = [part for part in re.split(r"[_\W]+", target.lower()) if len(part) > 2]
        target_lower = target.lower()
        if any(term in target_lower for term in ("capacity", "ultimate", " pu", "load")): words.extend(["load", "capacity", "ultimate"])
        if any(term in target_lower for term in ("deflection", "delta", "δ")): words.extend(["deflection", "delta"])
        words = list(dict.fromkeys(words))
        matching = sorted(clauses, key=lambda clause: sum(word in clause for word in words), reverse=True)
        clause = matching[0] if matching and any(word in matching[0] for word in words) else ""
        if not clause: continue
        numbers_in_clause = re.findall(r"[0-9]+(?:\.[0-9]+)?", clause)
        if not numbers_in_clause: continue
        if any(term in clause for term in ("below", "under", "at most", "maximum", "max ", "<=")): operator = "<="
        elif any(term in clause for term in ("equal", "exactly", "==")): operator = "=="
        else: operator = ">="
        constraints.append(TargetConstraint(target=target, operator=operator, value=float(numbers_in_clause[-1])))
    numbers = [float(value) for value in re.findall(r"[0-9]+(?:\.[0-9]+)?", text)]
    if not constraints and targets and numbers: constraints.append(TargetConstraint(target=targets[0], operator=">=", value=numbers[0]))
    if not constraints: raise ValueError("Could not identify a numerical target constraint. Please include a target and value.")
    fixed = {name: None for name in inputs if "unchanged" in lowered and any(word in lowered for word in re.split(r"[_\W]+", name.lower()) if len(word) > 3)}
    sustainability = next((name for name in inputs if any(term in name.lower() for term in ("sustain", "fly ash", "recycled", "replacement"))), None)
    objectives = [Objective(field=sustainability, source="feature", direction="maximize")] if sustainability else []
    return OptimizationRequest(targets=constraints, fixed_features=fixed, objectives=objectives)


def parse_request(text: str, targets: list[str], inputs: list[str]) -> dict[str, Any]:
    fallback = _deterministic_parse(text, targets, inputs)
    provider, model_id, token = hf_settings()
    if provider not in {"api", "local"} or not model_id:
        return {"request": fallback.model_dump(), "parser": "deterministic", "warnings": ["Hugging Face is unavailable or disabled; deterministic parsing was used."]}
    try:
        prompt = (
            'Return one JSON object in exactly this shape: '
            '{"targets":[{"target":"column name","operator":">=|<=|==","value":0}],'
            '"fixed_features":{},"bounds":{},"objectives":[],"algorithm":"grid"}. '
            "Use only supplied column names and numerical values. Never invent values. Request: " + text
        )
        if provider == "api":
            from huggingface_hub import InferenceClient

            client = InferenceClient(provider="auto", api_key=token or None, timeout=30)
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": "Translate the request to the specified JSON shape. Return JSON only and never invent numbers."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=300,
                response_format={"type": "json_object"},
            )
            generated = response.choices[0].message.content or ""
        else:
            from transformers import pipeline
            generated = pipeline("text2text-generation", model=model_id)(prompt, max_new_tokens=300)[0]["generated_text"]
        match = re.search(r"\{.*\}", generated, re.DOTALL)
        parsed = OptimizationRequest.model_validate(json.loads(match.group(0) if match else generated))
        parsed = _validate_hf_request(parsed, text, targets, inputs)
        return {"request": parsed.model_dump(), "parser": "hugging_face", "warnings": []}
    except Exception as exc:
        logger.warning("Hugging Face constraint parsing failed; using deterministic fallback: %s", exc)
        return {"request": fallback.model_dump(), "parser": "deterministic", "warnings": [f"Hugging Face parsing failed; deterministic parsing was used: {exc}"]}
