from __future__ import annotations

import logging
from typing import Any

from app.core.config import hf_settings


SYSTEM_PROMPT = (
    "You are an engineering explanation assistant. Explain only supplied verified numerical results. "
    "Never alter, infer, invent, or recalculate values. Distinguish ML predictions from experimental "
    "verification and never claim code compliance or structural safety approval."
)
logger = logging.getLogger(__name__)


def deterministic_prediction_explanation(result: dict, shap_result: dict | None = None) -> str:
    values = ", ".join(f"{name}: {value:.3f}" for name, value in result["predictions"].items())
    status = result["domain_analysis"]["overall"].replace("_", " ").lower()
    text = f"The trained {result['model_name']} regression model predicts {values}. The input is {status}."
    if shap_result and shap_result.get("feature_contributions"):
        first_target = next(iter(shap_result["feature_contributions"]))
        rows = shap_result["feature_contributions"][first_target]
        if rows:
            text += f" SHAP identifies {rows[0]['feature']} as the largest local model contribution for {first_target}."
    return text + " These values are model estimates, not experimentally verified performance or a structural safety approval."


def explain_verified(payload: dict[str, Any], fallback: str) -> str:
    provider, model_id, token = hf_settings()
    if provider == "disabled" or not model_id:
        return fallback
    try:
        if provider == "api":
            from huggingface_hub import InferenceClient

            client = InferenceClient(provider="auto", api_key=token or None, timeout=30)
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Explain this verified payload concisely: {payload}"},
                ],
                max_tokens=220,
            )
            content = response.choices[0].message.content
            return content.strip() if content else fallback
        if provider == "local":
            from transformers import pipeline
            generator = pipeline("text2text-generation", model=model_id)
            return generator(f"{SYSTEM_PROMPT}\nVerified payload: {payload}", max_new_tokens=220)[0]["generated_text"]
    except Exception as exc:
        logger.warning("Hugging Face explanation failed; using deterministic fallback: %s", exc)
        return fallback
    return fallback
