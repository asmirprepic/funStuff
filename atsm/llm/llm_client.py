from __future__ import annotations

import json
import os
from typing import Any, Dict

import requests

from llm.prompt_builder import build_prompts
from llm.schemas import LLM_OUTPUT_SCHEMA


class LLMClientError(RuntimeError):
    pass


def _extract_text_content(response_json: Dict[str, Any]) -> str:
    """
    Extract text output from the API response.
    This is written defensively because response shapes can evolve.
    """
    # Common Responses-style text output path
    output = response_json.get("output", [])
    texts: list[str] = []

    for item in output:
        content = item.get("content", [])
        for block in content:
            if block.get("type") in {"output_text", "text"} and isinstance(block.get("text"), str):
                texts.append(block["text"])

    if texts:
        return "\n".join(texts).strip()

    # Fallback for other shapes
    if isinstance(response_json.get("output_text"), str):
        return response_json["output_text"].strip()

    raise LLMClientError("Could not extract text content from LLM response.")


def _validate_required_keys(obj: Dict[str, Any]) -> None:
    required_top = {
        "recommended_model_overall",
        "overall_reasoning",
        "scenario_failure_analysis",
        "next_adversarial_scenario",
    }
    missing = required_top - set(obj.keys())
    if missing:
        raise LLMClientError(f"LLM JSON missing required keys: {sorted(missing)}")


def run_llm_analysis(
    payload: Dict[str, Any],
    model: str = "gpt-4.1",
    timeout_seconds: int = 90,
) -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise LLMClientError("OPENAI_API_KEY is not set in the environment.")

    developer_prompt, user_prompt = build_prompts(payload)

    body = {
        "model": model,
        "input": [
            {"role": "developer", "content": developer_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": LLM_OUTPUT_SCHEMA["name"],
                "schema": LLM_OUTPUT_SCHEMA["schema"],
                "strict": LLM_OUTPUT_SCHEMA["strict"],
            }
        },
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    resp = requests.post(
        "https://api.openai.com/v1/responses",
        headers=headers,
        json=body,
        timeout=timeout_seconds,
    )

    if resp.status_code >= 400:
        raise LLMClientError(
            f"OpenAI API error {resp.status_code}: {resp.text}"
        )

    response_json = resp.json()
    raw_text = _extract_text_content(response_json)

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise LLMClientError(f"LLM did not return valid JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise LLMClientError("LLM output was JSON but not an object.")

    _validate_required_keys(parsed)
    return parsed
