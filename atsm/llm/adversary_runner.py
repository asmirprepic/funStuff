from __future__ import annotations

import json
import os
from typing import Dict

import requests

from llm.adversary_prompt import build_adversary_prompt
from llm.adversary_schema import ADVERSARY_SCHEMA


def run_adversary_llm(payload: Dict, model: str = "gpt-4.1") -> Dict:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    system_prompt, user_prompt = build_adversary_prompt(payload)

    body = {
        "model": model,
        "input": [
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": ADVERSARY_SCHEMA["name"],
                "schema": ADVERSARY_SCHEMA["schema"],
                "strict": True,
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
        timeout=60,
    )

    if resp.status_code >= 400:
        raise RuntimeError(resp.text)

    response_json = resp.json()

    # extract text
    text = ""
    for item in response_json.get("output", []):
        for c in item.get("content", []):
            if c.get("type") in ["output_text", "text"]:
                text += c["text"]

    return json.loads(text)
