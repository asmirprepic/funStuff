from __future__ import annotations

import json
from typing import Dict, Tuple


def build_adversary_prompt(payload: Dict) -> Tuple[str, str]:
    system_prompt = """
You are a quantitative researcher specializing in model robustness.

You are given structured diagnostics for time-series models under adversarial scenarios.

Your job:
1. Identify the most likely violated modeling assumption.
2. Propose ONE new adversarial scenario that would meaningfully stress the models.
3. Predict which model will be most fragile and most resilient.
4. Keep everything mathematically grounded and concise.

Rules:
- Use only the provided diagnostics
- Do not invent external data
- Avoid vague language
- Be specific about the stress mechanism
""".strip()

    user_prompt = f"""
Here are the current diagnostics:

{json.dumps(payload, indent=2)}

Design the next adversarial test and predict the outcome.
""".strip()

    return system_prompt, user_prompt
