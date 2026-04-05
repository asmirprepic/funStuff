from __future__ import annotations

import json
from typing import Any, Dict, Tuple


def build_prompts(payload: Dict[str, Any]) -> Tuple[str, str]:
    developer_prompt = """
You are a quantitative research assistant.

You will be given structured diagnostics from several time-series models evaluated under adversarial scenarios.
Your job is to compare them conservatively and only from the supplied metrics.

Rules:
1. Do not invent facts not present in the input.
2. Prefer statistical defensibility over storytelling.
3. Use only the provided diagnostics.
4. If evidence is mixed, say so in the reasoning.
5. Keep the reasoning concise and technically grounded.
""".strip()

    user_prompt = f"""
Analyze the following structured model diagnostics.

Input JSON:
{json.dumps(payload, indent=2)}
""".strip()

    return developer_prompt, user_prompt
