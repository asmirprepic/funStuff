from __future__ import annotations

from typing import Dict

import numpy as np


def simulate_llm_scenario(
    base_x: np.ndarray,
    scenario_name: str,
    random_state: int = 123,
) -> Dict:
    """
    Simple transformation-based generator.
    We map LLM scenario -> actual simulation logic.
    """

    rng = np.random.default_rng(random_state)
    x = base_x.copy()

    if "drift" in scenario_name.lower():
        trend = np.linspace(0, 0.01, len(x))
        x = x + trend

    elif "volatility" in scenario_name.lower():
        scale = np.linspace(1.0, 3.0, len(x))
        x = x * scale

    elif "noise" in scenario_name.lower():
        x = x + rng.normal(0, np.std(x), size=len(x))

    else:
        # fallback: small random perturbation
        x = x + rng.normal(0, 0.5 * np.std(x), size=len(x))

    return {
        "scenario_name": f"llm_{scenario_name}",
        "x": x,
        "true_states": None,
    }
