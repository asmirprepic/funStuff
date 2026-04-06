from __future__ import annotations

from typing import Dict, Any


ADVERSARY_SCHEMA: Dict[str, Any] = {
    "name": "adversarial_scenario_design",
    "schema": {
        "type": "object",
        "properties": {
            "diagnosed_weakest_assumption": {"type": "string"},
            "proposed_scenario": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "targeted_weakness": {"type": "string"},
                },
                "required": ["name", "description", "targeted_weakness"],
            },
            "predicted_outcome": {
                "type": "object",
                "properties": {
                    "most_fragile_model": {"type": "string"},
                    "most_resilient_model": {"type": "string"},
                    "expected_effects": {"type": "string"},
                },
                "required": [
                    "most_fragile_model",
                    "most_resilient_model",
                    "expected_effects",
                ],
            },
        },
        "required": [
            "diagnosed_weakest_assumption",
            "proposed_scenario",
            "predicted_outcome",
        ],
        "additionalProperties": False,
    },
    "strict": True,
}
