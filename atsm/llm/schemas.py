from __future__ import annotations

from typing import Any, Dict, List


LLM_OUTPUT_SCHEMA: Dict[str, Any] = {
    "name": "adversarial_time_series_analysis",
    "schema": {
        "type": "object",
        "properties": {
            "recommended_model_overall": {"type": "string"},
            "overall_reasoning": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "maxItems": 6,
            },
            "scenario_failure_analysis": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "scenario": {"type": "string"},
                        "most_resilient_model": {"type": "string"},
                        "most_fragile_model": {"type": "string"},
                        "main_failure_pattern": {"type": "string"},
                    },
                    "required": [
                        "scenario",
                        "most_resilient_model",
                        "most_fragile_model",
                        "main_failure_pattern",
                    ],
                    "additionalProperties": False,
                },
            },
            "next_adversarial_scenario": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "targeted_weakness": {"type": "string"},
                },
                "required": ["name", "description", "targeted_weakness"],
                "additionalProperties": False,
            },
        },
        "required": [
            "recommended_model_overall",
            "overall_reasoning",
            "scenario_failure_analysis",
            "next_adversarial_scenario",
        ],
        "additionalProperties": False,
    },
    "strict": True,
}


def build_llm_input_payload(results_df) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []

    for _, row in results_df.iterrows():
        n_states = row["n_states"]
        entropy = row["entropy"]
        low_conf = row["low_conf_fraction"]
        entropy_inc = row["entropy_inc"]
        confidence_loss = row["confidence_loss"]

        rows.append(
            {
                "scenario": row["scenario"],
                "model_name": row["model_name"],
                "model_type": row["model_type"],
                "n_states": None if n_states != n_states else int(n_states),
                "loglik": float(row["loglik"]),
                "aic": float(row["aic"]),
                "bic": float(row["bic"]),
                "entropy": None if entropy != entropy else float(entropy),
                "low_conf_fraction": None if low_conf != low_conf else float(low_conf),
                "loglik_deg": float(row["loglik_deg"]),
                "entropy_inc": None if entropy_inc != entropy_inc else float(entropy_inc),
                "confidence_loss": None if confidence_loss != confidence_loss else float(confidence_loss),
                "failure_score": float(row["failure_score"]),
            }
        )

    return {
        "project": "Adversarial Time-Series Lab",
        "task": (
            "Compare the candidate time-series models conservatively using only the supplied metrics. "
            "Recommend the most defensible model overall, explain failure patterns by scenario, "
            "and suggest one additional mathematically meaningful adversarial scenario."
        ),
        "results": rows,
    }
