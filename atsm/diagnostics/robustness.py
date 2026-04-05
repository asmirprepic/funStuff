from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from config import FAILURE_SCORE_WEIGHTS, LOW_CONFIDENCE_THRESHOLD
from diagnostics.information_criteria import compute_aic, compute_bic, hmm_param_count
from diagnostics.uncertainty import low_confidence_fraction, posterior_entropy


def summarize_model_result(
    fitted_result: Dict,
    x: np.ndarray,
    scenario_name: str,
) -> Dict:
    model_type = fitted_result["model_type"]
    loglik = fitted_result["loglik"]

    if model_type == "hmm":
        n_states = fitted_result["n_states"]
        gamma = fitted_result["gamma"]

        n_params = hmm_param_count(n_states)
        aic = compute_aic(loglik, n_params)
        bic = compute_bic(loglik, n_params, len(x))
        entropy = posterior_entropy(gamma)
        low_conf = low_confidence_fraction(gamma, threshold=LOW_CONFIDENCE_THRESHOLD)

        return {
            "scenario": scenario_name,
            "model_name": fitted_result["model_name"],
            "model_type": model_type,
            "n_states": n_states,
            "loglik": loglik,
            "aic": aic,
            "bic": bic,
            "entropy": entropy,
            "low_conf_fraction": low_conf,
        }

    if model_type == "ar1":
        n_params = fitted_result["model_obj"].get_param_count()
        aic = compute_aic(loglik, n_params)
        bic = compute_bic(loglik, n_params, len(x) - 1)

        return {
            "scenario": scenario_name,
            "model_name": fitted_result["model_name"],
            "model_type": model_type,
            "n_states": None,
            "loglik": loglik,
            "aic": aic,
            "bic": bic,
            "entropy": np.nan,
            "low_conf_fraction": np.nan,
        }

    raise ValueError(f"Unknown model_type={model_type}")


def build_results_dataframe(all_rows: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(all_rows)
    return df.sort_values(["scenario", "model_name"]).reset_index(drop=True)


def attach_failure_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    baseline = df[df["scenario"] == "baseline"].copy()
    if baseline.empty:
        raise ValueError("Baseline scenario is required for failure scoring.")

    baseline_ref = baseline.set_index("model_name")

    loglik_deg = []
    entropy_inc = []
    confidence_loss = []

    for _, row in df.iterrows():
        ref = baseline_ref.loc[row["model_name"]]

        ll_deg = max(0.0, (ref["loglik"] - row["loglik"]) / max(abs(ref["loglik"]), 1e-12))
        loglik_deg.append(ll_deg)

        if np.isnan(row["entropy"]) or np.isnan(ref["entropy"]):
            entropy_inc.append(np.nan)
        else:
            entropy_inc.append(max(0.0, row["entropy"] - ref["entropy"]))

        if np.isnan(row["low_conf_fraction"]) or np.isnan(ref["low_conf_fraction"]):
            confidence_loss.append(np.nan)
        else:
            confidence_loss.append(max(0.0, row["low_conf_fraction"] - ref["low_conf_fraction"]))

    df["loglik_deg"] = loglik_deg
    df["entropy_inc"] = entropy_inc
    df["confidence_loss"] = confidence_loss

    # normalize each component to 0-1 across available values
    for col in ["loglik_deg", "entropy_inc", "confidence_loss"]:
        valid = df[col].dropna()
        if len(valid) == 0:
            df[f"{col}_norm"] = np.nan
            continue
        min_v = valid.min()
        max_v = valid.max()
        if np.isclose(max_v, min_v):
            df[f"{col}_norm"] = 0.0
        else:
            df[f"{col}_norm"] = (df[col] - min_v) / (max_v - min_v)

    weights = FAILURE_SCORE_WEIGHTS

    def compute_score(row: pd.Series) -> float:
        comps = []
        wts = []

        for col, key in [
            ("loglik_deg_norm", "loglik_deg"),
            ("entropy_inc_norm", "entropy_inc"),
            ("confidence_loss_norm", "confidence_loss"),
        ]:
            value = row[col]
            if pd.notna(value):
                comps.append(float(value) * weights[key])
                wts.append(weights[key])

        if not wts:
            return 0.0
        return float(sum(comps) / sum(wts))

    df["failure_score"] = df.apply(compute_score, axis=1)
    return df
