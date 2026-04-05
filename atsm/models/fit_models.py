from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from hmm import GaussianHMM
from models.ar1 import AR1Model


def fit_hmm_model(
    x: np.ndarray,
    n_states: int,
    random_state: int = 42,
) -> Dict[str, Any]:
    model = GaussianHMM(
        n_states=n_states,
        max_iter=150,
        tol=1e-5,
        min_variance=1e-8,
        random_state=random_state,
    )
    model.fit(x, verbose=False)

    gamma = model.predict_proba(x)
    decoded = model.predict(x)
    loglik = model.score(x)
    params = model.get_params()

    return {
        "model_name": f"hmm_{n_states}",
        "model_type": "hmm",
        "n_states": n_states,
        "model_obj": model,
        "loglik": float(loglik),
        "gamma": gamma,
        "decoded_states": decoded,
        "params": params,
    }


def fit_ar1_model(x: np.ndarray) -> Dict[str, Any]:
    model = AR1Model().fit(x)
    loglik = model.score(x)
    resid = model.residuals(x)
    summary = model.get_summary()

    return {
        "model_name": "ar1",
        "model_type": "ar1",
        "n_states": None,
        "model_obj": model,
        "loglik": float(loglik),
        "residuals": resid,
        "params": summary,
    }


def fit_all_models(x: np.ndarray, random_state: int = 42) -> List[Dict[str, Any]]:
    results = [
        fit_hmm_model(x, n_states=2, random_state=random_state),
        fit_hmm_model(x, n_states=3, random_state=random_state),
        fit_ar1_model(x),
    ]
    return results
