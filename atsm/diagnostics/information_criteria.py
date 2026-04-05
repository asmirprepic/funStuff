from __future__ import annotations

import numpy as np


def compute_aic(loglik: float, n_params: int) -> float:
    return float(2 * n_params - 2 * loglik)


def compute_bic(loglik: float, n_params: int, n_obs: int) -> float:
    return float(np.log(n_obs) * n_params - 2 * loglik)


def hmm_param_count(n_states: int) -> int:
    # initial probs: K-1
    # transition matrix: K*(K-1)
    # means: K
    # variances: K
    k = n_states
    return (k - 1) + k * (k - 1) + k + k
