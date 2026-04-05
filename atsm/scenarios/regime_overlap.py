from __future__ import annotations

from typing import Dict

import numpy as np


def simulate_regime_overlap_scenario(
    n: int = 1500,
    random_state: int = 45,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(random_state)

    init_probs = np.array([0.60, 0.25, 0.15], dtype=np.float64)
    trans_mat = np.array(
        [
            [0.93, 0.05, 0.02],
            [0.08, 0.86, 0.06],
            [0.05, 0.10, 0.85],
        ],
        dtype=np.float64,
    )

    # deliberately close means/vols
    means = np.array([0.0005, -0.0003, 0.0001], dtype=np.float64)
    stds = np.array([0.018, 0.022, 0.027], dtype=np.float64)

    states = np.zeros(n, dtype=np.int64)
    x = np.zeros(n, dtype=np.float64)

    states[0] = rng.choice(3, p=init_probs)
    x[0] = rng.normal(means[states[0]], stds[states[0]])

    for t in range(1, n):
        states[t] = rng.choice(3, p=trans_mat[states[t - 1]])
        x[t] = rng.normal(means[states[t]], stds[states[t]])

    return {
        "scenario_name": "regime_overlap",
        "x": x,
        "true_states": states,
    }
