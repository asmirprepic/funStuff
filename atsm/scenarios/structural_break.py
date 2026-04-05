from __future__ import annotations

from typing import Dict

import numpy as np


def simulate_structural_break_scenario(
    n: int = 1500,
    break_point: int | None = None,
    random_state: int = 44,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(random_state)

    if break_point is None:
        break_point = n // 2

    x = np.zeros(n, dtype=np.float64)
    states = np.zeros(n, dtype=np.int64)

    init_probs = np.array([0.85, 0.15], dtype=np.float64)

    trans_1 = np.array([[0.96, 0.04], [0.08, 0.92]], dtype=np.float64)
    means_1 = np.array([0.001, -0.002], dtype=np.float64)
    stds_1 = np.array([0.010, 0.025], dtype=np.float64)

    trans_2 = np.array([[0.80, 0.20], [0.25, 0.75]], dtype=np.float64)
    means_2 = np.array([-0.001, 0.000], dtype=np.float64)
    stds_2 = np.array([0.015, 0.050], dtype=np.float64)

    states[0] = rng.choice(2, p=init_probs)
    x[0] = rng.normal(means_1[states[0]], stds_1[states[0]])

    for t in range(1, n):
        if t < break_point:
            trans = trans_1
            means = means_1
            stds = stds_1
        else:
            trans = trans_2
            means = means_2
            stds = stds_2

        states[t] = rng.choice(2, p=trans[states[t - 1]])
        x[t] = rng.normal(means[states[t]], stds[states[t]])

    return {
        "scenario_name": "structural_break",
        "x": x,
        "true_states": states,
        "break_point": break_point,
    }
