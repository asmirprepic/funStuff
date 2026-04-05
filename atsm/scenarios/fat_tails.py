from __future__ import annotations

from typing import Dict

import numpy as np

from scenarios.baseline import simulate_baseline_series


def simulate_fat_tail_scenario(
    n: int = 1500,
    df: float = 4.0,
    random_state: int = 43,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(random_state)

    base = simulate_baseline_series(n=n, random_state=random_state)
    states = base["true_states"]
    means = base["means"]
    stds = base["stds"]

    x = np.zeros(n, dtype=np.float64)
    scale_adj = np.sqrt((df - 2.0) / df)  # variance normalization for t-dist with df > 2

    for t in range(n):
        z = rng.standard_t(df=df) * scale_adj
        x[t] = means[states[t]] + stds[states[t]] * z

    return {
        "scenario_name": "fat_tails",
        "x": x,
        "true_states": states,
    }
