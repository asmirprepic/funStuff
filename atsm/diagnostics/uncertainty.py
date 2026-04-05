from __future__ import annotations

import numpy as np


def posterior_entropy(gamma: np.ndarray) -> float:
    gamma = np.asarray(gamma, dtype=np.float64)
    gamma = np.clip(gamma, 1e-12, 1.0)
    ent = -np.sum(gamma * np.log(gamma), axis=1)
    return float(np.mean(ent))


def low_confidence_fraction(gamma: np.ndarray, threshold: float = 0.60) -> float:
    gamma = np.asarray(gamma, dtype=np.float64)
    max_probs = np.max(gamma, axis=1)
    return float(np.mean(max_probs < threshold))
