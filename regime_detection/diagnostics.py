from __future__ import annotations

from typing import Dict

import numpy as np


def compute_regime_statistics(x: np.ndarray, states: np.ndarray, gamma: np.ndarray) -> Dict[int, dict]:
    """
    Compute descriptive statistics per decoded regime.
    """
    x = np.asarray(x, dtype=np.float64)
    states = np.asarray(states, dtype=np.int64)
    gamma = np.asarray(gamma, dtype=np.float64)

    if x.ndim != 1 or states.ndim != 1:
        raise ValueError("x and states must be 1D.")
    if len(x) != len(states):
        raise ValueError("x and states must have same length.")
    if gamma.shape[0] != len(x):
        raise ValueError("gamma must have same number of rows as len(x).")

    regimes = np.unique(states)
    stats = {}

    for regime in regimes:
        mask = states == regime
        if not np.any(mask):
            continue

        x_reg = x[mask]

        stats[int(regime)] = {
            "count": int(mask.sum()),
            "fraction": float(mask.mean()),
            "mean_decoded": float(np.mean(x_reg)),
            "std_decoded": float(np.std(x_reg, ddof=1)) if len(x_reg) > 1 else 0.0,
            "min_decoded": float(np.min(x_reg)),
            "max_decoded": float(np.max(x_reg)),
            "avg_smoothed_probability": float(np.mean(gamma[:, regime])),
        }

    return stats


def expected_regime_durations(trans_mat: np.ndarray) -> np.ndarray:
    """
    For a Markov chain, expected duration in regime i is approximately:
        1 / (1 - p_ii)
    """
    trans_mat = np.asarray(trans_mat, dtype=np.float64)
    diag = np.diag(trans_mat)
    denom = np.maximum(1.0 - diag, 1e-12)
    return 1.0 / denom


def transition_count_matrix(states: np.ndarray, n_states: int) -> np.ndarray:
    states = np.asarray(states, dtype=np.int64)
    counts = np.zeros((n_states, n_states), dtype=np.int64)

    for t in range(len(states) - 1):
        counts[states[t], states[t + 1]] += 1

    return counts


def format_ai_summary(
    means: np.ndarray,
    variances: np.ndarray,
    trans_mat: np.ndarray,
    durations: np.ndarray,
    regime_stats: Dict[int, dict],
) -> str:
    """
    Creates a compact text block you can feed to an LLM for interpretation.
    """
    lines = []
    lines.append("Model summary for regime interpretation:")
    lines.append("")

    for k in range(len(means)):
        lines.append(f"Regime {k}:")
        lines.append(f"  Mean: {means[k]:.6f}")
        lines.append(f"  Variance: {variances[k]:.6f}")
        lines.append(f"  Std Dev: {np.sqrt(variances[k]):.6f}")
        lines.append(f"  Expected Duration: {durations[k]:.3f}")
        if k in regime_stats:
            lines.append(f"  Fraction of decoded path: {regime_stats[k]['fraction']:.3f}")
            lines.append(f"  Decoded min/max: {regime_stats[k]['min_decoded']:.6f} / {regime_stats[k]['max_decoded']:.6f}")
        lines.append("")

    lines.append("Transition Matrix:")
    lines.append(np.array2string(trans_mat, precision=4, suppress_small=True))

    lines.append("")
    lines.append(
        "Please interpret the regimes in plain but technically serious language. "
        "Describe persistence, volatility differences, likely use cases, and where this model might fail."
    )
    return "\n".join(lines)
