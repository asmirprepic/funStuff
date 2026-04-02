from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from diagnostics import (
    compute_regime_statistics,
    expected_regime_durations,
    format_ai_summary,
    transition_count_matrix,
)
from hmm import GaussianHMM


def simulate_regime_series(
    n: int,
    init_probs: np.ndarray,
    trans_mat: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)

    k = len(init_probs)
    states = np.zeros(n, dtype=np.int64)
    x = np.zeros(n, dtype=np.float64)

    states[0] = rng.choice(k, p=init_probs)
    x[0] = rng.normal(means[states[0]], stds[states[0]])

    for t in range(1, n):
        states[t] = rng.choice(k, p=trans_mat[states[t - 1]])
        x[t] = rng.normal(means[states[t]], stds[states[t]])

    return x, states


def plot_series_with_states(x: np.ndarray, states: np.ndarray, title: str) -> None:
    plt.figure(figsize=(12, 5))
    plt.plot(x, linewidth=1.0)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Observation")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 2.8))
    plt.step(np.arange(len(states)), states, where="post")
    plt.title(f"{title} - decoded states")
    plt.xlabel("Time")
    plt.ylabel("State")
    plt.tight_layout()
    plt.show()


def plot_smoothed_probabilities(gamma: np.ndarray) -> None:
    for k in range(gamma.shape[1]):
        plt.figure(figsize=(12, 3))
        plt.plot(gamma[:, k], linewidth=1.0)
        plt.title(f"Smoothed Probability - Regime {k}")
        plt.xlabel("Time")
        plt.ylabel("Probability")
        plt.ylim(0.0, 1.0)
        plt.tight_layout()
        plt.show()


def plot_log_likelihood(history: list[float]) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(history, marker="o")
    plt.title("EM Log-Likelihood Progression")
    plt.xlabel("Iteration")
    plt.ylabel("Log-Likelihood")
    plt.tight_layout()
    plt.show()


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Simulate sample data
    # ------------------------------------------------------------------
    true_init = np.array([0.70, 0.20, 0.10], dtype=np.float64)
    true_trans = np.array(
        [
            [0.94, 0.05, 0.01],
            [0.06, 0.90, 0.04],
            [0.03, 0.10, 0.87],
        ],
        dtype=np.float64,
    )
    true_means = np.array([0.001, -0.002, 0.000], dtype=np.float64)
    true_stds = np.array([0.008, 0.020, 0.040], dtype=np.float64)

    x, true_states = simulate_regime_series(
        n=1500,
        init_probs=true_init,
        trans_mat=true_trans,
        means=true_means,
        stds=true_stds,
        random_state=123,
    )

    # ------------------------------------------------------------------
    # 2. Fit HMM
    # ------------------------------------------------------------------
    model = GaussianHMM(
        n_states=3,
        max_iter=150,
        tol=1e-5,
        min_variance=1e-8,
        random_state=123,
    )

    model.fit(x, verbose=True)

    params = model.get_params()
    gamma = model.predict_proba(x)
    decoded_states = model.predict(x)
    ll = model.score(x)

    print("\nFinal log-likelihood:", ll)
    print("\nEstimated parameters:")
    print("Initial probabilities:\n", params["init_probs"])
    print("Transition matrix:\n", params["trans_mat"])
    print("Means:\n", params["means"])
    print("Variances:\n", params["variances"])
    print("Std devs:\n", params["std_devs"])

    # ------------------------------------------------------------------
    # 3. Diagnostics
    # ------------------------------------------------------------------
    regime_stats = compute_regime_statistics(x, decoded_states, gamma)
    durations = expected_regime_durations(params["trans_mat"])
    trans_counts = transition_count_matrix(decoded_states, n_states=3)

    print("\nExpected regime durations:")
    print(durations)

    print("\nDecoded transition counts:")
    print(trans_counts)

    print("\nRegime statistics:")
    for regime, stats in regime_stats.items():
        print(f"Regime {regime}:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    # ------------------------------------------------------------------
    # 4. AI-ready summary text
    # ------------------------------------------------------------------
    ai_prompt_text = format_ai_summary(
        means=params["means"],
        variances=params["variances"],
        trans_mat=params["trans_mat"],
        durations=durations,
        regime_stats=regime_stats,
    )

    print("\n" + "=" * 80)
    print("AI-READY SUMMARY")
    print("=" * 80)
    print(ai_prompt_text)

    # ------------------------------------------------------------------
    # 5. Plots
    # ------------------------------------------------------------------
    plot_series_with_states(x, true_states, "Simulated Series (True States)")
    plot_series_with_states(x, decoded_states, "Simulated Series (Decoded States)")
    plot_smoothed_probabilities(gamma)
    plot_log_likelihood(model.log_likelihood_history_)


if __name__ == "__main__":
    main()
