from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_scenario_series(x: np.ndarray, title: str) -> None:
    plt.figure(figsize=(12, 4))
    plt.plot(x, linewidth=1.0)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Observation")
    plt.tight_layout()
    plt.show()


def plot_failure_heatmap(df: pd.DataFrame) -> None:
    pivot = df.pivot(index="scenario", columns="model_name", values="failure_score")

    plt.figure(figsize=(8, 4.5))
    plt.imshow(pivot.values, aspect="auto")
    plt.colorbar(label="Failure Score")
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.title("Model Failure Severity by Scenario")
    plt.tight_layout()
    plt.show()
