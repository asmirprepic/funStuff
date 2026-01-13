from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Tuple

@dataclass(frozen=True)
class ExperimentConfig:
    # Core grid
    n_list: Sequence[int] = (250,1000,5000)
    eps_list: Sequence[float] = (0.0, 0.005, 0.01, 0.02)
    persitence_list: Sequence[float] = (0.80, 0.95, 0.99)
    vol_ratio_list: Sequence[float] = (3.0,6.0)

    sigma1: float = 1.0
    mean: float = 0.0

    # Contamination
    outlier_k: float = 10.0

    # Monte Carlo
    reps: int = 1000
    seed: int = 12345

    #Output
    output_dir: str = "outputs"
    results_csv: str = "outputs/results.csv"

    # Plot
    dpi: int = 160

    def transition_matrix(self, p:float) -> Tuple[Tuple[float,float], Tuple[float]]
