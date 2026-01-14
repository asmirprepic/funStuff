from __future__ import annotations
import numpy as np

def bias_rmse(samples: np.ndarray, truth: float) -> tuple[float, float]:
    """
    samples: (reps, ) esimator values


    """

    err = samples-truth
    bias = float(np.mean(err))
    rmse = float(np.sqrt(np.mean(err **2)))
    return bias, rmse

