from __future__ import annotations
import numpy as np

def mean_est(x: np.ndarray) -> float:
    return float(np.mean(x))

def median_est(x: np.ndarray) -> float:
    return float(np.median(x))

def sd_est(x: np.ndarray, ddof: int = 1) -> float:
    return float(np.std(x, ddof=ddof))

def mad_est(x: np.ndarray) -> float:
    """
    MAD scaled for normal consistency: 1.4826 * median(|x - median(x)|)
    """

    med = np.median(x)
    mad = np.median(np.abs(x-med))
    return float(1.4826*mad)
