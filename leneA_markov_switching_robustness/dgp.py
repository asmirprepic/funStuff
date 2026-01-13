from __future__ import annotations
import numpy as np
from .markov import simulate_two_state_markov


def unconditional_sigma(P: np.ndarray, sigma1: float, sigma2: float) -> float:
    """Unconditional SD under switching Gaussian with mean 0 and stationary distribution."""
    p11, p22 = P[0, 0], P[1, 1]
    denom = (1.0 - p11) + (1.0 - p22)
    pi0 = (1.0 - p22) / denom if denom > 0 else 0.5
    pi1 = 1.0 - pi0
    var = pi0 * (sigma1 ** 2) + pi1 * (sigma2 ** 2)
    return float(np.sqrt(var))


def simulate_switching_gaussian(
    n: int,
    P: np.ndarray,
    mean: float,
    sigma1: float,
    sigma2: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      x: (n,) observations
      s: (n,) latent states in {0,1}
    """
    s = simulate_two_state_markov(n=n, P=P, rng=rng)
    sigmas = np.where(s == 0, sigma1, sigma2)
    x = mean + sigmas * rng.standard_normal(n)
    return x, s


def contaminate_huber_option1(
    x: np.ndarray,
    eps: float,
    sigma2: float,
    k: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Huber epsilon contamination:
      with prob eps replace x_t by outlier y_t ~ N(0, (k*sigma2)^2)
    """
    if eps <= 0:
        return x

    n = x.shape[0]
    mask = rng.random(n) < eps
    y = (k * sigma2) * rng.standard_normal(n)
    out = x.copy()
    out[mask] = y[mask]
    return out
