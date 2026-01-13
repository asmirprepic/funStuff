from __future__ import annotations
import numpy as np

def simulate_two_state_markov(n: int, P: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Simulate a 2-state Markov chain S_t in {0,1} with transition matrix P
    Starts from stationary
    """

    if P.shape != (2,2):
        raise ValueError("P must be 2x2")

    p11, p12 = P[0,0],P[0,1]
    p21, p22 = P[1,0], P[1,1]

    denom = (1.0-p11) + (1.0 - p22)
    pi0 = (1.0 - p22) / denom if denom > 0 else 0.5
    pi = np.array([pi0, 1.0 - pi0], dtype=float)

    S = np.empty(n, dtype=np.int8)
    S[0] = rng.choice(2,p = pi)

    u = rng.random(n-1)
    for t in range(1,n):
        prev = S[t-1]
        if prev == 0:
            S[t] = 0 if u[t-1] < P[0,0] else 1
        else:
            S[t] = 1 if u[t-1] < P[1,1] else 0
        return S
