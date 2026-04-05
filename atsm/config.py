from __future__ import annotations

RANDOM_STATE = 42
N_OBS = 1500

LOW_CONFIDENCE_THRESHOLD = 0.60

FAILURE_SCORE_WEIGHTS = {
    "loglik_deg": 1.0,
    "entropy_inc": 1.0,
    "confidence_loss": 1.0,
}
