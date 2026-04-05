from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class AR1Params:
    intercept: float
    phi: float
    sigma2: float


class AR1Model:
    def __init__(self) -> None:
        self.params_: AR1Params | None = None
        self.is_fitted_: bool = False

    def fit(self, x: np.ndarray) -> "AR1Model":
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 1 or len(x) < 3:
            raise ValueError("x must be a 1D array with length >= 3.")

        y = x[1:]
        X = np.column_stack([np.ones(len(x) - 1), x[:-1]])

        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        intercept, phi = beta

        resid = y - X @ beta
        sigma2 = float(np.mean(resid**2))
        sigma2 = max(sigma2, 1e-10)

        self.params_ = AR1Params(
            intercept=float(intercept),
            phi=float(phi),
            sigma2=sigma2,
        )
        self.is_fitted_ = True
        return self

    def predict_in_sample(self, x: np.ndarray) -> np.ndarray:
        self._require_fitted()
        x = np.asarray(x, dtype=np.float64)
        out = np.empty_like(x)
        out[0] = x[0]
        out[1:] = self.params_.intercept + self.params_.phi * x[:-1]
        return out

    def residuals(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        fitted = self.predict_in_sample(x)
        resid = np.empty_like(x)
        resid[0] = 0.0
        resid[1:] = x[1:] - fitted[1:]
        return resid

    def score(self, x: np.ndarray) -> float:
        self._require_fitted()
        x = np.asarray(x, dtype=np.float64)

        resid = self.residuals(x)[1:]
        sigma2 = self.params_.sigma2
        n = len(resid)

        ll = -0.5 * n * np.log(2.0 * np.pi * sigma2) - 0.5 * np.sum(resid**2) / sigma2
        return float(ll)

    def get_param_count(self) -> int:
        return 3  # intercept, phi, sigma2

    def get_summary(self) -> Dict[str, float]:
        self._require_fitted()
        return {
            "intercept": self.params_.intercept,
            "phi": self.params_.phi,
            "sigma2": self.params_.sigma2,
            "sigma": float(np.sqrt(self.params_.sigma2)),
        }

    def _require_fitted(self) -> None:
        if not self.is_fitted_ or self.params_ is None:
            raise RuntimeError("AR1Model is not fitted.")
