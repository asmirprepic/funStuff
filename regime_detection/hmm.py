from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


ArrayLike = np.ndarray


def _check_1d(x: ArrayLike, name: str = "x") -> ArrayLike:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"{name} must be a 1D array, got shape={x.shape}.")
    if len(x) < 2:
        raise ValueError(f"{name} must contain at least 2 observations.")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contains non-finite values.")
    return x


def _logsumexp(a: ArrayLike, axis: Optional[int] = None, keepdims: bool = False) -> ArrayLike:
    """
    Stable log(sum(exp(a))) implementation.
    """
    a_max = np.max(a, axis=axis, keepdims=True)
    shifted = a - a_max
    out = a_max + np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    if not keepdims and axis is not None:
        out = np.squeeze(out, axis=axis)
    return out


def _normalize_rows(mat: ArrayLike, eps: float = 1e-16) -> ArrayLike:
    row_sums = np.sum(mat, axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, eps)
    return mat / row_sums


def _gaussian_logpdf(x: ArrayLike, means: ArrayLike, variances: ArrayLike) -> ArrayLike:
    """
    Return log p(x_t | state=k) for all t,k.
    Shape:
        x: (T,)
        means: (K,)
        variances: (K,)
    Returns:
        logpdf: (T, K)
    """
    x = x[:, None]                # (T, 1)
    means = means[None, :]        # (1, K)
    variances = variances[None, :]  # (1, K)

    if np.any(variances <= 0):
        raise ValueError("All variances must be strictly positive.")

    return -0.5 * (
        np.log(2.0 * np.pi * variances)
        + ((x - means) ** 2) / variances
    )


@dataclass
class HMMParams:
    init_probs: ArrayLike   # (K,)
    trans_mat: ArrayLike    # (K, K)
    means: ArrayLike        # (K,)
    variances: ArrayLike    # (K,)


class GaussianHMM:
    """
    Hidden Markov Model with Gaussian state-dependent emissions.

    Observations:
        X_t | S_t = k ~ N(mu_k, sigma_k^2)

    State process:
        P(S_t = j | S_{t-1} = i) = A_{ij}
    """

    def __init__(
        self,
        n_states: int,
        max_iter: int = 200,
        tol: float = 1e-6,
        min_variance: float = 1e-6,
        random_state: Optional[int] = None,
    ) -> None:
        if n_states < 2:
            raise ValueError("n_states must be at least 2.")
        self.n_states = int(n_states)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.min_variance = float(min_variance)
        self.rng = np.random.default_rng(random_state)

        self.params_: Optional[HMMParams] = None
        self.log_likelihood_history_: list[float] = []
        self.is_fitted_: bool = False

    # ---------------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------------
    def _initialize_params(self, x: ArrayLike) -> HMMParams:
        T = len(x)
        K = self.n_states

        # Quantile-based initialization for means
        quantiles = np.linspace(0.1, 0.9, K)
        means = np.quantile(x, quantiles)

        overall_var = np.var(x, ddof=1)
        overall_var = max(overall_var, self.min_variance)
        variances = np.full(K, overall_var, dtype=np.float64)

        # Mildly persistent transition matrix
        trans_mat = np.full((K, K), 1.0 / K, dtype=np.float64)
        np.fill_diagonal(trans_mat, 0.85)
        off_diag = (1.0 - 0.85) / (K - 1)
        for i in range(K):
            for j in range(K):
                if i != j:
                    trans_mat[i, j] = off_diag

        init_probs = np.full(K, 1.0 / K, dtype=np.float64)

        return HMMParams(
            init_probs=init_probs,
            trans_mat=trans_mat,
            means=means.astype(np.float64),
            variances=variances.astype(np.float64),
        )

    # ---------------------------------------------------------------------
    # Core probability routines
    # ---------------------------------------------------------------------
    def _compute_log_emission_probs(self, x: ArrayLike, params: HMMParams) -> ArrayLike:
        return _gaussian_logpdf(x, params.means, params.variances)

    def _forward_log(self, log_emissions: ArrayLike, params: HMMParams) -> Tuple[ArrayLike, float]:
        """
        Forward recursion in log space.
        Returns:
            log_alpha: (T, K)
            log_likelihood: scalar
        """
        T, K = log_emissions.shape
        log_alpha = np.zeros((T, K), dtype=np.float64)

        log_init = np.log(np.maximum(params.init_probs, 1e-300))
        log_trans = np.log(np.maximum(params.trans_mat, 1e-300))

        log_alpha[0] = log_init + log_emissions[0]

        for t in range(1, T):
            # For each current state j:
            # log_alpha[t,j] = log_emission[t,j] + logsumexp_i(log_alpha[t-1,i] + logA[i,j])
            tmp = log_alpha[t - 1][:, None] + log_trans
            log_alpha[t] = log_emissions[t] + _logsumexp(tmp, axis=0)

        log_likelihood = float(_logsumexp(log_alpha[-1], axis=0))
        return log_alpha, log_likelihood

    def _backward_log(self, log_emissions: ArrayLike, params: HMMParams) -> ArrayLike:
        """
        Backward recursion in log space.
        Returns:
            log_beta: (T, K)
        """
        T, K = log_emissions.shape
        log_beta = np.zeros((T, K), dtype=np.float64)

        log_trans = np.log(np.maximum(params.trans_mat, 1e-300))

        for t in range(T - 2, -1, -1):
            # log_beta[t,i] = logsumexp_j(logA[i,j] + log_emission[t+1,j] + log_beta[t+1,j])
            tmp = log_trans + log_emissions[t + 1][None, :] + log_beta[t + 1][None, :]
            log_beta[t] = _logsumexp(tmp, axis=1)

        return log_beta

    def _smoothed_posteriors(
        self,
        log_alpha: ArrayLike,
        log_beta: ArrayLike,
        log_emissions: ArrayLike,
        params: HMMParams,
        log_likelihood: float,
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        Returns:
            gamma: smoothed state probabilities, shape (T, K)
            xi: expected transition probabilities, shape (T-1, K, K)
        """
        T, K = log_alpha.shape
        log_trans = np.log(np.maximum(params.trans_mat, 1e-300))

        # gamma[t, k] = P(S_t = k | x_1:T)
        log_gamma = log_alpha + log_beta - log_likelihood
        gamma = np.exp(log_gamma)
        gamma = gamma / np.sum(gamma, axis=1, keepdims=True)

        xi = np.zeros((T - 1, K, K), dtype=np.float64)
        for t in range(T - 1):
            # xi[t,i,j] proportional to:
            # alpha[t,i] * A[i,j] * emission[t+1,j] * beta[t+1,j]
            log_xi_t = (
                log_alpha[t][:, None]
                + log_trans
                + log_emissions[t + 1][None, :]
                + log_beta[t + 1][None, :]
                - log_likelihood
            )
            xi_t = np.exp(log_xi_t)
            denom = np.sum(xi_t)
            if denom <= 0:
                raise FloatingPointError("Encountered non-positive xi normalization constant.")
            xi[t] = xi_t / denom

        return gamma, xi

    # ---------------------------------------------------------------------
    # EM algorithm
    # ---------------------------------------------------------------------
    def _m_step(self, x: ArrayLike, gamma: ArrayLike, xi: ArrayLike) -> HMMParams:
        T, K = gamma.shape

        init_probs = np.maximum(gamma[0], 1e-16)
        init_probs = init_probs / np.sum(init_probs)

        trans_mat = np.sum(xi, axis=0)  # (K, K)
        trans_row_sums = np.sum(gamma[:-1], axis=0, keepdims=True).T  # (K,1)
        trans_mat = trans_mat / np.maximum(trans_row_sums, 1e-16)
        trans_mat = _normalize_rows(np.maximum(trans_mat, 1e-16))

        gamma_sums = np.sum(gamma, axis=0)  # (K,)
        means = np.sum(gamma * x[:, None], axis=0) / np.maximum(gamma_sums, 1e-16)

        centered_sq = (x[:, None] - means[None, :]) ** 2
        variances = np.sum(gamma * centered_sq, axis=0) / np.maximum(gamma_sums, 1e-16)
        variances = np.maximum(variances, self.min_variance)

        return HMMParams(
            init_probs=init_probs,
            trans_mat=trans_mat,
            means=means,
            variances=variances,
        )

    def fit(self, x: ArrayLike, verbose: bool = True) -> "GaussianHMM":
        x = _check_1d(x)

        params = self._initialize_params(x)
        self.log_likelihood_history_ = []

        prev_ll = -np.inf

        for iteration in range(1, self.max_iter + 1):
            log_emissions = self._compute_log_emission_probs(x, params)
            log_alpha, ll = self._forward_log(log_emissions, params)
            log_beta = self._backward_log(log_emissions, params)
            gamma, xi = self._smoothed_posteriors(log_alpha, log_beta, log_emissions, params, ll)

            params = self._m_step(x, gamma, xi)
            self.log_likelihood_history_.append(ll)

            improvement = ll - prev_ll
            if verbose:
                print(f"[EM] iter={iteration:03d} loglik={ll:.6f} improvement={improvement:.6e}")

            if iteration > 1 and abs(improvement) < self.tol:
                if verbose:
                    print(f"[EM] Converged at iteration {iteration}.")
                break

            if ll < prev_ll - 1e-8:
                # Small numerical drops can happen, but persistent drops would be a concern.
                if verbose:
                    print("[EM] Warning: log-likelihood decreased slightly; continuing.")
            prev_ll = ll

        self.params_ = params
        self.is_fitted_ = True
        return self

    # ---------------------------------------------------------------------
    # Inference
    # ---------------------------------------------------------------------
    def score(self, x: ArrayLike) -> float:
        self._require_fitted()
        x = _check_1d(x)
        log_emissions = self._compute_log_emission_probs(x, self.params_)
        _, ll = self._forward_log(log_emissions, self.params_)
        return ll

    def predict_proba(self, x: ArrayLike) -> ArrayLike:
        """
        Smoothed state probabilities gamma_t(k) = P(S_t=k | x_1:T)
        """
        self._require_fitted()
        x = _check_1d(x)

        log_emissions = self._compute_log_emission_probs(x, self.params_)
        log_alpha, ll = self._forward_log(log_emissions, self.params_)
        log_beta = self._backward_log(log_emissions, self.params_)
        gamma, _ = self._smoothed_posteriors(log_alpha, log_beta, log_emissions, self.params_, ll)
        return gamma

    def predict(self, x: ArrayLike) -> ArrayLike:
        """
        Most likely state path via Viterbi.
        """
        self._require_fitted()
        x = _check_1d(x)
        return self._viterbi(x)

    def filtered_state_probabilities(self, x: ArrayLike) -> ArrayLike:
        """
        P(S_t = k | x_1:t)
        """
        self._require_fitted()
        x = _check_1d(x)

        log_emissions = self._compute_log_emission_probs(x, self.params_)
        log_alpha, _ = self._forward_log(log_emissions, self.params_)
        log_norm = _logsumexp(log_alpha, axis=1, keepdims=True)
        filtered = np.exp(log_alpha - log_norm)
        return filtered

    def _viterbi(self, x: ArrayLike) -> ArrayLike:
        params = self.params_
        log_emissions = self._compute_log_emission_probs(x, params)
        T, K = log_emissions.shape

        log_init = np.log(np.maximum(params.init_probs, 1e-300))
        log_trans = np.log(np.maximum(params.trans_mat, 1e-300))

        delta = np.zeros((T, K), dtype=np.float64)
        psi = np.zeros((T, K), dtype=np.int64)

        delta[0] = log_init + log_emissions[0]

        for t in range(1, T):
            for j in range(K):
                vals = delta[t - 1] + log_trans[:, j]
                psi[t, j] = int(np.argmax(vals))
                delta[t, j] = vals[psi[t, j]] + log_emissions[t, j]

        states = np.zeros(T, dtype=np.int64)
        states[-1] = int(np.argmax(delta[-1]))

        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states

    # ---------------------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------------------
    def get_params(self) -> Dict[str, ArrayLike]:
        self._require_fitted()
        return {
            "init_probs": self.params_.init_probs.copy(),
            "trans_mat": self.params_.trans_mat.copy(),
            "means": self.params_.means.copy(),
            "variances": self.params_.variances.copy(),
            "std_devs": np.sqrt(self.params_.variances.copy()),
        }

    def _require_fitted(self) -> None:
        if not self.is_fitted_ or self.params_ is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
