from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Tuple
import numpy as np
import pandas as pd

from config import ExperimentConfig
from dgp import simulate_switching_gaussian,contaminate_huber_option1,unconditional_sigma
from metrics import bias_rmse
from estimators import mean_est, median_est, sd_est, mad_est


def run_grid(cfg: ExperimentConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    rows = []

    for n in cfg.n_list:
        for eps in cfg.eps_list:
            for p in cfg.persitence_list:
                P = np.array(cfg.transition_matrix(p),dtype = float)
                for ratio in cfg.vol_ratio_list:
                    sigma1 = cfg.sigma1
                    sigma2 = cfg.sigma1 * float(ratio)

                    # truth
                    mu_true = cfg.mean
                    sigma_true = unconditional_sigma(P = P, sigma = sigma1, sigma2 = sigma2)

                    mean_vals = np.empty(cfg.reps)
                    med_vals = np.empty(cfg.reps)
                    sd_vals = np.empty(cfg.reps)
                    mad_vals = np.empty(cfg.reps)

                    for r in range(cfg.reps):
                        x, _s = simulate_switching_gaussian(
                            n=n, P=P, mean = cfg.mean, sigma1 = sigma1, sigma2 = sigma2, rng=rng
                        )
                        mean_vals[r] = mean_est(x)
                        med_vals[r] = median_est(x)
                        sd_vals[r] = sd_est(x, ddof = 1)
                        mad_vals[r] = mad_est(x)

                    # metrics
                    mean_bias, mean_rmse = bias_rmse(mean_vals,mu_true)
                    med_bias, med_rmse = bias_rmse(med_vals, mu_true)
                    sd_bias, sd_rmse = bias_rmse(sd_vals, sigma_true)
                    mad_bias, mad_rmse = bias_rmse(mad_vals, sigma_true)

                    rows.append(
                        dict(
                           n=int(n),
                            eps=float(eps),
                            p=float(p),
                            ratio=float(ratio),
                            sigma1=float(sigma1),
                            sigma2=float(sigma2),
                            outlier_k=float(cfg.outlier_k),
                            reps=int(cfg.reps),
                            mu_true=float(mu_true),
                            sigma_true=float(sigma_true),
                            mean_bias=float(mean_bias),
                            mean_rmse=float(mean_rmse),
                            median_bias=float(med_bias),
                            median_rmse=float(med_rmse),
                            sd_bias=float(sd_bias),
                            sd_rmse=float(sd_rmse),
                            mad_bias=float(mad_bias),
                            mad_rmse=float(mad_rmse),
                        )
                    )
    return pd.DataFrame(rows)


