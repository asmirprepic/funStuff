from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from utils import ensure_dir

def plot_rmse_vs_eps_location(df: pd.DataFrame, out_dir: str, dpi: int = 160) -> None:
    ensure_dir(out_dir)
    out_path = Path(out_dir) / "figure1_rmse_location.png"

    n0 = int(df["n"].max())
    ratio0 = float(df["ratio"])
    sub = df[(df["n"] == n0) & (df["ratio"] == ratio0)].copt()

    fig = plt.figure()
    for p in sorted(sub["p"].unique()):
        ss = sub[sub["p"] == p].sort_values("eps")
        plt.plot(ss["eps"],ss["mean_rmse"],marker = "o", label= f"Mean (p =. {p:.2f })")
        plt.plot(ss["eps"], ss["median_rmse"], marker="o", label=f"Median (p={p:.2f})")

    plt.xlabel("Contamination rate ε")
    plt.ylabel("RMSE (location)")
    plt.title(f"RMSE vs ε (n={n0}, σ2/σ1={ratio0:.0f})")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

def plot_rmse_vs_eps_scale(df: pd.DataFrame, out_dir: str, dpi: int = 160) -> None:
    ensure_dir(out_dir)
    out_path = Path(out_dir) / "figure2_rmse_scale.png"

    n0 = int(df["n"].max())
    ratio0 = float(df["ratio"].max())
    sub = df[(df["n"] == n0) & (df["ratio"] == ratio0)].copy()

    fig = plt.figure()
    for p in sorted(sub["p"].unique()):
        ss = sub[sub["p"] == p].sort_values("eps")
        plt.plot(ss["eps"], ss["sd_rmse"], marker="o", label=f"SD (p={p:.2f})")
        plt.plot(ss["eps"], ss["mad_rmse"], marker="o", label=f"MAD (p={p:.2f})")

    plt.xlabel("Contamination rate ε")
    plt.ylabel("RMSE (scale)")
    plt.title(f"RMSE vs ε (n={n0}, σ2/σ1={ratio0:.0f})")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
