from __future__ import annotations
from pathlib import Path
import pandas as pd
from .utils import ensure_dir


def make_tables(df: pd.DataFrame, out_dir: str) -> None:
    ensure_dir(out_dir)
    out_dir_p = Path(out_dir)

    t1 = (
        df[df["eps"]== 0.0]
        .loc[:, ["n","ratio","mean_rmse","median_rmse","sd_rmse","mad_rmse"]]
        .sort_values(["n","ratio","p"])
        .reset_index(drop = True)
    )

    t1.to_csv(out_dir_p/"table1_eps0.csv",index=False)

    eps_max = float(df["eps"].max())
    t2 = (
        df[df["eps"]==eps_max]
        .loc[:, ["n", "p", "ratio", "mean_rmse", "median_rmse", "sd_rmse", "mad_rmse"]]
        .ort_values(["n", "ratio", "p"])
        .reset_index(drop=True)
    )
    t2.to_csv(out_dir_p / f"table2_eps{eps_max:.3f}.csv", index=False)
