from __future__ import annotations
from pathlib import Path
import pandas as pd

from config import ExperimentConfig
from experiment import run_grid
from tables import make_tables
from utils import ensure_dir

def main() -> None:
    cfg = ExperimentConfig()

    ensure_dir(cfg.output_dir)
    ensure_dir(Path(cfg.output_dir)/"figures")
    ensure_dir(Path(cfg.output_dir)/"tables")

    df: pd.DataFrame = run_grid(cfg)
    df.to_csv(cfg.results_csv,index = False)

    # put plotting here

    make_tables(df, out_dir=str(Path(cfg.output_dir) / "tables"))

    print(f"Saved results to: {cfg.results_csv}")
    print(f"Saved figures to: {Path(cfg.output_dir) / 'figures'}")
    print(f"Saved tables to: {Path(cfg.output_dir) / 'tables'}")

if __name__ == "__main__":
    main()
