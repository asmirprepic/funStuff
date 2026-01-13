from __future__ import annotations
import os
from pathlib import Path

def ensure_dir(path: str | os.PathLike) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

