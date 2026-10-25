"""Shared utilities for the paper experiments."""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

EXP_DIR  = Path(__file__).resolve().parent
RUN_DIR  = EXP_DIR / "runs"

UAV_DIR            = RUN_DIR / "uav_mpc"


def ensure_dirs() -> None:
    for p in (RUN_DIR, UAV_DIR):
        p.mkdir(parents=True, exist_ok=True)


def subdirs(base: Path) -> dict[str, Path]:
    dirs = {
        "base": base, "logs": base / "logs", "models": base / "models",
        "figures": base / "figures", "data": base / "data",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def read_json(path: Path, default):
    if not path.exists():
        return default
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def setup_logger(name: str, log_path: Path, *, console: bool = True) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h); h.close()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    if os.environ.get("EXP_LOG_TO_STDOUT_ONLY") == "1":
        sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
        logger.addHandler(sh); return logger
    fh = logging.FileHandler(log_path, mode="w"); fh.setFormatter(fmt)
    logger.addHandler(fh)
    if console:
        sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
        logger.addHandler(sh)
    return logger
