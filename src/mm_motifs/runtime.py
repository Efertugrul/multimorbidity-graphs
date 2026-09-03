from __future__ import annotations

import json
import logging
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from mm_motifs import __version__
from mm_motifs.config import ProjectConfig


def raw_xpt_path(config: ProjectConfig) -> Path:
    raw_dir = config.resolve(config.analysis["data"]["raw_dir"])
    return raw_dir / str(config.year) / f"LLCP{config.year}.XPT"


def output_directory(config: ProjectConfig, phase: str) -> Path:
    root = config.resolve(config.analysis["outputs"]["root"])
    return root / phase / f"year={config.year}" / f"run={config.digest}"


def interim_directory(config: ProjectConfig) -> Path:
    root = config.resolve(config.analysis["outputs"]["interim"])
    return root / f"year={config.year}" / f"run={config.digest}"


def configure_logging(path: Path, verbose: bool = False) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("mm_motifs")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def git_commit(project_root: Path) -> str | None:
    try:
        value = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return value or None


def manifest(config: ProjectConfig, phase: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    criteria = config.graph["edge_criteria"]
    value: dict[str, Any] = {
        "phase": phase,
        "dataset_year": config.year,
        "condition_registry_version": config.conditions["registry_version"],
        "population_definition": f"state × {config.active_ses}",
        "edge_estimator": config.graph["estimator"]["name"],
        "edge_threshold": criteria["minimum_effect"],
        "multiple_testing_rule": (
            f"BH FDR q <= {criteria['maximum_q_value']}"
            if criteria["require_fdr"]
            else "not used for descriptive Phase 2 graph thresholding"
        ),
        "minimum_graph_sample_size": config.analysis["prototype"]["minimum_population_n"],
        "gspan_support": None,
        "software_version": __version__,
        "git_commit": git_commit(config.project_root),
        "random_seed": config.random_seed,
        "config_digest": config.digest,
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    if extra:
        value.update(extra)
    return value


def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)
