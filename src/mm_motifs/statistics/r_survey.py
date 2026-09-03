from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from mm_motifs.runtime import write_csv


def _empty_model_results(jobs: pd.DataFrame, mode: str) -> pd.DataFrame:
    point_columns = [
        "model_status",
        "model_message",
        "log_odds_ratio",
        "standard_error",
        "confidence_low_log",
        "confidence_high_log",
        "p_value",
        "converged",
        "reverse_model_status",
        "reverse_model_message",
        "reverse_log_odds_ratio",
        "reverse_standard_error",
        "reverse_p_value",
        "reverse_converged",
        "direction_concordant",
    ]
    bootstrap_columns = [
        "bootstrap_status",
        "bootstrap_message",
        "bootstrap_log_odds_ratio",
        "bootstrap_replicates_requested",
        "bootstrap_replicates_valid",
        "positive_stability",
        "bootstrap_low_log",
        "bootstrap_high_log",
    ]
    columns = point_columns if mode == "point" else bootstrap_columns
    return jobs.reindex(columns=[*jobs.columns, *columns]).copy()


def resolve_rscript(project_root: Path) -> Path:
    configured = os.environ.get("MM_MOTIFS_RSCRIPT")
    candidates = [
        Path(configured) if configured else None,
        project_root / ".conda" / "r-survey" / "bin" / "Rscript",
        Path(shutil.which("Rscript")) if shutil.which("Rscript") else None,
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    raise RuntimeError(
        "Rscript is required for Phase 2.5. Create the Conda environment from "
        "environment.yml or set MM_MOTIFS_RSCRIPT."
    )


def r_survey_versions(project_root: Path) -> dict[str, str]:
    rscript = resolve_rscript(project_root)
    expression = (
        'cat(R.version.string, "\\n"); '
        'cat(as.character(packageVersion("survey")), "\\n"); '
        'cat(as.character(packageVersion("data.table")), "\\n")'
    )
    completed = subprocess.run(
        [str(rscript), "-e", expression],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) < 3:
        raise RuntimeError(f"Unable to read R package versions: {completed.stdout}")
    return {
        "r": lines[-3],
        "survey": lines[-2],
        "data_table": lines[-1],
        "rscript": str(rscript),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_cache_key(
    script: Path,
    harmonized_csv: Path,
    jobs: pd.DataFrame,
    mode: str,
    confidence_level: float,
    replicates: int,
    seed: int,
    runtime_versions: dict[str, str],
) -> str:
    payload = {
        "script_sha256": _file_sha256(script),
        "harmonized_sha256": _file_sha256(harmonized_csv),
        "jobs_csv": jobs.to_csv(index=False, lineterminator="\n"),
        "mode": mode,
        "confidence_level": confidence_level,
        "replicates": replicates,
        "seed": seed,
        "runtime_versions": runtime_versions,
    }
    encoded = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def _valid_model_results(
    result: pd.DataFrame,
    jobs: pd.DataFrame,
    mode: str,
) -> bool:
    keys = ["graph_id", "source_condition", "target_condition"]
    required = (
        {
            "model_status",
            "model_message",
            "log_odds_ratio",
            "standard_error",
            "confidence_low_log",
            "confidence_high_log",
            "p_value",
            "converged",
            "reverse_model_status",
            "reverse_model_message",
            "reverse_log_odds_ratio",
            "reverse_standard_error",
            "reverse_p_value",
            "reverse_converged",
            "direction_concordant",
        }
        if mode == "point"
        else {
            "bootstrap_status",
            "bootstrap_message",
            "bootstrap_log_odds_ratio",
            "bootstrap_replicates_requested",
            "bootstrap_replicates_valid",
            "positive_stability",
            "bootstrap_low_log",
            "bootstrap_high_log",
        }
    )
    if not set(keys).issubset(result) or not required.issubset(result):
        return False
    expected = jobs[keys].astype(str).agg("\r".join, axis=1).sort_values()
    observed = result[keys].astype(str).agg("\r".join, axis=1).sort_values()
    return len(result) == len(jobs) and expected.tolist() == observed.tolist()


def run_survey_models(
    project_root: Path,
    mode: str,
    harmonized_csv: Path,
    jobs: pd.DataFrame,
    output_directory: Path,
    confidence_level: float,
    replicates: int,
    seed: int,
    workers: int,
    reuse_existing: bool = True,
    runtime_versions: dict[str, str] | None = None,
) -> pd.DataFrame:
    if mode not in {"point", "bootstrap"}:
        raise ValueError(f"Unsupported R survey mode: {mode}")
    if jobs.empty:
        result = _empty_model_results(jobs, mode)
        result.attrs["model_cache_key"] = "empty"
        return result
    rscript = resolve_rscript(project_root)
    script = project_root / "scripts" / "adjusted_associations.R"
    versions = runtime_versions or r_survey_versions(project_root)
    cache_key = _model_cache_key(
        script,
        harmonized_csv,
        jobs,
        mode,
        confidence_level,
        replicates,
        seed,
        versions,
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    jobs_path = output_directory / f"{mode}_{cache_key}_jobs.csv"
    result_path = output_directory / f"{mode}_{cache_key}_model_results.csv"
    log_path = output_directory / f"{mode}_{cache_key}_r.log"
    if reuse_existing and jobs_path.exists() and result_path.exists():
        existing_jobs = pd.read_csv(jobs_path)
        if (
            list(existing_jobs.columns) == list(jobs.columns)
            and existing_jobs.astype(str).equals(
                jobs.reset_index(drop=True).astype(str)
            )
        ):
            existing_result = pd.read_csv(result_path)
            if _valid_model_results(existing_result, jobs, mode):
                existing_result.attrs["model_cache_key"] = cache_key
                return existing_result
    write_csv(jobs, jobs_path)
    command = [
        str(rscript),
        str(script),
        mode,
        str(harmonized_csv),
        str(jobs_path),
        str(result_path),
        str(confidence_level),
        str(replicates),
        str(seed),
        str(workers),
    ]
    environment = os.environ.copy()
    if workers > 1:
        environment["OPENBLAS_NUM_THREADS"] = "1"
        environment["OMP_NUM_THREADS"] = "1"
        environment["MKL_NUM_THREADS"] = "1"
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=environment,
    )
    log_path.write_text(
        f"command={' '.join(command)}\n"
        f"returncode={completed.returncode}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}\n",
        encoding="utf-8",
    )
    if completed.returncode:
        raise RuntimeError(
            f"R survey {mode} models failed; inspect {log_path}: "
            f"{completed.stderr.strip()}"
        )
    if not result_path.exists():
        raise RuntimeError(f"R survey {mode} models produced no result file")
    result = pd.read_csv(result_path)
    if not _valid_model_results(result, jobs, mode):
        raise RuntimeError(f"R survey {mode} result failed schema/key validation")
    result.attrs["model_cache_key"] = cache_key
    return result


def adjusted_edge_table(
    model_results: pd.DataFrame,
    pair_counts: pd.DataFrame,
    maximum_q_value: float,
    adjuster: Callable[[pd.Series], pd.Series],
    rule_id: str = "adjusted_fdr",
) -> pd.DataFrame:
    keys = ["graph_id", "source_condition", "target_condition"]
    count_columns = [
        *keys,
        "n_complete",
        "source_cases",
        "target_cases",
        "cooccurring_cases",
        "weighted_n",
        "kish_effective_n",
        "pair_eligible",
        "exclusion_reason",
    ]
    result = model_results.merge(pair_counts[count_columns], on=keys, how="left")
    result["odds_ratio"] = np.exp(result["log_odds_ratio"])
    result["reverse_odds_ratio"] = np.exp(result["reverse_log_odds_ratio"])
    result["confidence_low"] = np.exp(result["confidence_low_log"])
    result["confidence_high"] = np.exp(result["confidence_high_log"])
    result["q_value"] = result.groupby("graph_id")["p_value"].transform(
        lambda values: adjuster(values).to_numpy()
    )
    result["rule_id"] = rule_id
    result["association"] = result["log_odds_ratio"]
    result["association_scale"] = "log_odds_ratio"
    result["estimator"] = "r_survey_svyglm"
    result["inference_scope"] = "design_based"
    result["edge_present"] = (
        result["pair_eligible"].fillna(False)
        & result["model_status"].eq("ok")
        & result["converged"].fillna(False)
        & result["log_odds_ratio"].gt(0)
        & result["confidence_low"].gt(1)
        & result["q_value"].le(maximum_q_value)
    )
    return result
