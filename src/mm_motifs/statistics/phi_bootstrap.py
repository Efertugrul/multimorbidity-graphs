from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pandas as pd

from mm_motifs.runtime import write_csv
from mm_motifs.statistics.r_survey import resolve_rscript, r_survey_versions


RESULT_COLUMNS = [
    "graph_id",
    "state_code",
    "ses_definition",
    "ses_category",
    "disease_a",
    "disease_b",
    "point_phi",
    "p_phi_gt_zero",
    "p_phi_ge_012",
    "bootstrap_selection_mask",
    "bootstrap_phi_median",
    "bootstrap_phi_025",
    "bootstrap_phi_975",
    "bootstrap_replicates_requested",
    "bootstrap_replicates_valid",
    "bootstrap_status",
    "bootstrap_message",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_key(
    script: Path,
    harmonized_csv: Path,
    jobs: pd.DataFrame,
    replicates: int,
    threshold: float,
    seed: int,
    runtime_versions: dict[str, str],
) -> str:
    payload = {
        "script_sha256": _sha256(script),
        "harmonized_sha256": _sha256(harmonized_csv),
        "jobs_csv": jobs.to_csv(index=False, lineterminator="\n"),
        "replicates": replicates,
        "threshold": threshold,
        "seed": seed,
        "runtime_versions": runtime_versions,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()[:16]


def _valid_results(
    result: pd.DataFrame,
    jobs: pd.DataFrame,
    replicates: int,
) -> bool:
    keys = ["graph_id", "disease_a", "disease_b"]
    if not set(RESULT_COLUMNS).issubset(result):
        return False
    expected = jobs[keys].astype(str).agg("\r".join, axis=1).sort_values()
    observed = result[keys].astype(str).agg("\r".join, axis=1).sort_values()
    masks = result["bootstrap_selection_mask"].astype(str)
    mask_values = masks.str.removeprefix("b")
    return (
        len(result) == len(jobs)
        and expected.tolist() == observed.tolist()
        and result["bootstrap_status"].isin(["ok", "partial"]).all()
        and result["bootstrap_replicates_requested"].eq(replicates).all()
        and result["bootstrap_replicates_valid"].gt(0).all()
        and result["bootstrap_replicates_valid"].le(replicates).all()
        and masks.str.startswith("b").all()
        and mask_values.str.len().eq(replicates).all()
        and mask_values.str.fullmatch(r"[01x]+").all()
        and mask_values.str.count(r"[01]").eq(
            result["bootstrap_replicates_valid"]
        ).all()
    )


def run_phi_bootstrap(
    project_root: Path,
    harmonized_csv: Path,
    jobs: pd.DataFrame,
    output_directory: Path,
    replicates: int,
    threshold: float,
    seed: int,
    workers: int,
    reuse_existing: bool = True,
    runtime_versions: dict[str, str] | None = None,
) -> pd.DataFrame:
    if jobs.empty:
        result = pd.DataFrame(columns=RESULT_COLUMNS)
        result.attrs["model_cache_key"] = "empty"
        return result
    rscript = resolve_rscript(project_root)
    script = project_root / "scripts" / "phi_bootstrap.R"
    versions = runtime_versions or r_survey_versions(project_root)
    cache_key = _cache_key(
        script,
        harmonized_csv,
        jobs,
        replicates,
        threshold,
        seed,
        versions,
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    jobs_path = output_directory / f"phi_{cache_key}_jobs.csv"
    result_path = output_directory / f"phi_{cache_key}_results.csv"
    log_path = output_directory / f"phi_{cache_key}_r.log"
    if reuse_existing and jobs_path.exists() and result_path.exists():
        existing_jobs = pd.read_csv(jobs_path)
        if (
            list(existing_jobs.columns) == list(jobs.columns)
            and existing_jobs.astype(str).equals(
                jobs.reset_index(drop=True).astype(str)
            )
        ):
            existing_result = pd.read_csv(result_path)
            if _valid_results(existing_result, jobs, replicates):
                existing_result.attrs["model_cache_key"] = cache_key
                return existing_result
    write_csv(jobs, jobs_path)
    command = [
        str(rscript),
        str(script),
        str(harmonized_csv),
        str(jobs_path),
        str(result_path),
        str(replicates),
        str(threshold),
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
            f"R phi bootstrap failed; inspect {log_path}: "
            f"{completed.stderr.strip()}"
        )
    result = pd.read_csv(result_path)
    if not _valid_results(result, jobs, replicates):
        raise RuntimeError("R phi bootstrap result failed schema/key validation")
    result.attrs["model_cache_key"] = cache_key
    return result
