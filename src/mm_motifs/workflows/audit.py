from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mm_motifs.config import ProjectConfig
from mm_motifs.data.load import iter_xport, variable_metadata
from mm_motifs.features.conditions import (
    condition_registry_table,
    condition_sources,
    derive_conditions,
    primary_conditions,
)
from mm_motifs.features.populations import state_name
from mm_motifs.features.socioeconomic import derive_ses, ses_registry_table, ses_source
from mm_motifs.runtime import (
    configure_logging,
    manifest,
    output_directory,
    raw_xpt_path,
    write_csv,
    write_json,
)
from mm_motifs.statistics.survey import valid_weights


def _sum_frames(frames: list[pd.DataFrame], keys: list[str]) -> pd.DataFrame:
    combined = pd.concat(frames, ignore_index=True)
    numeric = [column for column in combined.columns if column not in keys]
    return combined.groupby(keys, as_index=False, dropna=False)[numeric].sum()


def _prevalence_chunk(
    state: pd.Series,
    weights: pd.Series,
    conditions: pd.DataFrame,
) -> pd.DataFrame:
    frames = []
    numeric_weights = pd.to_numeric(weights, errors="coerce")
    weight_ok = valid_weights(numeric_weights)
    for condition in conditions:
        values = pd.to_numeric(conditions[condition], errors="coerce")
        valid = values.notna() & weight_ok
        cases = valid & (values == 1)
        work = pd.DataFrame(
            {
                "geography_code": state,
                "n_valid": valid.astype(int),
                "n_cases": cases.astype(int),
                "valid_weight": np.where(valid, numeric_weights, 0.0),
                "case_weight": np.where(cases, numeric_weights, 0.0),
                "weight_squared": np.where(valid, np.square(numeric_weights), 0.0),
            }
        )
        by_state = work.groupby("geography_code", as_index=False, dropna=False).sum()
        by_state["condition"] = condition
        national = pd.DataFrame(
            [
                {
                    "geography_code": "ALL",
                    "n_valid": int(valid.sum()),
                    "n_cases": int(cases.sum()),
                    "valid_weight": float(work["valid_weight"].sum()),
                    "case_weight": float(work["case_weight"].sum()),
                    "weight_squared": float(work["weight_squared"].sum()),
                    "condition": condition,
                }
            ]
        )
        frames.extend([by_state, national])
    return pd.concat(frames, ignore_index=True)


def _analysis_roles(config: ProjectConfig) -> dict[str, str]:
    roles: dict[str, list[str]] = defaultdict(list)
    for condition in primary_conditions(config.conditions):
        source = condition["source_by_year"][str(config.year)]
        values = source if isinstance(source, list) else [source]
        for value in values:
            roles[value].append(f"condition:{condition['canonical_name']}")
    for name in config.analysis["socioeconomic"]["definitions"]:
        roles[ses_source(config.analysis, name, config.year)].append(f"ses:{name}")
    for role, variable in config.analysis["survey_design"].items():
        roles[variable].append(f"survey:{role}")
    return {variable: "|".join(values) for variable, values in roles.items()}


def run_audit(config: ProjectConfig, verbose: bool = False) -> Path:
    source = raw_xpt_path(config)
    if not source.exists():
        raise FileNotFoundError(f"BRFSS XPT file not found: {source}")
    output = output_directory(config, "audit")
    logger = configure_logging(output / "run.log", verbose)
    logger.info("Starting BRFSS %s audit", config.year)

    metadata = variable_metadata(source)
    available = set(metadata["variable"])
    required_conditions = set(condition_sources(config.conditions, config.year))
    missing_sources = sorted(required_conditions - available)
    if missing_sources:
        raise KeyError(f"Condition variables missing from XPT: {missing_sources}")

    labels = dict(zip(metadata["variable"], metadata["label"], strict=True))
    system_missing = pd.Series(0, index=metadata["variable"], dtype="int64")
    total_rows = 0
    state_parts: list[pd.DataFrame] = []
    prevalence_parts: list[pd.DataFrame] = []
    analytic_missing: dict[tuple[str, str, str], int] = defaultdict(int)
    ses_counts: dict[tuple[str, str], int] = defaultdict(int)
    design = config.analysis["survey_design"]
    chunk_size = int(config.analysis["data"]["audit_chunk_size"])

    for chunk_number, chunk in enumerate(iter_xport(source, chunk_size), start=1):
        total_rows += len(chunk)
        system_missing = system_missing.add(chunk.isna().sum(), fill_value=0).astype("int64")
        state = pd.to_numeric(chunk[design["state"]], errors="coerce")
        weights = pd.to_numeric(chunk[design["weight"]], errors="coerce")
        weight_ok = valid_weights(weights)
        state_work = pd.DataFrame(
            {
                "geography_code": state,
                "n_unweighted": 1,
                "weighted_population_estimate": np.where(weight_ok, weights, 0.0),
                "n_invalid_weight": (~weight_ok).astype(int),
            }
        )
        state_parts.append(
            state_work.groupby("geography_code", as_index=False, dropna=False).sum()
        )

        derived = derive_conditions(chunk, config.conditions, config.year)
        prevalence_parts.append(_prevalence_chunk(state, weights, derived))
        for name in derived:
            analytic_missing[("condition", name, "")] += int(derived[name].isna().sum())

        for definition_name in config.analysis["socioeconomic"]["definitions"]:
            ses = derive_ses(chunk, config.analysis, config.year, definition_name)
            source_name = ses_source(config.analysis, definition_name, config.year)
            analytic_missing[("ses", definition_name, source_name)] += int(ses.isna().sum())
            for category, count in ses.value_counts(dropna=False).items():
                label = "missing" if pd.isna(category) else str(category)
                ses_counts[(definition_name, label)] += int(count)
        logger.info("Audited chunk %s (%s cumulative rows)", chunk_number, total_rows)

    expected = int(config.analysis["data"]["expected_records"][str(config.year)])
    if total_rows != expected:
        raise ValueError(f"Expected {expected} BRFSS rows, read {total_rows}")

    roles = _analysis_roles(config)
    metadata["analysis_roles"] = metadata["variable"].map(roles).fillna("")
    metadata["n_system_missing"] = metadata["variable"].map(system_missing).astype(int)
    metadata["system_missing_fraction"] = metadata["n_system_missing"] / total_rows

    states = _sum_frames(state_parts, ["geography_code"])
    states["geography_code"] = states["geography_code"].astype(int)
    states["geography"] = states["geography_code"].map(state_name)
    states = states[
        [
            "geography_code",
            "geography",
            "n_unweighted",
            "weighted_population_estimate",
            "n_invalid_weight",
        ]
    ].sort_values("geography_code")

    prevalence = _sum_frames(prevalence_parts, ["geography_code", "condition"])
    prevalence["unweighted_prevalence"] = prevalence["n_cases"] / prevalence["n_valid"]
    prevalence["weighted_prevalence"] = prevalence["case_weight"] / prevalence["valid_weight"]
    prevalence["kish_effective_n"] = (
        np.square(prevalence["valid_weight"]) / prevalence["weight_squared"]
    )
    prevalence["geography"] = prevalence["geography_code"].map(
        lambda value: "All reporting areas"
        if value == "ALL"
        else state_name(int(float(value)))
    )
    prevalence = prevalence.rename(
        columns={"valid_weight": "weighted_population_denominator"}
    ).drop(columns=["case_weight", "weight_squared"])
    prevalence = prevalence[
        [
            "geography_code",
            "geography",
            "condition",
            "n_valid",
            "n_cases",
            "unweighted_prevalence",
            "weighted_prevalence",
            "weighted_population_denominator",
            "kish_effective_n",
        ]
    ].sort_values(["geography_code", "condition"], key=lambda value: value.astype(str))

    candidates = condition_registry_table(
        config.conditions,
        config.year,
        available,
        labels,
    )
    national = prevalence[prevalence["geography_code"] == "ALL"][
        ["condition", "n_valid", "n_cases", "weighted_prevalence"]
    ]
    candidates = candidates.merge(
        national,
        left_on="canonical_condition_name",
        right_on="condition",
        how="left",
    ).drop(columns=["condition"])
    candidates["n_analytic_missing"] = candidates["canonical_condition_name"].map(
        lambda name: analytic_missing[("condition", name, "")]
    )
    candidates["analytic_missing_fraction"] = (
        candidates["n_analytic_missing"] / total_rows
    )
    selection_policy = config.conditions.get("selection_policy", {})
    selection_pass = (
        candidates["source_available"]
        & (
            candidates["weighted_prevalence"]
            >= float(
                selection_policy.get("minimum_national_weighted_prevalence", 0)
            )
        )
        & (
            candidates["analytic_missing_fraction"]
            <= float(selection_policy.get("maximum_analytic_missing_fraction", 1))
        )
    )
    candidates["prototype_selection_status"] = selection_pass.map(
        {True: "provisional_include", False: "review"}
    )

    ses_table = ses_registry_table(config.analysis, config.year, labels, available)
    ses_table["n_lower"] = ses_table["definition_name"].map(
        lambda name: ses_counts[(name, "lower")]
    )
    ses_table["n_higher"] = ses_table["definition_name"].map(
        lambda name: ses_counts[(name, "higher")]
    )
    ses_table["n_analytic_missing"] = ses_table["definition_name"].map(
        lambda name: analytic_missing[
            ("ses", name, ses_source(config.analysis, name, config.year))
        ]
    )
    ses_table["analytic_missing_fraction"] = ses_table["n_analytic_missing"] / total_rows

    survey_rows = []
    for role, variable in design.items():
        row = metadata.loc[metadata["variable"] == variable]
        survey_rows.append(
            {
                "role": role,
                "variable": variable,
                "label": row["label"].iloc[0] if not row.empty else None,
                "source_available": not row.empty,
                "n_system_missing": (
                    int(row["n_system_missing"].iloc[0]) if not row.empty else total_rows
                ),
                "system_missing_fraction": (
                    float(row["system_missing_fraction"].iloc[0]) if not row.empty else 1.0
                ),
            }
        )
    survey_table = pd.DataFrame(survey_rows)

    missing_rows = [
        {
            "item_type": "raw_variable",
            "item": row.variable,
            "source_variable": row.variable,
            "n_total": total_rows,
            "n_missing": int(row.n_system_missing),
            "missing_fraction": float(row.system_missing_fraction),
            "definition": "System missing only",
        }
        for row in metadata.itertuples()
    ]
    for (item_type, item, source_variable), count in analytic_missing.items():
        missing_rows.append(
            {
                "item_type": item_type,
                "item": item,
                "source_variable": source_variable,
                "n_total": total_rows,
                "n_missing": count,
                "missing_fraction": count / total_rows,
                "definition": "System missing plus configured unknown/refused codes",
            }
        )
    missingness = pd.DataFrame(missing_rows).sort_values(["item_type", "item"])

    write_csv(metadata, output / "variable_dictionary.csv")
    write_csv(candidates, output / "candidate_chronic_conditions.csv")
    write_csv(ses_table, output / "ses_variables.csv")
    write_csv(survey_table, output / "survey_design_variables.csv")
    write_csv(states, output / "state_sample_sizes.csv")
    write_csv(prevalence, output / "disease_prevalence.csv")
    write_csv(missingness, output / "missingness_report.csv")

    download_metadata_path = source.with_suffix(".download.json")
    download_metadata: dict[str, Any] | None = None
    if download_metadata_path.exists():
        download_metadata = json.loads(download_metadata_path.read_text(encoding="utf-8"))
    write_json(
        manifest(
            config,
            "audit",
            {
                "source_file": str(source.relative_to(config.project_root)),
                "source_download": download_metadata,
                "records_read": total_rows,
                "variables_read": len(metadata),
                "outputs": sorted(path.name for path in output.glob("*.csv")),
                "inference_scope": "descriptive weighted estimates; no variance inference",
            },
        ),
        output / "manifest.json",
    )
    logger.info("Audit complete: %s", output)
    return output
