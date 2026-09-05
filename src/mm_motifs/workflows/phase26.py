from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mm_motifs.config import ProjectConfig
from mm_motifs.data.harmonize import (
    harmonize_phase25,
    phase25_selected_columns,
)
from mm_motifs.data.load import read_selected
from mm_motifs.features.conditions import condition_names
from mm_motifs.features.populations import phase25_population_registry
from mm_motifs.graphs.calibration import similarity_summary
from mm_motifs.graphs.construct import node_prevalence_table
from mm_motifs.graphs.phase26 import (
    adjusted_concordance,
    build_phi_edge_scenarios,
    phase26_viability_report,
    point_phi_stability_diagnostic,
    threshold_topology,
)
from mm_motifs.graphs.qc import pairwise_similarity
from mm_motifs.runtime import (
    configure_logging,
    interim_directory,
    manifest,
    output_directory,
    raw_xpt_path,
    write_csv,
    write_json,
)
from mm_motifs.statistics.disease_association import estimate_edge_table
from mm_motifs.statistics.phi_bootstrap import run_phi_bootstrap
from mm_motifs.statistics.r_survey import r_survey_versions
from mm_motifs.visualization.calibration import draw_rule_density_distributions
from mm_motifs.visualization.networks import draw_similarity_heatmap
from mm_motifs.visualization.phase26 import (
    draw_point_phi_vs_selection_stability,
    draw_selection_stability,
    draw_threshold_trajectories,
)
from mm_motifs.workflows.phase25 import (
    _calibration_summary,
    _construct_graphs,
    _edge_support,
    _git_dirty,
    _json_safe,
    _population_frame,
    _sha256,
    _source_tree_sha256,
    _write_parquet,
    _write_r_csv,
)


def _selection_intervals(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    count = pd.to_numeric(
        result["bootstrap_replicates_valid"],
        errors="coerce",
    ).where(lambda values: values > 0)
    probability = pd.to_numeric(result["p_phi_ge_012"], errors="coerce")
    z_value = 1.959963984540054
    denominator = 1 + z_value**2 / count
    center = (probability + z_value**2 / (2 * count)) / denominator
    radius = (
        z_value
        * np.sqrt(
            probability * (1 - probability) / count
            + z_value**2 / (4 * count**2)
        )
        / denominator
    )
    result["selection_stability_standard_error"] = np.sqrt(
        probability * (1 - probability) / count
    )
    result["selection_stability_wilson_low"] = center - radius
    result["selection_stability_wilson_high"] = center + radius
    return result


def _bootstrap_jobs(
    pair_counts: pd.DataFrame,
    registry: pd.DataFrame,
) -> pd.DataFrame:
    metadata = registry[
        [
            "graph_id",
            "geography_code",
            "ses_definition",
            "ses_category",
        ]
    ].rename(columns={"geography_code": "state_code"})
    return pair_counts[
        [
            "graph_id",
            "source_condition",
            "target_condition",
            "association",
        ]
    ].rename(
        columns={
            "source_condition": "disease_a",
            "target_condition": "disease_b",
            "association": "point_phi",
        }
    ).merge(
        metadata,
        on="graph_id",
        how="left",
        validate="many_to_one",
    )[
        [
            "graph_id",
            "state_code",
            "ses_definition",
            "ses_category",
            "disease_a",
            "disease_b",
            "point_phi",
        ]
    ]


def _attach_pair_eligibility(
    bootstrap: pd.DataFrame,
    pair_counts: pd.DataFrame,
) -> pd.DataFrame:
    eligibility = pair_counts[
        [
            "graph_id",
            "source_condition",
            "target_condition",
            "pair_eligible",
            "exclusion_reason",
        ]
    ].rename(
        columns={
            "source_condition": "disease_a",
            "target_condition": "disease_b",
        }
    )
    return bootstrap.merge(
        eligibility,
        on=["graph_id", "disease_a", "disease_b"],
        how="left",
        validate="one_to_one",
    )


def _bootstrap_graph_summary(
    bootstrap: pd.DataFrame,
    point_threshold: float,
    stability_threshold: float,
) -> pd.DataFrame:
    values = bootstrap.copy()
    values["raw_selected"] = (
        values["pair_eligible"]
        & values["point_phi"].ge(point_threshold)
    )
    values["stable_selected"] = (
        values["raw_selected"]
        & values["p_phi_ge_012"].ge(stability_threshold)
        & values["bootstrap_status"].eq("ok")
        & values["bootstrap_replicates_valid"].eq(
            values["bootstrap_replicates_requested"]
        )
    )
    rows = []
    for graph_id, frame in values.groupby("graph_id", sort=True):
        raw_count = int(frame["raw_selected"].sum())
        stable_count = int(frame["stable_selected"].sum())
        raw_stabilities = frame.loc[
            frame["raw_selected"],
            "p_phi_ge_012",
        ]
        rows.append(
            {
                "graph_id": graph_id,
                "raw_phi_012_edge_count": raw_count,
                "stable_phi_012_edge_count": stable_count,
                "raw_edge_stable_fraction": (
                    stable_count / raw_count if raw_count else math.nan
                ),
                "minimum_raw_edge_selection_stability": (
                    float(raw_stabilities.min())
                    if not raw_stabilities.empty
                    else math.nan
                ),
                "median_raw_edge_selection_stability": (
                    float(raw_stabilities.median())
                    if not raw_stabilities.empty
                    else math.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _load_adjusted_baseline(
    config: ProjectConfig,
    settings: dict[str, Any],
    primary_ses: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    archive = config.resolve(settings["archive_directory"])
    manifest_path = archive / "manifest.json"
    edge_path = archive / "edge_table_all_rules.csv.gz"
    if not manifest_path.exists() or not edge_path.exists():
        raise FileNotFoundError(f"Missing frozen Phase 2.5 archive: {archive}")
    archived_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if archived_manifest["config_digest"] != settings["config_digest"]:
        raise ValueError("Phase 2.5 archive configuration digest mismatch")
    edges = pd.read_csv(edge_path, low_memory=False)
    adjusted = edges[
        edges["rule_id"].eq(settings["adjusted_rule_id"])
        & edges["ses_definition"].eq(primary_ses)
    ].copy()
    if adjusted.empty:
        raise ValueError("Frozen Phase 2.5 archive has no adjusted edges")
    return adjusted, archived_manifest


def run_phase26(
    config: ProjectConfig,
    verbose: bool = False,
    workers: int = 4,
) -> Path:
    source = raw_xpt_path(config)
    if not source.exists():
        raise FileNotFoundError(f"BRFSS XPT file not found: {source}")
    settings = config.graph["phase26"]
    analysis = config.analysis["phase25"]
    primary_ses = str(settings["primary_ses_definition"])
    primary_rule = str(settings["primary_rule_id"])
    stable_rule = str(settings["stable_rule_id"])
    primary_threshold = next(
        float(rule["minimum_effect"])
        for rule in settings["thresholds"]
        if str(rule["rule_id"]) == primary_rule
    )
    bootstrap_settings = settings["bootstrap"]
    base_output = output_directory(config, "phase26")
    output = base_output / "full"
    model_output = base_output / "r"
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    logger = configure_logging(output / "run.log", verbose)

    source_sha256 = _sha256(source)
    source_tree_sha256 = _source_tree_sha256(config.project_root)
    harmonized_cache_key = hashlib.sha256(
        f"{config.digest}\n{source_sha256}\n{source_tree_sha256}".encode()
    ).hexdigest()[:16]
    intermediate = interim_directory(config)
    parquet_path = (
        intermediate / f"phase26_harmonized_{harmonized_cache_key}.parquet"
    )
    r_csv_path = (
        intermediate / f"phase26_harmonized_{harmonized_cache_key}.csv"
    )
    geography_codes = [int(value) for value in analysis["geography_codes"]]
    if parquet_path.exists():
        harmonized = pd.read_parquet(parquet_path)
        logger.info("Loaded cached Phase 2.6 harmonized data")
    else:
        raw = read_selected(source, phase25_selected_columns(config))
        harmonized = harmonize_phase25(raw, config, geography_codes)
        del raw
        _write_parquet(harmonized, parquet_path)
        logger.info("Harmonized %s state/DC respondents", len(harmonized))
    if not r_csv_path.exists():
        _write_r_csv(harmonized, r_csv_path)

    registry = phase25_population_registry(
        harmonized,
        config.year,
        primary_ses,
        int(analysis["minimum_population_n"]),
        geography_codes,
    )
    write_csv(registry, output / "population_registry.csv")
    eligible_registry = registry[
        registry["eligibility_status"].eq("eligible")
    ].copy()
    logger.info(
        "Eligible education populations: %s",
        len(eligible_registry),
    )

    names = condition_names(config.conditions)
    pair_parts = []
    node_parts = []
    for row in eligible_registry.itertuples(index=False):
        population = _population_frame(harmonized, row)
        pairs = estimate_edge_table(
            population,
            names,
            row.graph_id,
            config.graph["edge_criteria"],
        )
        pairs["ses_definition"] = row.ses_definition
        pairs["ses_category"] = row.ses_category
        nodes = node_prevalence_table(population, names, row.graph_id)
        nodes["ses_definition"] = row.ses_definition
        nodes["ses_category"] = row.ses_category
        pair_parts.append(pairs)
        node_parts.append(nodes)
    pair_counts = pd.concat(pair_parts, ignore_index=True)
    nodes = pd.concat(node_parts, ignore_index=True)
    write_csv(pair_counts, output / "pair_estimates_weighted_phi.csv")
    write_csv(nodes, output / "disease_prevalence_by_graph.csv")

    jobs = _bootstrap_jobs(pair_counts, eligible_registry)
    r_versions = r_survey_versions(config.project_root)
    bootstrap = run_phi_bootstrap(
        project_root=config.project_root,
        harmonized_csv=r_csv_path,
        jobs=jobs,
        output_directory=model_output,
        replicates=int(bootstrap_settings["replicates"]),
        threshold=primary_threshold,
        seed=int(bootstrap_settings["seed"]),
        workers=workers,
        runtime_versions=r_versions,
    )
    bootstrap_cache_key = bootstrap.attrs.get("model_cache_key")
    bootstrap = _selection_intervals(
        _attach_pair_eligibility(bootstrap, pair_counts)
    )
    write_csv(bootstrap, output / "phi_bootstrap_edge_stability.csv")
    bootstrap_summary = _bootstrap_graph_summary(
        bootstrap,
        primary_threshold,
        float(bootstrap_settings["selection_stability_threshold"]),
    )
    write_csv(
        bootstrap_summary,
        output / "phi_bootstrap_graph_summary.csv",
    )
    draw_selection_stability(
        bootstrap,
        output / "figures" / "phi_012_selection_stability.png",
        primary_threshold,
        float(bootstrap_settings["selection_stability_threshold"]),
        int(config.graph["visualization"]["dpi"]),
    )
    stability_bands, stability_diagnostic = point_phi_stability_diagnostic(
        bootstrap,
        primary_threshold,
        float(bootstrap_settings["selection_stability_threshold"]),
    )
    write_csv(
        stability_bands,
        output / "point_phi_stability_by_band.csv",
    )
    write_json(
        _json_safe(stability_diagnostic),
        output / "point_phi_stability_diagnostic.json",
    )
    draw_point_phi_vs_selection_stability(
        bootstrap,
        output / "figures" / "point_phi_vs_selection_stability.png",
        primary_threshold,
        float(bootstrap_settings["selection_stability_threshold"]),
        int(config.graph["visualization"]["dpi"]),
    )

    edge_scenarios = build_phi_edge_scenarios(
        pair_counts,
        bootstrap,
        settings["thresholds"],
        primary_rule,
        stable_rule,
        float(bootstrap_settings["selection_stability_threshold"]),
    )
    write_csv(edge_scenarios, output / "edge_table_all_rules.csv")
    (
        graph_collections,
        edge_collections,
        graph_statistics,
        graph_index,
    ) = _construct_graphs(
        harmonized,
        eligible_registry,
        nodes,
        edge_scenarios,
        config,
        output,
    )
    write_csv(graph_statistics, output / "graph_statistics.csv")
    write_csv(graph_index, output / "graph_database.csv")

    similarity_parts = []
    similarity_summary_parts = []
    for (ses_definition, rule_id), graphs in graph_collections.items():
        similarities = pairwise_similarity(
            graphs,
            edge_collections[(ses_definition, rule_id)],
            names,
        )
        similarities["ses_definition"] = ses_definition
        similarities["rule_id"] = rule_id
        similarity_parts.append(similarities)
        summary = similarity_summary(similarities)
        summary["ses_definition"] = ses_definition
        summary["rule_id"] = rule_id
        similarity_summary_parts.append(summary)
        draw_similarity_heatmap(
            similarities,
            list(graphs),
            output / "figures" / f"{rule_id}_edge_jaccard_heatmap.png",
            int(config.graph["visualization"]["dpi"]),
            show_labels=False,
            title=f"Education — {rule_id} edge-set similarity",
        )
    similarities = pd.concat(similarity_parts, ignore_index=True)
    similarity_summaries = pd.concat(
        similarity_summary_parts,
        ignore_index=True,
    )
    write_csv(similarities, output / "structural_similarity.csv")
    write_csv(
        similarity_summaries,
        output / "similarity_comparison_summary.csv",
    )

    support = _edge_support(graph_collections, names)
    calibration = _calibration_summary(
        graph_statistics,
        settings["viability"],
    )
    write_csv(support, output / "edge_support_by_rule.csv")
    write_csv(calibration, output / "graph_rule_comparison.csv")
    (
        threshold_statistics,
        threshold_transitions,
        continuity_summary,
    ) = threshold_topology(
        graph_collections,
        primary_ses,
        settings["thresholds"],
    )
    write_csv(
        threshold_statistics,
        output / "threshold_graph_statistics.csv",
    )
    write_csv(
        threshold_transitions,
        output / "threshold_topology_transitions.csv",
    )
    write_csv(
        continuity_summary,
        output / "threshold_topology_summary.csv",
    )
    draw_threshold_trajectories(
        threshold_statistics,
        output / "figures" / "threshold_edge_trajectories.png",
        int(config.graph["visualization"]["dpi"]),
    )

    adjusted, phase25_manifest = _load_adjusted_baseline(
        config,
        settings["phase25_baseline"],
        primary_ses,
    )
    concordance_edges, concordance_summary = adjusted_concordance(
        edge_scenarios,
        adjusted,
        [primary_rule, stable_rule],
    )
    write_csv(
        concordance_edges,
        output / "adjusted_concordance_edges.csv",
    )
    write_csv(
        concordance_summary,
        output / "adjusted_concordance_summary.csv",
    )

    draw_rule_density_distributions(
        graph_statistics,
        output / "figures" / "graph_density_by_rule.png",
    )
    viability_settings = {
        **settings["viability"],
        **settings["topology_continuity"],
        **settings["adjusted_validation"],
        "selection_stability_threshold": float(
            bootstrap_settings["selection_stability_threshold"]
        ),
    }
    viability = phase26_viability_report(
        graph_statistics,
        similarities,
        support,
        bootstrap,
        continuity_summary,
        concordance_summary,
        stable_rule,
        primary_rule,
        viability_settings,
    )
    write_json(_json_safe(viability), output / "viability_report.json")
    write_json(
        _json_safe(
            manifest(
                config,
                "phase26",
                {
                    "phase25_baseline_config_digest": phase25_manifest[
                        "config_digest"
                    ],
                    "source_data_sha256": source_sha256,
                    "source_tree_sha256": source_tree_sha256,
                    "harmonized_cache_key": harmonized_cache_key,
                    "git_dirty": _git_dirty(config.project_root),
                    "population_definition": (
                        "available US state × education-primary SES"
                    ),
                    "eligible_population_graphs": len(eligible_registry),
                    "r_runtime": r_versions,
                    "phi_bootstrap_cache_key": bootstrap_cache_key,
                    "survey_lonely_psu": "adjust",
                    "survey_adjust_domain_lonely": True,
                    "bootstrap_replicates": int(
                        bootstrap_settings["replicates"]
                    ),
                    "point_phi_threshold": primary_threshold,
                    "selection_stability_threshold": float(
                        bootstrap_settings[
                            "selection_stability_threshold"
                        ]
                    ),
                    "threshold_neighborhood": [
                        float(rule["minimum_effect"])
                        for rule in settings["thresholds"]
                    ],
                    "adjusted_models_role": "validation_only",
                    "edge_semantics": (
                        "Undirected survey-weighted phi association; "
                        "not causal or directional"
                    ),
                    "decision_excludes_ses_significance": True,
                    "phase_boundary": (
                        "Stopped before frequent subgraph mining"
                    ),
                },
            )
        ),
        output / "manifest.json",
    )
    logger.info("Phase 2.6 complete: %s", output)
    return output
