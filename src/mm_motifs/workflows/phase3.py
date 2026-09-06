from __future__ import annotations

import hashlib
import json
import math
import os
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
from mm_motifs.motifs.mining import (
    annotate_opportunity_support,
    annotate_redundancy,
    edge_universe,
    graph_database_from_edges,
    graph_edge_masks,
    graph_opportunity_masks,
    mine_frequent_motifs,
    motif_edge_masks,
    occurrence_matrix,
    paired_graph_indices,
    write_gspan_database,
)
from mm_motifs.motifs.robustness import (
    bootstrap_discovery_set_stability,
    bootstrap_graph_masks,
    bootstrap_motif_support,
    density_null_support,
    frozen_vocabulary_stability,
    motif_shape_summary,
    paired_ses_permutation,
    restrict_to_frozen_edges,
)
from mm_motifs.runtime import (
    configure_logging,
    interim_directory,
    manifest,
    output_directory,
    raw_xpt_path,
    write_csv,
    write_json,
)
from mm_motifs.statistics.phi_bootstrap import run_phi_bootstrap
from mm_motifs.statistics.r_survey import r_survey_versions
from mm_motifs.visualization.phase3 import (
    draw_density_null,
    draw_discovery_set_stability,
    draw_motif_bootstrap_stability,
    draw_motif_support_spectrum,
    draw_ses_permutation,
)
from mm_motifs.workflows.phase25 import (
    _git_dirty,
    _json_safe,
    _sha256,
    _source_tree_sha256,
    _write_r_csv,
)
from mm_motifs.workflows.phase26 import (
    _attach_pair_eligibility,
    _bootstrap_jobs,
)


def _write_gzip_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(
        temporary,
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    os.replace(temporary, path)


def _load_phase26_archive(
    config: ProjectConfig,
    settings: dict[str, Any],
) -> tuple[
    Path,
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    archive = config.resolve(settings["archive_directory"])
    manifest_path = archive / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing Phase 2.6 archive: {archive}")
    checksum_path = archive / "SHA256SUMS"
    if not checksum_path.exists():
        raise FileNotFoundError(f"Missing Phase 2.6 checksums: {archive}")
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        expected, relative_path = line.split("  ", maxsplit=1)
        archived_path = archive / relative_path
        if not archived_path.exists() or _sha256(archived_path) != expected:
            raise ValueError(
                f"Phase 2.6 archive checksum mismatch: {relative_path}"
            )
    phase26_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if phase26_manifest["config_digest"] != settings["config_digest"]:
        raise ValueError("Phase 2.6 archive configuration digest mismatch")
    edge_table = pd.read_csv(
        archive / "edge_table_all_rules.csv.gz",
        low_memory=False,
    )
    registry = pd.read_csv(archive / "population_registry.csv")
    pair_estimates = pd.read_csv(
        archive / "pair_estimates_weighted_phi.csv.gz",
        low_memory=False,
    )
    return archive, phase26_manifest, edge_table, registry, pair_estimates


def _validate_frozen_graph(
    edge_table: pd.DataFrame,
    stable_rule_id: str,
    graph_definition: dict[str, Any],
) -> None:
    stable = edge_table[edge_table["rule_id"].eq(stable_rule_id)]
    if stable.empty:
        raise ValueError("Frozen Phase 2.6 stable rule is absent")
    selected = stable[stable["edge_present"].fillna(False)]
    valid = (
        selected["pair_eligible"].fillna(False)
        & selected["association"].ge(
            float(graph_definition["point_phi_threshold"])
        )
        & selected["p_phi_ge_012"].ge(
            float(graph_definition["minimum_selection_stability"])
        )
        & selected["bootstrap_status"].eq("ok")
        & selected["bootstrap_replicates_valid"].eq(
            selected["bootstrap_replicates_requested"]
        )
    )
    if not valid.all():
        raise ValueError("Frozen graph violates its stable edge definition")


def _resolve_harmonized_csv(
    config: ProjectConfig,
    phase26_manifest: dict[str, Any],
    logger: Any,
) -> Path:
    baseline_path = (
        config.resolve(config.analysis["outputs"]["interim"])
        / f"year={config.year}"
        / f"run={phase26_manifest['config_digest']}"
        / (
            "phase26_harmonized_"
            f"{phase26_manifest['harmonized_cache_key']}.csv"
        )
    )
    if baseline_path.exists():
        logger.info("Using frozen-run harmonized survey cache")
        return baseline_path
    source = raw_xpt_path(config)
    source_sha256 = _sha256(source)
    source_tree_sha256 = _source_tree_sha256(config.project_root)
    cache_key = hashlib.sha256(
        f"{config.digest}\n{source_sha256}\n{source_tree_sha256}".encode()
    ).hexdigest()[:16]
    target = interim_directory(config) / f"phase3_harmonized_{cache_key}.csv"
    if target.exists():
        logger.info("Using Phase 3 harmonized survey cache")
        return target
    geography_codes = [
        int(value) for value in config.analysis["phase25"]["geography_codes"]
    ]
    raw = read_selected(source, phase25_selected_columns(config))
    harmonized = harmonize_phase25(raw, config, geography_codes)
    _write_r_csv(harmonized, target)
    logger.info("Harmonized %s respondents for Phase 3", len(harmonized))
    return target


def _summarize_stability(
    stability: pd.DataFrame,
    support_fractions: list[float],
    probability_cutpoints: list[float],
) -> pd.DataFrame:
    rows = []
    for fraction in sorted(support_fractions):
        suffix = int(round(fraction * 100))
        column = f"p_boot_support_{int(round(fraction * 100)):02d}pct"
        frame = stability[stability[f"support_{suffix:02d}pct"]]
        for node_count, group in frame.groupby("node_count", sort=True):
            row = {
                "support_fraction": fraction,
                "minimum_support_count": int(
                    group[f"support_count_{suffix:02d}pct"].min()
                ),
                "maximum_support_count": int(
                    group[f"support_count_{suffix:02d}pct"].max()
                ),
                "node_count": int(node_count),
                "baseline_motif_count": len(group),
                "median_bootstrap_support_probability": float(
                    group[column].median()
                ),
                "minimum_bootstrap_support_probability": float(
                    group[column].min()
                ),
            }
            for cutoff in probability_cutpoints:
                row[
                    f"fraction_probability_ge_{int(round(cutoff * 100)):02d}"
                ] = float(group[column].ge(cutoff).mean())
            rows.append(row)
    return pd.DataFrame(rows)


def _summarize_discovery_stability(
    discovery: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for fraction, frame in discovery.groupby(
        "support_fraction",
        sort=True,
    ):
        rows.append(
            {
                "support_fraction": fraction,
                "bootstrap_replicates": len(frame),
                "median_discovered_motif_count": float(
                    frame["replicate_discovered_motif_count"].median()
                ),
                "median_baseline_retention": float(
                    frame["discovery_set_retention"].median()
                ),
                "q025_baseline_retention": float(
                    frame["discovery_set_retention"].quantile(0.025)
                ),
                "q975_baseline_retention": float(
                    frame["discovery_set_retention"].quantile(0.975)
                ),
                "median_discovery_set_jaccard": float(
                    frame["discovery_set_jaccard"].median()
                ),
                "median_novel_motif_count": float(
                    frame["novel_motif_count"].median()
                ),
            }
        )
    return pd.DataFrame(rows)


def _feasibility_report(
    graph_index: pd.DataFrame,
    motifs: pd.DataFrame,
    shape_summary: pd.DataFrame,
    stability: pd.DataFrame,
    stability_summary: pd.DataFrame,
    discovery_summary: pd.DataFrame,
    threshold_sensitivity_summary: pd.DataFrame,
    density_null: pd.DataFrame,
    degree_diagnostics: pd.DataFrame,
    ses_results: pd.DataFrame,
    support_fractions: list[float],
    primary_support_fraction: float,
    minimum_paired_states: int,
) -> dict[str, Any]:
    primary_probability = (
        f"p_boot_support_{int(round(primary_support_fraction * 100)):02d}pct"
    )
    primary_stability = stability[stability["primary_support"]]
    fixed_null = density_null[
        density_null["null_model"].eq("fixed_eligible_dyad_edge_count")
    ]
    degree_null = density_null[
        density_null["null_model"].eq(
            "degree_preserving_eligible_dyad_edge_swap"
        )
    ]
    degree_groups = degree_diagnostics.groupby("graph_index", sort=False)
    unique_realizations = degree_groups["randomized_edge_mask"].nunique()
    changed_fraction = degree_groups["changed"].mean()
    swap_completion = np.divide(
        degree_diagnostics["accepted_swaps"],
        degree_diagnostics["target_swaps"],
        out=np.ones(len(degree_diagnostics), dtype=float),
        where=degree_diagnostics["target_swaps"].gt(0),
    )
    support_spectrum = {
        f"{fraction:.0%}": {
            "motif_count": int(
                motifs[
                    f"support_{int(round(fraction * 100)):02d}pct"
                ].sum()
            ),
            "minimum_support_count": int(
                motifs[
                    f"support_count_{int(round(fraction * 100)):02d}pct"
                ].min()
            ),
            "maximum_support_count": int(
                motifs[
                    f"support_count_{int(round(fraction * 100)):02d}pct"
                ].max()
            ),
        }
        for fraction in support_fractions
    }
    return {
        "phase26_retention_decision": "HOLD",
        "structural_feasibility_decision": "GO",
        "phase3_status": "EXPLORATORY_COMPLETE",
        "frozen_graph_definition": (
            "point phi >= 0.12 and P_boot(phi >= 0.12) >= 0.90"
        ),
        "graph_count": len(graph_index),
        "mean_edge_count": float(graph_index["edge_count"].mean()),
        "minimum_edge_count": int(graph_index["edge_count"].min()),
        "maximum_edge_count": int(graph_index["edge_count"].max()),
        "graphs_with_at_least_five_edges": int(
            graph_index["edge_count"].ge(5).sum()
        ),
        "graphs_with_five_node_connected_component": int(
            graph_index["largest_component_nodes"].ge(5).sum()
        ),
        "graphs_with_triangles": int(
            graph_index["triangle_count"].gt(0).sum()
        ),
        "support_spectrum": support_spectrum,
        "minimum_paired_states": minimum_paired_states,
        "paired_state_count_range": [
            int(motifs["paired_state_count"].min()),
            int(motifs["paired_state_count"].max()),
        ],
        "ineligible_graph_dyad_cells": int(
            graph_index["ineligible_dyad_count"].sum()
        ),
        "motif_count_total": len(motifs),
        "closed_motif_count": int(motifs["is_closed"].sum()),
        "maximal_motif_count": int(motifs["is_maximal"].sum()),
        "occurrence_class_count": int(
            motifs["occurrence_class_id"].nunique()
        ),
        "primary_support_fraction": primary_support_fraction,
        "primary_motif_count": len(primary_stability),
        "primary_motif_probability_median": float(
            primary_stability[primary_probability].median()
        ),
        "primary_motif_probability_ge_080_fraction": float(
            primary_stability[primary_probability].ge(0.80).mean()
        ),
        "primary_motif_probability_ge_080_count": int(
            primary_stability[primary_probability].ge(0.80).sum()
        ),
        "primary_motif_probability_ge_090_fraction": float(
            primary_stability[primary_probability].ge(0.90).mean()
        ),
        "primary_motif_probability_ge_090_count": int(
            primary_stability[primary_probability].ge(0.90).sum()
        ),
        "discovery_set_stability": discovery_summary.to_dict(
            orient="records"
        ),
        "raw_threshold_sensitivity": (
            threshold_sensitivity_summary.to_dict(orient="records")
        ),
        "fixed_edge_null_positive_z_fraction": float(
            fixed_null["density_null_z"].gt(0).mean()
        ),
        "degree_null_positive_z_fraction": float(
            degree_null["density_null_z"].gt(0).mean()
        ),
        "degree_null_minimum_unique_realizations": int(
            unique_realizations.min()
        ),
        "degree_null_minimum_changed_fraction": float(
            changed_fraction.min()
        ),
        "degree_null_minimum_swap_completion": float(
            swap_completion.min()
        ),
        "degree_null_median_proposal_acceptance": float(
            degree_diagnostics["acceptance_rate"].median()
        ),
        "exploratory_ses_motif_count": len(ses_results),
        "exploratory_ses_max_t_p_below_005": int(
            ses_results["permutation_max_t_p_value"].lt(0.05).sum()
        ),
        "automatic_motif_go_gate": None,
        "shape_summary_rows": len(shape_summary),
        "stability_summary_rows": len(stability_summary),
        "interpretation": (
            "Motif discovery, robustness, density nulls, and SES comparisons "
            "are exploratory. Density-null p-values are selection-conditional; "
            "SES contrasts require within-state exchangeability and are not "
            "causal. Density conditioning does not remove edge-detection "
            "precision differences; 2023 replication remains required."
        ),
        "bootstrap_boundary": (
            "An independent evaluation bank allows only frozen stable edges "
            "to drop at phi 0.12. This is conditional edge-retention "
            "robustness, not full outer-inner stable-pipeline rediscovery. "
            "All eligible threshold-crossing edges are evaluated separately."
        ),
        "motif_interpretation": (
            "Motifs are recurring patterns of pairwise disease associations, "
            "not respondent-level higher-order disease combinations."
        ),
    }


def run_phase3(
    config: ProjectConfig,
    verbose: bool = False,
    workers: int = 4,
) -> Path:
    settings = config.graph["phase3"]
    mining = settings["mining"]
    bootstrap_settings = settings["bootstrap"]
    density_settings = settings["density_null"]
    ses_settings = settings["ses_permutation"]
    output_root = output_directory(config, "phase3")
    output = output_root / "full"
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    logger = configure_logging(output / "run.log", verbose)

    (
        archive,
        phase26_manifest,
        edge_table,
        registry,
        pair_estimates,
    ) = _load_phase26_archive(config, settings["phase26_baseline"])
    _validate_frozen_graph(
        edge_table,
        str(settings["phase26_baseline"]["stable_rule_id"]),
        settings["graph_definition"],
    )
    graphs, condition_labels = graph_database_from_edges(
        edge_table,
        str(settings["phase26_baseline"]["stable_rule_id"]),
        int(mining["edge_label"]),
    )
    graph_ids = [str(graph.graph["graph_id"]) for graph in graphs]
    write_csv(condition_labels, output / "condition_label_mapping.csv")
    graph_index = write_gspan_database(
        graphs,
        output / "gspan_graph_database.txt",
    )
    graph_index = graph_index.merge(
        registry[
            [
                "graph_id",
                "geography_code",
                "geography",
                "ses_definition",
                "ses_category",
                "n_unweighted",
                "kish_effective_n",
            ]
        ],
        on="graph_id",
        how="left",
        validate="one_to_one",
    )
    logger.info("Loaded %s frozen stable graphs", len(graphs))

    fractions = [float(value) for value in mining["support_fractions"]]
    primary_fraction = float(mining["primary_support_fraction"])
    minimum_paired_states = int(mining["minimum_paired_states"])
    candidate_minimum_support = math.ceil(
        min(fractions) * 2 * minimum_paired_states
    )
    conditions = condition_labels["condition"].tolist()
    universe = edge_universe(conditions)
    edge_index = {edge: index for index, edge in enumerate(universe)}
    baseline_graph_masks = graph_edge_masks(graphs, edge_index)
    opportunity_masks = graph_opportunity_masks(
        edge_table,
        str(settings["phase26_baseline"]["stable_rule_id"]),
        graph_ids,
        edge_index,
    )
    if np.any(
        np.bitwise_and(baseline_graph_masks, opportunity_masks)
        != baseline_graph_masks
    ):
        raise RuntimeError("Frozen graph contains an ineligible dyad")
    graph_index["eligible_dyad_count"] = [
        int(int(mask).bit_count()) for mask in opportunity_masks
    ]
    graph_index["ineligible_dyad_count"] = (
        len(universe) - graph_index["eligible_dyad_count"]
    )
    write_csv(graph_index, output / "graph_database_index.csv")
    motifs, gspan_metadata = mine_frequent_motifs(
        graphs,
        condition_labels,
        fractions,
        primary_fraction,
        int(mining["minimum_nodes"]),
        int(mining["maximum_nodes"]),
        int(mining["threads"]),
        str(mining["backend_version"]),
        include_redundancy=False,
        candidate_minimum_support=candidate_minimum_support,
    )
    candidate_motifs = motifs
    candidate_motif_count = len(candidate_motifs)
    motifs, paired_evaluable = annotate_opportunity_support(
        candidate_motifs,
        baseline_graph_masks,
        opportunity_masks,
        graph_ids,
        registry,
        edge_index,
        fractions,
        primary_fraction,
        minimum_paired_states,
    )
    pair_table = paired_graph_indices(graph_ids, registry)
    permutation = np.arange(len(graph_ids))
    lower_indices = pair_table["lower_index"].to_numpy(dtype=int)
    higher_indices = pair_table["higher_index"].to_numpy(dtype=int)
    permutation[lower_indices] = higher_indices
    permutation[higher_indices] = lower_indices
    swapped_motifs, _ = annotate_opportunity_support(
        candidate_motifs,
        baseline_graph_masks[permutation],
        opportunity_masks[permutation],
        graph_ids,
        registry,
        edge_index,
        fractions,
        primary_fraction,
        minimum_paired_states,
    )
    invariant_columns = [
        "motif_id",
        "paired_state_count",
        "support_count",
        *[
            f"support_{int(round(fraction * 100)):02d}pct"
            for fraction in fractions
        ],
    ]
    if not motifs[invariant_columns].equals(
        swapped_motifs[invariant_columns]
    ):
        raise RuntimeError("Pooled motif family is not SES-swap invariant")
    motifs = annotate_redundancy(motifs)
    gspan_metadata["candidate_backend_support_counts"] = (
        gspan_metadata.pop("support_counts")
    )
    gspan_metadata["candidate_motif_count"] = candidate_motif_count
    gspan_metadata["motif_count"] = len(motifs)
    gspan_metadata["minimum_paired_states"] = minimum_paired_states
    gspan_metadata["support_semantics"] = (
        "motif_specific_pair_complete_state_opportunity"
    )
    baseline_occurrence = occurrence_matrix(
        baseline_graph_masks,
        motif_edge_masks(motifs, edge_index),
    ) & paired_evaluable
    if not np.array_equal(
        baseline_occurrence.sum(axis=0),
        motifs["support_count"].to_numpy(),
    ):
        raise RuntimeError("Independent motif support validation failed")
    for motif_index, encoded_graph_ids in enumerate(motifs["graph_ids"]):
        expected_graph_ids = [
            graph_ids[index]
            for index in np.flatnonzero(
                baseline_occurrence[:, motif_index]
            )
        ]
        if json.loads(encoded_graph_ids) != expected_graph_ids:
            raise RuntimeError(
                "Independent motif occurrence validation failed"
            )
    shape_summary = motif_shape_summary(motifs)
    write_csv(motifs, output / "motif_catalog.csv")
    write_csv(shape_summary, output / "motif_shape_summary.csv")
    logger.info("Mined %s pooled motifs", len(motifs))

    harmonized_csv = _resolve_harmonized_csv(
        config,
        phase26_manifest,
        logger,
    )
    eligible_registry = registry[
        registry["graph_id"].isin(graph_ids)
    ].copy()
    jobs = _bootstrap_jobs(pair_estimates, eligible_registry)
    r_versions = r_survey_versions(config.project_root)
    bootstrap = run_phi_bootstrap(
        project_root=config.project_root,
        harmonized_csv=harmonized_csv,
        jobs=jobs,
        output_directory=output_root / "r",
        replicates=int(bootstrap_settings["replicates"]),
        threshold=float(bootstrap_settings["threshold"]),
        seed=int(bootstrap_settings["seed"]),
        workers=max(1, workers),
        runtime_versions=r_versions,
    )
    bootstrap_cache_key = bootstrap.attrs.get("model_cache_key")
    bootstrap = _attach_pair_eligibility(bootstrap, pair_estimates)
    if not (
        bootstrap["bootstrap_status"].eq("ok").all()
        and bootstrap["bootstrap_replicates_valid"]
        .eq(bootstrap["bootstrap_replicates_requested"])
        .all()
    ):
        raise RuntimeError("Phase 3 requires complete bootstrap edge masks")
    _write_gzip_csv(
        bootstrap,
        output / "bootstrap_edge_selection_masks.csv.gz",
    )
    threshold_replicate_masks = bootstrap_graph_masks(
        bootstrap,
        graph_ids,
        edge_index,
        int(bootstrap_settings["replicates"]),
    )
    threshold_replicate_masks = np.bitwise_and(
        threshold_replicate_masks,
        opportunity_masks[np.newaxis, :],
    )
    stable_replicate_masks = restrict_to_frozen_edges(
        threshold_replicate_masks,
        baseline_graph_masks,
    )
    stability, stability_long, bootstrap_support = bootstrap_motif_support(
        stable_replicate_masks,
        motifs,
        edge_index,
        fractions,
        paired_evaluable,
    )
    stability_summary = _summarize_stability(
        stability,
        fractions,
        [
            float(value)
            for value in bootstrap_settings[
                "reporting_probability_cutpoints"
            ]
        ],
    )
    write_csv(stability, output / "motif_bootstrap_stability.csv")
    write_csv(
        stability_summary,
        output / "motif_bootstrap_stability_summary.csv",
    )
    _write_gzip_csv(
        stability_long,
        output / "motif_bootstrap_support_replicates.csv.gz",
    )
    vocabulary_stability = frozen_vocabulary_stability(
        motifs,
        bootstrap_support,
        fractions,
    )
    write_csv(
        vocabulary_stability,
        output / "frozen_vocabulary_stability.csv",
    )
    logger.info(
        "Evaluated motif support across %s graph realizations",
        int(bootstrap_settings["replicates"]),
    )

    if bool(bootstrap_settings["discovery_reruns"]):
        discovery_stability = bootstrap_discovery_set_stability(
            stable_replicate_masks,
            graph_ids,
            condition_labels,
            universe,
            edge_index,
            opportunity_masks,
            registry,
            motifs,
            fractions,
            primary_fraction,
            minimum_paired_states,
            candidate_minimum_support,
            int(mining["minimum_nodes"]),
            int(mining["maximum_nodes"]),
            int(mining["threads"]),
            str(mining["backend_version"]),
        )
    else:
        discovery_stability = vocabulary_stability.rename(
            columns={
                "baseline_vocabulary_retention": "discovery_set_retention",
                "baseline_vocabulary_jaccard": "discovery_set_jaccard",
                "replicate_baseline_vocabulary_count": (
                    "replicate_discovered_motif_count"
                ),
            }
        )
        discovery_stability["novel_motif_count"] = np.nan
    discovery_summary = _summarize_discovery_stability(
        discovery_stability
    )
    write_csv(
        discovery_stability,
        output / "bootstrap_discovery_set_stability.csv",
    )
    write_csv(
        discovery_summary,
        output / "bootstrap_discovery_set_summary.csv",
    )
    logger.info("Completed frozen-edge bootstrap gSpan reruns")

    if bool(bootstrap_settings["raw_threshold_sensitivity_reruns"]):
        threshold_sensitivity = bootstrap_discovery_set_stability(
            threshold_replicate_masks,
            graph_ids,
            condition_labels,
            universe,
            edge_index,
            opportunity_masks,
            registry,
            motifs,
            fractions,
            primary_fraction,
            minimum_paired_states,
            candidate_minimum_support,
            int(mining["minimum_nodes"]),
            int(mining["maximum_nodes"]),
            int(mining["threads"]),
            str(mining["backend_version"]),
        )
        threshold_sensitivity_summary = _summarize_discovery_stability(
            threshold_sensitivity
        )
        write_csv(
            threshold_sensitivity,
            output / "raw_threshold_discovery_sensitivity.csv",
        )
        write_csv(
            threshold_sensitivity_summary,
            output / "raw_threshold_discovery_sensitivity_summary.csv",
        )
    else:
        threshold_sensitivity_summary = pd.DataFrame()
    logger.info("Completed raw-threshold discovery sensitivity")

    density_null, degree_diagnostics = density_null_support(
        baseline_graph_masks,
        opportunity_masks,
        paired_evaluable,
        motifs,
        edge_index,
        int(density_settings["fixed_edge_replicates"]),
        int(density_settings["degree_preserving_replicates"]),
        int(density_settings["swaps_per_edge"]),
        int(density_settings["seed"]),
    )
    write_csv(density_null, output / "motif_density_null.csv")
    write_csv(
        degree_diagnostics,
        output / "degree_null_diagnostics.csv",
    )
    logger.info("Completed fixed-density and degree-preserving nulls")

    primary_motifs = motifs[motifs["primary_support"]].reset_index(
        drop=True
    )
    ses_results = paired_ses_permutation(
        graph_ids,
        baseline_graph_masks,
        opportunity_masks,
        primary_motifs,
        registry,
        edge_index,
        int(ses_settings["permutations"]),
        int(ses_settings["seed"]),
        minimum_paired_states,
    )
    ses_results = ses_results.merge(
        primary_motifs[
            [
                "motif_id",
                "node_count",
                "edge_count",
                "support_count",
            ]
        ],
        on="motif_id",
        how="left",
        validate="one_to_one",
    )
    write_csv(ses_results, output / "exploratory_ses_permutation.csv")
    logger.info("Completed paired density-adjusted SES permutations")

    fixed_null = density_null[
        density_null["null_model"].eq("fixed_eligible_dyad_edge_count")
    ].drop(columns="null_model")
    degree_null = density_null[
        density_null["null_model"].eq(
            "degree_preserving_eligible_dyad_edge_swap"
        )
    ].drop(columns="null_model")
    enriched = motifs.merge(
        stability,
        on=["motif_id", "node_count", "edge_count"],
        how="left",
        validate="one_to_one",
    )
    enriched = enriched.merge(
        fixed_null.add_prefix("fixed_").rename(
            columns={"fixed_motif_id": "motif_id"}
        ),
        on="motif_id",
        how="left",
        validate="one_to_one",
    )
    enriched = enriched.merge(
        degree_null.add_prefix("degree_").rename(
            columns={"degree_motif_id": "motif_id"}
        ),
        on="motif_id",
        how="left",
        validate="one_to_one",
    )
    enriched = enriched.merge(
        ses_results.add_prefix("ses_").rename(
            columns={"ses_motif_id": "motif_id"}
        ),
        on="motif_id",
        how="left",
        validate="one_to_one",
    )
    write_csv(enriched, output / "motif_catalog_enriched.csv")

    dpi = int(config.graph["visualization"]["dpi"])
    draw_motif_support_spectrum(
        shape_summary,
        output / "figures" / "motif_support_spectrum.png",
        dpi,
    )
    draw_motif_bootstrap_stability(
        stability,
        output / "figures" / "motif_bootstrap_stability.png",
        int(round(primary_fraction * 100)),
        dpi,
    )
    draw_discovery_set_stability(
        discovery_stability,
        output / "figures" / "discovery_set_stability.png",
        dpi,
        "Frozen-edge gSpan discovery-set stability",
        "discovery_set_retention",
        "Baseline motif-set retention",
    )
    if bool(bootstrap_settings["raw_threshold_sensitivity_reruns"]):
        draw_discovery_set_stability(
            threshold_sensitivity,
            output / "figures" / "raw_threshold_sensitivity.png",
            dpi,
            "Raw-threshold gSpan discovery sensitivity",
            "discovery_set_jaccard",
            "Baseline/replicate motif-set Jaccard",
        )
    draw_density_null(
        density_null,
        output / "figures" / "motif_density_null.png",
        dpi,
    )
    draw_ses_permutation(
        ses_results,
        output / "figures" / "exploratory_ses_permutation.png",
        dpi,
    )

    report = _feasibility_report(
        graph_index,
        motifs,
        shape_summary,
        stability,
        stability_summary,
        discovery_summary,
        threshold_sensitivity_summary,
        density_null,
        degree_diagnostics,
        ses_results,
        fractions,
        primary_fraction,
        minimum_paired_states,
    )
    write_json(_json_safe(report), output / "phase3_report.json")
    source = raw_xpt_path(config)
    write_json(
        _json_safe(
            manifest(
                config,
                "phase3",
                {
                    "phase26_baseline_config_digest": phase26_manifest[
                        "config_digest"
                    ],
                    "phase26_baseline_manifest_sha256": _sha256(
                        archive / "manifest.json"
                    ),
                    "source_data_sha256": _sha256(source),
                    "source_tree_sha256": _source_tree_sha256(
                        config.project_root
                    ),
                    "git_dirty": _git_dirty(config.project_root),
                    "frozen_graph_definition": (
                        "point phi >= 0.12 and "
                        "P_boot(phi >= 0.12) >= 0.90"
                    ),
                    "phase26_retention_decision": "HOLD",
                    "structural_feasibility_decision": "GO",
                    "population_definition": (
                        "47 eligible jurisdictions × paired education SES"
                    ),
                    "multiple_testing_rule": (
                        "Exploratory paired maxT FWER and BH FDR"
                    ),
                    "gspan": gspan_metadata,
                    "gspan_support": {
                        f"{fraction:.2f}": {
                            "denominator": (
                                "motif-specific paired-complete state graphs"
                            ),
                            "minimum_count": int(
                                motifs[
                                    "support_count_"
                                    f"{int(round(fraction * 100)):02d}pct"
                                ].min()
                            ),
                            "maximum_count": int(
                                motifs[
                                    "support_count_"
                                    f"{int(round(fraction * 100)):02d}pct"
                                ].max()
                            ),
                        }
                        for fraction in fractions
                    },
                    "bootstrap_replicates": int(
                        bootstrap_settings["replicates"]
                    ),
                    "bootstrap_bank_role": str(
                        bootstrap_settings["bank_role"]
                    ),
                    "bootstrap_selection_bank_seed": int(
                        bootstrap_settings["selection_bank_seed"]
                    ),
                    "bootstrap_evaluation_bank_seed": int(
                        bootstrap_settings["seed"]
                    ),
                    "bootstrap_probability_cutpoints_role": str(
                        bootstrap_settings["reporting_cutpoints_role"]
                    ),
                    "bootstrap_graph_estimand": (
                        "Conditional retention of frozen stable edges when "
                        "their independent evaluation-bank survey bootstrap "
                        "phi remains at least 0.12; not full-pipeline "
                        "stable-graph rediscovery"
                    ),
                    "raw_threshold_sensitivity_estimand": (
                        "All eligible edges crossing phi 0.12 in each "
                        "aligned survey bootstrap replicate"
                    ),
                    "bootstrap_alignment": (
                        "Lower/higher domains share state replicate weights; "
                        "equal replicate indices combine independent states"
                    ),
                    "motif_opportunity_definition": (
                        "Required dyads eligible in both SES graphs; support "
                        "uses motif-specific paired-complete states"
                    ),
                    "minimum_paired_states": minimum_paired_states,
                    "ineligible_graph_dyad_cells": int(
                        graph_index["ineligible_dyad_count"].sum()
                    ),
                    "phi_bootstrap_cache_key": bootstrap_cache_key,
                    "r_runtime": r_versions,
                    "density_null_primary": (
                        "fixed_eligible_dyad_edge_count"
                    ),
                    "density_null_sensitivity": (
                        "degree_preserving_eligible_dyad_edge_swap"
                    ),
                    "ses_permutation_scheme": (
                        "within_state_lower_higher_label_swap"
                    ),
                    "ses_statistics_role": "exploratory_only",
                    "ses_estimand": (
                        "equal-state lower-minus-higher structural contrast "
                        "among motif-evaluable state pairs"
                    ),
                    "causal_interpretation": False,
                    "replication_status": "2023_not_run",
                    "phase_boundary": (
                        "Stopped after exploratory Phase 3 motif analysis"
                    ),
                },
            )
        ),
        output / "manifest.json",
    )
    logger.info("Phase 3 complete: %s", output)
    return output
