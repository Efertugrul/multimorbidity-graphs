from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import yaml

from mm_motifs.config import ProjectConfig
from mm_motifs.data.harmonize import (
    harmonize_phase25,
    phase25_selected_columns,
)
from mm_motifs.data.load import read_selected, xport_dimensions
from mm_motifs.features.conditions import condition_names
from mm_motifs.features.populations import phase25_population_registry
from mm_motifs.graphs.construct import node_prevalence_table
from mm_motifs.graphs.phase26 import build_phi_edge_scenarios
from mm_motifs.motifs.mining import (
    edge_universe,
    graph_database_from_edges,
    graph_edge_masks,
    graph_opportunity_masks,
)
from mm_motifs.replication.confirmatory import (
    evaluate_confirmatory_replication,
    hypothesis_state_differences,
)
from mm_motifs.replication.methodological import (
    evaluate_methodological_vocabulary,
    summarize_occurrence_families,
)
from mm_motifs.replication.protocol import validate_replication_protocol
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
from mm_motifs.workflows.phase25 import (
    _git_dirty,
    _json_safe,
    _population_frame,
    _sha256,
    _source_tree_sha256,
    _write_parquet,
    _write_r_csv,
)
from mm_motifs.workflows.phase26 import (
    _attach_pair_eligibility,
    _bootstrap_jobs,
    _selection_intervals,
)


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def _validate_source_input(
    config: ProjectConfig,
    source: Path,
) -> tuple[str, dict[str, Any]]:
    metadata_path = source.with_suffix(".download.json")
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Frozen Phase 4 download metadata not found: {metadata_path}"
        )
    download_metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )
    expected_url = config.analysis["data"]["urls"]["2023"]
    source_sha256 = _sha256(source)
    if (
        int(download_metadata["year"]) != 2023
        or download_metadata["source_url"] != expected_url
        or download_metadata["xpt_sha256"] != source_sha256
        or int(download_metadata["xpt_bytes"]) != source.stat().st_size
        or len(str(download_metadata["archive_sha256"])) != 64
    ):
        raise ValueError("2023 XPT download provenance is invalid")
    row_count, columns = xport_dimensions(source)
    expected_rows = int(
        config.analysis["data"]["expected_records"]["2023"]
    )
    if row_count != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} BRFSS rows, found {row_count}"
        )
    missing = sorted(set(phase25_selected_columns(config)) - set(columns))
    if missing:
        raise ValueError(f"2023 XPT is missing required variables: {missing}")
    return source_sha256, {
        "metadata_path": str(metadata_path),
        "source_url": expected_url,
        "archive_sha256": download_metadata["archive_sha256"],
        "xpt_sha256": source_sha256,
        "xpt_bytes": source.stat().st_size,
        "xpt_row_count": row_count,
    }


def _validate_target_configuration(
    config: ProjectConfig,
    protocol_directory: Path,
) -> None:
    expected_analysis = _read_yaml(protocol_directory / "analysis_phase4.yaml")
    expected_graph = _read_yaml(protocol_directory / "graph_phase4.yaml")
    expected_conditions = _read_yaml(protocol_directory / "conditions.yaml")
    if config.analysis != expected_analysis:
        raise ValueError("Active analysis config differs from frozen Phase 4 config")
    if config.graph != expected_graph:
        raise ValueError("Active graph config differs from frozen Phase 4 config")
    if config.conditions != expected_conditions:
        raise ValueError("Active condition registry differs from frozen Phase 4 registry")
    if _sha256(config.project_root / "scripts/phi_bootstrap.R") != _sha256(
        protocol_directory / "phi_bootstrap.R"
    ):
        raise ValueError("Active bootstrap script differs from frozen Phase 4 script")


def _validate_runtime(
    protocol: dict[str, Any],
    r_versions: dict[str, str],
) -> None:
    expected = protocol["runtime"]
    observed = {
        "python": ".".join(str(value) for value in sys.version_info[:3]),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "networkx": nx.__version__,
    }
    if observed != expected:
        raise RuntimeError(
            f"Python runtime differs from protocol: {observed} != {expected}"
        )
    expected_r = protocol["edge_bootstrap"]["runtime"]
    observed_r = {
        "r": str(r_versions["r"]).split()[2],
        "survey": str(r_versions["survey"]),
        "data_table": str(r_versions["data_table"]),
    }
    normalized_expected_r = {
        name: str(value) for name, value in expected_r.items()
    }
    if observed_r != normalized_expected_r:
        raise RuntimeError(
            "R runtime differs from protocol: "
            f"{observed_r} != {normalized_expected_r}"
        )


def _complete_bootstrap_graphs(
    bootstrap: pd.DataFrame,
    replicates: int,
) -> pd.DataFrame:
    scoped = bootstrap[bootstrap["pair_eligible"].fillna(False)].copy()
    scoped["complete"] = (
        scoped["bootstrap_status"].eq("ok")
        & scoped["bootstrap_replicates_requested"].eq(replicates)
        & scoped["bootstrap_replicates_valid"].eq(replicates)
    )
    return (
        scoped.groupby("graph_id", sort=True)
        .agg(
            eligible_dyad_count=("pair_eligible", "size"),
            complete_bootstrap_dyad_count=("complete", "sum"),
            graph_bootstrap_complete=("complete", "all"),
        )
        .reset_index()
    )


def _phase4_bootstrap_jobs(
    pair_counts: pd.DataFrame,
    registry: pd.DataFrame,
) -> pd.DataFrame:
    eligible = pair_counts[pair_counts["pair_eligible"].fillna(False)].copy()
    return _bootstrap_jobs(eligible, registry)


def _paired_usable_registry(
    registry: pd.DataFrame,
    graph_quality: pd.DataFrame,
) -> pd.DataFrame:
    eligible = registry[registry["eligibility_status"].eq("eligible")].merge(
        graph_quality,
        on="graph_id",
        how="left",
        validate="one_to_one",
    )
    eligible["graph_bootstrap_complete"] = (
        eligible["graph_bootstrap_complete"].fillna(False)
    )
    usable = eligible[eligible["graph_bootstrap_complete"]].copy()
    paired_states = (
        usable.groupby("geography_code")["ses_category"]
        .agg(lambda values: set(values) == {"lower", "higher"})
    )
    paired_codes = set(paired_states[paired_states].index.astype(int))
    usable["phase4_graph_status"] = np.where(
        usable["geography_code"].isin(paired_codes),
        "usable_paired",
        "excluded_unpaired",
    )
    return usable[usable["phase4_graph_status"].eq("usable_paired")].copy()


def _graph_index(
    graph_ids: list[str],
    graph_masks: np.ndarray,
    opportunity_masks: np.ndarray,
    registry: pd.DataFrame,
) -> pd.DataFrame:
    metadata = registry.set_index("graph_id")
    rows = []
    for index, graph_id in enumerate(graph_ids):
        row = metadata.loc[graph_id]
        rows.append(
            {
                "graph_id": graph_id,
                "geography_code": int(row["geography_code"]),
                "geography": row["geography"],
                "ses_category": row["ses_category"],
                "selected_edge_count": int(
                    int(graph_masks[index]).bit_count()
                ),
                "eligible_dyad_count": int(
                    int(opportunity_masks[index]).bit_count()
                ),
            }
        )
    return pd.DataFrame(rows)


def run_phase4(
    config: ProjectConfig,
    verbose: bool = False,
    workers: int = 4,
) -> Path:
    settings = config.graph["phase4"]
    protocol_directory = config.resolve(settings["protocol_directory"])
    lock = validate_replication_protocol(
        protocol_directory,
        require_release_context=True,
    )
    _validate_target_configuration(config, protocol_directory)
    protocol = _read_yaml(protocol_directory / "protocol.yaml")
    source = raw_xpt_path(config)
    if not source.exists():
        raise FileNotFoundError(f"BRFSS XPT file not found: {source}")
    source_sha256, input_provenance = _validate_source_input(config, source)
    r_versions = r_survey_versions(config.project_root)
    _validate_runtime(protocol, r_versions)
    output = output_directory(config, "phase4") / "full"
    model_output = output_directory(config, "phase4") / "r"
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    logger = configure_logging(output / "run.log", verbose)

    source_tree_sha256 = _source_tree_sha256(config.project_root)
    harmonized_key = hashlib.sha256(
        f"{config.digest}\n{source_sha256}\n{source_tree_sha256}".encode()
    ).hexdigest()[:16]
    intermediate = interim_directory(config)
    parquet_path = intermediate / f"phase4_harmonized_{harmonized_key}.parquet"
    r_csv_path = intermediate / f"phase4_harmonized_{harmonized_key}.csv"
    geography_codes = [
        int(value) for value in config.analysis["phase25"]["geography_codes"]
    ]
    if parquet_path.exists():
        harmonized = pd.read_parquet(parquet_path)
        logger.info("Loaded cached frozen Phase 4 harmonized data")
    else:
        raw = read_selected(source, phase25_selected_columns(config))
        expected_rows = int(
            config.analysis["data"]["expected_records"]["2023"]
        )
        if len(raw) != expected_rows:
            raise ValueError(
                f"Expected {expected_rows} selected rows, found {len(raw)}"
            )
        harmonized = harmonize_phase25(raw, config, geography_codes)
        del raw
        _write_parquet(harmonized, parquet_path)
        logger.info("Harmonized %s 2023 respondents", len(harmonized))
    if not r_csv_path.exists():
        _write_r_csv(harmonized, r_csv_path)

    registry = phase25_population_registry(
        harmonized,
        config.year,
        str(settings["primary_ses_definition"]),
        int(config.analysis["phase25"]["minimum_population_n"]),
        geography_codes,
    )
    eligible_registry = registry[
        registry["eligibility_status"].eq("eligible")
    ].copy()
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
        nodes = node_prevalence_table(
            population,
            names,
            row.graph_id,
        )
        nodes["ses_definition"] = row.ses_definition
        nodes["ses_category"] = row.ses_category
        pair_parts.append(pairs)
        node_parts.append(nodes)
    if not pair_parts:
        raise RuntimeError("No 2023 education populations meet eligibility")
    pair_counts = pd.concat(pair_parts, ignore_index=True)
    nodes = pd.concat(node_parts, ignore_index=True)

    bootstrap_settings = settings["bootstrap"]
    jobs = _phase4_bootstrap_jobs(pair_counts, eligible_registry)
    bootstrap = run_phi_bootstrap(
        project_root=config.project_root,
        harmonized_csv=r_csv_path,
        jobs=jobs,
        output_directory=model_output,
        replicates=int(bootstrap_settings["replicates"]),
        threshold=float(settings["bootstrap_selection_threshold"]),
        seed=int(bootstrap_settings["seed"]),
        workers=workers,
        runtime_versions=r_versions,
    )
    bootstrap_cache_key = bootstrap.attrs.get("model_cache_key")
    bootstrap = _selection_intervals(
        _attach_pair_eligibility(bootstrap, pair_counts)
    )
    graph_quality = _complete_bootstrap_graphs(
        bootstrap,
        int(bootstrap_settings["replicates"]),
    )
    analysis_registry = _paired_usable_registry(registry, graph_quality)
    if analysis_registry.empty:
        raise RuntimeError("No paired usable 2023 jurisdictions")

    edge_scenarios = build_phi_edge_scenarios(
        pair_counts,
        bootstrap,
        [
            {
                "rule_id": "phi_012",
                "minimum_effect": float(settings["point_phi_threshold"]),
            }
        ],
        "phi_012",
        str(settings["stable_rule_id"]),
        float(settings["minimum_bootstrap_selection_probability"]),
    )
    usable_graph_ids = set(analysis_registry["graph_id"])
    edge_scenarios = edge_scenarios[
        edge_scenarios["graph_id"].isin(usable_graph_ids)
    ].copy()
    stable_edges = edge_scenarios[
        edge_scenarios["rule_id"].eq(settings["stable_rule_id"])
    ].copy()
    graphs, condition_labels = graph_database_from_edges(
        stable_edges,
        str(settings["stable_rule_id"]),
    )
    graph_ids = [str(graph.graph["graph_id"]) for graph in graphs]
    universe = edge_universe(condition_labels["condition"].tolist())
    edge_index = {edge: index for index, edge in enumerate(universe)}
    graph_masks = graph_edge_masks(graphs, edge_index)
    opportunity_masks = graph_opportunity_masks(
        stable_edges,
        str(settings["stable_rule_id"]),
        graph_ids,
        edge_index,
    )
    if np.any(np.bitwise_and(graph_masks, opportunity_masks) != graph_masks):
        raise RuntimeError("Phase 4 graph contains an ineligible dyad")

    vocabulary = pd.read_csv(
        protocol_directory
        / protocol["methodological_replication"]["registry"],
        low_memory=False,
    )
    methodological, methodological_summary = (
        evaluate_methodological_vocabulary(
            graph_ids,
            graph_masks,
            opportunity_masks,
            vocabulary,
            analysis_registry,
            edge_index,
            float(
                protocol["methodological_replication"][
                    "primary_2023_reproduction"
                ]["pooled_support_minimum"]
            ),
            int(
                protocol["methodological_replication"][
                    "primary_2023_reproduction"
                ]["minimum_state_pairs"]
            ),
        )
    )
    families = pd.read_csv(
        protocol_directory
        / protocol["methodological_replication"]["family_registry"],
        low_memory=False,
    )
    family_results = summarize_occurrence_families(
        methodological,
        families,
    )
    hypotheses = pd.read_csv(
        protocol_directory / protocol["confirmatory_replication"]["registry"]
    )
    discovery_states = set(
        pd.read_csv(
            protocol_directory / protocol["population"]["discovery_registry"]
        )["geography_code"].astype(int)
    )
    confirmatory = evaluate_confirmatory_replication(
        graph_ids,
        graph_masks,
        opportunity_masks,
        hypotheses,
        analysis_registry,
        edge_index,
        discovery_states,
        int(protocol["confirmatory_replication"]["permutations"]),
        int(protocol["confirmatory_replication"]["seed"]),
        int(protocol["population"]["minimum_motif_evaluable_state_pairs"]),
        float(protocol["confirmatory_replication"]["multiplicity"]["alpha"]),
    )
    state_differences = hypothesis_state_differences(
        graph_ids,
        graph_masks,
        opportunity_masks,
        hypotheses.sort_values("family_order").reset_index(drop=True),
        analysis_registry,
        edge_index,
    )
    state_differences.index.name = "geography_code"

    write_csv(registry, output / "population_registry_all_targets.csv")
    write_csv(graph_quality, output / "bootstrap_graph_completeness.csv")
    write_csv(
        analysis_registry,
        output / "population_registry_primary_analysis.csv",
    )
    write_csv(pair_counts, output / "pair_estimates_weighted_phi.csv")
    write_csv(nodes, output / "disease_prevalence_by_graph.csv")
    write_csv(bootstrap, output / "phi_bootstrap_edge_stability.csv")
    write_csv(stable_edges, output / "stable_edge_table.csv")
    write_csv(
        _graph_index(
            graph_ids,
            graph_masks,
            opportunity_masks,
            analysis_registry,
        ),
        output / "graph_database_index.csv",
    )
    write_csv(
        methodological,
        output / "methodological_vocabulary_replication.csv",
    )
    write_json(
        _json_safe(methodological_summary),
        output / "methodological_replication_summary.json",
    )
    write_csv(
        family_results,
        output / "methodological_family_replication.csv",
    )
    write_csv(confirmatory, output / "confirmatory_hypothesis_results.csv")
    write_csv(
        state_differences.reset_index(),
        output / "confirmatory_state_differences.csv",
    )
    run_manifest = manifest(
        config,
        "phase4",
        {
            "protocol_id": lock["protocol_id"],
            "protocol_sha256": lock["protocol_sha256"],
            "execution_release_tag": lock["execution_release_tag"],
            "protocol_deviation_id": lock["protocol_deviation_id"],
            "source_xpt": str(source),
            "source_sha256": source_sha256,
            "input_provenance": input_provenance,
            "source_tree_sha256": source_tree_sha256,
            "source_tree_dirty": _git_dirty(config.project_root),
            "target_year": 2023,
            "motif_discovery_performed": False,
            "primary_decision_set": "year_specific_eligible_from_target_registry",
            "sensitivity_can_rescue_primary_failure": False,
            "bootstrap_model_cache_key": bootstrap_cache_key,
            "r_runtime": r_versions,
            "paired_jurisdiction_count": int(
                analysis_registry["geography_code"].nunique()
            ),
            "methodological_motif_count": int(len(methodological)),
            "confirmatory_hypothesis_count": int(
                hypotheses["hypothesis_id"].nunique()
            ),
        },
    )
    write_json(_json_safe(run_manifest), output / "manifest.json")
    logger.info("Completed frozen Phase 4 replication at %s", output)
    return output
