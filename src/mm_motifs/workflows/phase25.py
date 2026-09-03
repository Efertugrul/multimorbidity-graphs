from __future__ import annotations

import hashlib
import itertools
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import networkx as nx
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
from mm_motifs.graphs.calibration import (
    classify_similarity_pairs,
    graph_outlier_scores,
    ses_label_permutation_test,
    similarity_summary,
)
from mm_motifs.graphs.construct import construct_graph, node_prevalence_table
from mm_motifs.graphs.qc import (
    fixed_edge_null_jaccard,
    graph_statistics,
    pairwise_similarity,
)
from mm_motifs.graphs.serialize import write_graph
from mm_motifs.runtime import (
    configure_logging,
    interim_directory,
    manifest,
    output_directory,
    raw_xpt_path,
    write_csv,
    write_json,
)
from mm_motifs.statistics.disease_association import estimate_phi_scenarios
from mm_motifs.statistics.multiple_testing import benjamini_hochberg
from mm_motifs.statistics.r_survey import (
    adjusted_edge_table,
    r_survey_versions,
    run_survey_models,
)
from mm_motifs.visualization.calibration import (
    draw_rule_density_distributions,
    draw_similarity_distributions,
)
from mm_motifs.visualization.networks import draw_similarity_heatmap


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_tree_sha256(project_root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted((project_root / "src" / "mm_motifs").rglob("*.py"))
    paths.extend(sorted((project_root / "scripts").glob("*.R")))
    for path in paths:
        digest.update(str(path.relative_to(project_root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_dirty(project_root: Path) -> bool | None:
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    return bool(completed.stdout.strip()) if completed.returncode == 0 else None


def _write_r_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, na_rep="")
    os.replace(temporary, path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _population_frame(
    harmonized: pd.DataFrame,
    row: Any,
) -> pd.DataFrame:
    ses_column = f"ses_{row.ses_definition}"
    return harmonized[
        (harmonized["state_code"] == row.geography_code)
        & (harmonized[ses_column] == row.ses_category)
    ]


def _build_model_jobs(
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
        ["graph_id", "source_condition", "target_condition"]
    ].merge(metadata, on="graph_id", how="left")[
        [
            "graph_id",
            "state_code",
            "ses_definition",
            "ses_category",
            "source_condition",
            "target_condition",
        ]
    ]


def _bootstrap_stable_table(
    adjusted: pd.DataFrame,
    bootstrap_results: pd.DataFrame,
    primary_ses: str,
    stability_threshold: float,
    rule_id: str,
) -> pd.DataFrame:
    keys = ["graph_id", "source_condition", "target_condition"]
    bootstrap_columns = [
        *keys,
        "bootstrap_status",
        "bootstrap_message",
        "bootstrap_log_odds_ratio",
        "bootstrap_replicates_requested",
        "bootstrap_replicates_valid",
        "positive_stability",
        "stability_standard_error",
        "stability_wilson_low",
        "stability_wilson_high",
        "bootstrap_low_log",
        "bootstrap_high_log",
    ]
    stable = adjusted[adjusted["ses_definition"] == primary_ses].copy()
    stable = stable.merge(
        bootstrap_results[bootstrap_columns],
        on=keys,
        how="left",
    )
    stable["rule_id"] = rule_id
    stable["edge_present"] = (
        stable["edge_present"]
        & stable["bootstrap_status"].eq("ok")
        & stable["bootstrap_replicates_valid"].eq(
            stable["bootstrap_replicates_requested"]
        )
        & stable["positive_stability"].ge(stability_threshold)
    )
    return stable


def _add_stability_intervals(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    count = pd.to_numeric(
        result["bootstrap_replicates_valid"],
        errors="coerce",
    ).where(lambda values: values > 0)
    probability = pd.to_numeric(result["positive_stability"], errors="coerce")
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
    result["stability_standard_error"] = np.sqrt(
        probability * (1 - probability) / count
    )
    result["stability_wilson_low"] = center - radius
    result["stability_wilson_high"] = center + radius
    return result


def _construct_graphs(
    harmonized: pd.DataFrame,
    registry: pd.DataFrame,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    config: ProjectConfig,
    output: Path,
) -> tuple[
    dict[tuple[str, str], dict[str, nx.Graph]],
    dict[tuple[str, str], dict[str, pd.DataFrame]],
    pd.DataFrame,
    pd.DataFrame,
]:
    registry_index = registry.set_index("graph_id")
    node_groups = {key: value for key, value in nodes.groupby("graph_id")}
    graph_collections: dict[tuple[str, str], dict[str, nx.Graph]] = {}
    edge_collections: dict[tuple[str, str], dict[str, pd.DataFrame]] = {}
    statistics_rows = []
    index_rows = []
    for (population_graph_id, rule_id), edge_table in edges.groupby(
        ["graph_id", "rule_id"],
        sort=True,
    ):
        if population_graph_id not in registry_index.index:
            continue
        metadata = registry_index.loc[population_graph_id]
        population = harmonized[
            (harmonized["state_code"] == metadata["geography_code"])
            & (
                harmonized[f"ses_{metadata['ses_definition']}"]
                == metadata["ses_category"]
            )
        ]
        graph_id = f"{population_graph_id}__{rule_id}"
        estimator = str(edge_table["estimator"].iloc[0])
        graph = construct_graph(
            graph_id=graph_id,
            year=config.year,
            state_code=int(metadata["geography_code"]),
            geography=str(metadata["geography"]),
            ses_category=str(metadata["ses_category"]),
            population=population,
            node_table=node_groups[population_graph_id],
            edge_table=edge_table,
            graph_config=config.graph,
            population_graph_id=population_graph_id,
            rule_id=rule_id,
            estimator_name=estimator,
        )
        graph.graph["ses_definition"] = str(metadata["ses_definition"])
        graphml_path, json_path = write_graph(
            graph,
            output / "graphs" / str(metadata["ses_definition"]) / rule_id,
        )
        collection_key = (str(metadata["ses_definition"]), rule_id)
        graph_collections.setdefault(collection_key, {})[
            population_graph_id
        ] = graph
        edge_collections.setdefault(collection_key, {})[
            population_graph_id
        ] = edge_table
        stats = graph_statistics(graph)
        stats.update(
            {
                "population_graph_id": population_graph_id,
                "ses_definition": str(metadata["ses_definition"]),
                "rule_id": rule_id,
            }
        )
        statistics_rows.append(stats)
        index_rows.append(
            {
                "graph_id": graph_id,
                "population_graph_id": population_graph_id,
                "ses_definition": str(metadata["ses_definition"]),
                "rule_id": rule_id,
                "graphml_path": str(graphml_path.relative_to(config.project_root)),
                "json_path": str(json_path.relative_to(config.project_root)),
            }
        )
    return (
        graph_collections,
        edge_collections,
        pd.DataFrame(statistics_rows),
        pd.DataFrame(index_rows),
    )


def _calibration_summary(statistics: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    rows = []
    minimum = int(settings["preferred_edge_count_minimum"])
    maximum = int(settings["preferred_edge_count_maximum"])
    density_caution = float(settings["maximum_density_caution"])
    for (ses_definition, rule_id), frame in statistics.groupby(
        ["ses_definition", "rule_id"],
        sort=True,
    ):
        rows.append(
            {
                "ses_definition": ses_definition,
                "rule_id": rule_id,
                "graph_count": len(frame),
                "edge_count_minimum": int(frame["edge_count"].min()),
                "edge_count_median": float(frame["edge_count"].median()),
                "edge_count_maximum": int(frame["edge_count"].max()),
                "density_minimum": float(frame["density"].min()),
                "density_median": float(frame["density"].median()),
                "density_maximum": float(frame["density"].max()),
                "fraction_in_preferred_edge_range": float(
                    frame["edge_count"].between(minimum, maximum).mean()
                ),
                "high_density_graph_count": int(
                    frame["density"].ge(density_caution).sum()
                ),
                "median_edge_target_met": bool(
                    minimum <= frame["edge_count"].median() <= maximum
                ),
                "maximum_density_target_met": bool(
                    frame["density"].max() < density_caution
                ),
            }
        )
    return pd.DataFrame(rows)


def _edge_support(
    graph_collections: dict[tuple[str, str], dict[str, nx.Graph]],
    names: list[str],
) -> pd.DataFrame:
    rows = []
    possible = list(itertools.combinations(names, 2))
    for (ses_definition, rule_id), graphs in graph_collections.items():
        for source, target in possible:
            count = sum(graph.has_edge(source, target) for graph in graphs.values())
            rows.append(
                {
                    "ses_definition": ses_definition,
                    "rule_id": rule_id,
                    "source_condition": source,
                    "target_condition": target,
                    "graph_count": len(graphs),
                    "support_count": count,
                    "support_fraction": count / len(graphs) if graphs else math.nan,
                }
            )
    return pd.DataFrame(rows)


def _ses_definition_sensitivity(
    graph_collections: dict[tuple[str, str], dict[str, nx.Graph]],
    names: list[str],
    primary_ses: str,
    sensitivity_definitions: list[str],
) -> pd.DataFrame:
    possible_edges = math.comb(len(names), 2)
    rows = []
    primary_rules = {
        rule
        for definition, rule in graph_collections
        if definition == primary_ses
    }
    for sensitivity_ses in sensitivity_definitions:
        sensitivity_rules = {
            rule
            for definition, rule in graph_collections
            if definition == sensitivity_ses
        }
        for rule_id in sorted(primary_rules & sensitivity_rules):
            primary_graphs = graph_collections[(primary_ses, rule_id)]
            sensitivity_graphs = graph_collections[(sensitivity_ses, rule_id)]
            primary_lookup = {
                (graph.graph["state_code"], graph.graph["ses_category"]): graph
                for graph in primary_graphs.values()
            }
            sensitivity_lookup = {
                (graph.graph["state_code"], graph.graph["ses_category"]): graph
                for graph in sensitivity_graphs.values()
            }
            for key in sorted(primary_lookup.keys() & sensitivity_lookup.keys()):
                graph_a = primary_lookup[key]
                graph_b = sensitivity_lookup[key]
                edges_a = {tuple(sorted(edge)) for edge in graph_a.edges()}
                edges_b = {tuple(sorted(edge)) for edge in graph_b.edges()}
                union = edges_a | edges_b
                raw = len(edges_a & edges_b) / len(union) if union else 1.0
                density_null = fixed_edge_null_jaccard(
                    len(edges_a),
                    len(edges_b),
                    possible_edges,
                    raw,
                )
                rows.append(
                    {
                        "rule_id": rule_id,
                        "geography_code": key[0],
                        "ses_category": key[1],
                        "primary_ses_definition": primary_ses,
                        "sensitivity_ses_definition": sensitivity_ses,
                        "primary_edge_count": len(edges_a),
                        "sensitivity_edge_count": len(edges_b),
                        "edge_jaccard": raw,
                        "density_adjusted_jaccard_z": density_null["z_score"],
                        "density_null_expected_jaccard": density_null["expected"],
                    }
                )
    return pd.DataFrame(rows)


def _california_diagnostics(
    pair_counts: pd.DataFrame,
    all_edges: pd.DataFrame,
    bootstrap_results: pd.DataFrame,
    outlier_scores: pd.DataFrame,
    nodes: pd.DataFrame,
    registry: pd.DataFrame,
    year: int,
    primary_ses: str,
    phi_rules: list[dict[str, Any]],
    adjusted_rule_id: str,
    stable_rule_id: str,
    stability_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    target = f"{year}_06_{primary_ses}_lower"
    pairs = pair_counts[pair_counts["graph_id"] == target][
        [
            "graph_id",
            "source_condition",
            "target_condition",
            "association",
            "n_complete",
            "source_cases",
            "target_cases",
            "cooccurring_cases",
            "kish_effective_n",
        ]
    ].rename(columns={"association": "weighted_phi"})
    rule_ids = [
        *(str(rule["rule_id"]) for rule in phi_rules),
        adjusted_rule_id,
        stable_rule_id,
    ]
    for rule_id in rule_ids:
        rule = all_edges[
            (all_edges["graph_id"] == target) & (all_edges["rule_id"] == rule_id)
        ][["source_condition", "target_condition", "edge_present"]].rename(
            columns={"edge_present": f"edge_present_{rule_id}"}
        )
        pairs = pairs.merge(
            rule,
            on=["source_condition", "target_condition"],
            how="left",
        )
    adjusted_columns = [
        "source_condition",
        "target_condition",
        "odds_ratio",
        "reverse_odds_ratio",
        "confidence_low",
        "confidence_high",
        "q_value",
        "direction_concordant",
    ]
    adjusted = all_edges[
        (all_edges["graph_id"] == target)
        & (all_edges["rule_id"] == adjusted_rule_id)
    ]
    pairs = pairs.merge(
        adjusted[adjusted_columns],
        on=["source_condition", "target_condition"],
        how="left",
    )
    if not bootstrap_results.empty:
        pairs = pairs.merge(
            bootstrap_results[
                [
                    "source_condition",
                    "target_condition",
                    "positive_stability",
                    "stability_wilson_low",
                    "stability_wilson_high",
                    "bootstrap_replicates_valid",
                ]
            ],
            on=["source_condition", "target_condition"],
            how="left",
        )
    target_scores = outlier_scores[outlier_scores["graph_id"] == target].copy()
    rank_rows = []
    for row in target_scores.itertuples(index=False):
        rank_rows.append(
            {
                "rule_id": row.rule_id,
                "mean_same_ses_jaccard": row.mean_same_ses_jaccard,
                "robust_similarity_z": row.robust_similarity_z,
                "mean_same_ses_density_adjusted_z": (
                    row.mean_same_ses_density_adjusted_z
                ),
                "robust_density_adjusted_similarity_z": (
                    row.robust_density_adjusted_similarity_z
                ),
                "edge_count": row.edge_count,
                "density": row.density,
            }
        )
    target_nodes = nodes[nodes["graph_id"] == target]
    peer_nodes = nodes[
        nodes["graph_id"].str.endswith(f"_{primary_ses}_lower")
        & nodes["graph_id"].ne(target)
    ]
    prevalence_rows = []
    for row in target_nodes.itertuples(index=False):
        peers = peer_nodes[peer_nodes["condition"] == row.condition][
            "weighted_prevalence"
        ].dropna()
        peer_median = float(peers.median())
        median_absolute_deviation = float(
            (peers - peer_median).abs().median()
        )
        robust_scale = 1.4826 * median_absolute_deviation
        prevalence_rows.append(
            {
                "condition": row.condition,
                "california_weighted_prevalence": row.weighted_prevalence,
                "peer_median_weighted_prevalence": peer_median,
                "peer_robust_prevalence_z": (
                    (row.weighted_prevalence - peer_median) / robust_scale
                    if robust_scale > 0
                    else math.nan
                ),
                "california_n_valid": row.n_valid,
                "california_cases": row.n_cases,
                "california_node_kish_effective_n": row.kish_effective_n,
            }
        )
    prevalence_diagnostics = pd.DataFrame(prevalence_rows)
    target_population = registry[registry["graph_id"] == target].iloc[0]
    lower_registry = registry[
        registry["graph_id"].str.endswith(f"_{primary_ses}_lower")
        & registry["eligibility_status"].eq("eligible")
    ]
    target_bootstrap = bootstrap_results
    reference_phi_rule = min(
        phi_rules,
        key=lambda rule: float(rule["minimum_effect"]),
    )
    reference_phi_threshold = float(reference_phi_rule["minimum_effect"])
    summary = {
        "graph_id": target,
        "interpretation": (
            "Descriptive threshold, composition, adjusted-association, and stability "
            "diagnostics; no causal or directional interpretation."
        ),
        "coding_registry_shared_across_states": True,
        "reference_phi_threshold": reference_phi_threshold,
        "phi_edges_within_002_below_reference_threshold": int(
            pairs["weighted_phi"]
            .between(
                reference_phi_threshold - 0.02,
                reference_phi_threshold,
                inclusive="left",
            )
            .sum()
        ),
        "population_n_unweighted": int(target_population["n_unweighted"]),
        "population_kish_effective_n": float(
            target_population["kish_effective_n"]
        ),
        "population_design_effect": float(
            target_population["n_unweighted"]
            / target_population["kish_effective_n"]
        ),
        "population_effective_n_percentile": float(
            lower_registry["kish_effective_n"].rank(pct=True).loc[
                target_population.name
            ]
        ),
        "minimum_pair_complete_n": int(pairs["n_complete"].min()),
        "minimum_node_valid_fraction": float(
            (target_nodes["n_valid"] / target_population["n_unweighted"]).min()
        ),
        "adjusted_candidate_edge_count": int(
            pairs[f"edge_present_{adjusted_rule_id}"].fillna(False).sum()
        ),
        "stable_adjusted_edge_count": int(
            pairs[f"edge_present_{stable_rule_id}"].fillna(False).sum()
        ),
        "candidate_edges_below_stability_threshold": int(
            (
                target_bootstrap["positive_stability"]
                < stability_threshold
            ).sum()
        )
        if not target_bootstrap.empty
        else None,
        "largest_absolute_prevalence_deviations": (
            prevalence_diagnostics.assign(
                absolute_z=prevalence_diagnostics[
                    "peer_robust_prevalence_z"
                ].abs()
            )
            .sort_values("absolute_z", ascending=False)
            .head(3)[
                [
                    "condition",
                    "california_weighted_prevalence",
                    "peer_median_weighted_prevalence",
                    "peer_robust_prevalence_z",
                ]
            ]
            .to_dict(orient="records")
        ),
        "node_kish_effective_n_minimum": float(
            target_nodes["kish_effective_n"].min()
        ),
        "node_kish_effective_n_median": float(
            target_nodes["kish_effective_n"].median()
        ),
        "rule_ranks": rank_rows,
    }
    return pairs, prevalence_diagnostics, summary


def _neutral_viability_report(
    calibration: pd.DataFrame,
    edge_support: pd.DataFrame,
    similarities: pd.DataFrame,
    registry: pd.DataFrame,
    config: ProjectConfig,
    bootstrap_complete: bool,
    candidate_rule: str,
    bootstrap_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    primary_ses = config.analysis["phase25"]["primary_ses_definition"]
    candidate = calibration[
        (calibration["ses_definition"] == primary_ses)
        & (calibration["rule_id"] == candidate_rule)
    ]
    settings = config.graph["phase25"]["calibration"]
    if candidate.empty:
        diagnostics = {}
        criteria = {
            "enough_graphs": False,
            "moderately_sparse": False,
            "heterogeneous_but_comparable": False,
            "recurrent_edge_backbone": False,
            "bootstrap_complete": bootstrap_complete,
        }
    else:
        row = candidate.iloc[0]
        candidate_similarity = similarities[
            (similarities["ses_definition"] == primary_ses)
            & (similarities["rule_id"] == candidate_rule)
            & (similarities["graph_id_a"] < similarities["graph_id_b"])
        ]
        median_similarity = float(candidate_similarity["edge_jaccard"].median())
        median_adjusted_similarity = float(
            candidate_similarity["density_adjusted_jaccard_z"].median()
        )
        support = edge_support[
            (edge_support["ses_definition"] == primary_ses)
            & (edge_support["rule_id"] == candidate_rule)
        ]
        variable_recurrent = int(
            support["support_fraction"].between(0.10, 0.90).sum()
        )
        criteria = {
            "enough_graphs": int(row["graph_count"])
            >= int(config.graph["viability"]["minimum_eligible_graphs"]),
            "moderately_sparse": (
                float(settings["preferred_edge_count_minimum"])
                <= float(row["edge_count_median"])
                <= float(settings["preferred_edge_count_maximum"])
                and float(row["density_maximum"])
                < float(settings["maximum_density_caution"])
            ),
            "heterogeneous_but_comparable": (
                median_similarity < 0.95 and median_adjusted_similarity > 0
            ),
            "recurrent_edge_backbone": variable_recurrent >= 5,
            "bootstrap_complete": bootstrap_complete,
        }
        diagnostics = {
            "median_raw_jaccard": median_similarity,
            "median_density_adjusted_jaccard_z": median_adjusted_similarity,
            "variable_recurrent_edge_count": variable_recurrent,
            "median_edge_count": float(row["edge_count_median"]),
            "maximum_density": float(row["density_maximum"]),
        }
    primary_registry = registry[
        registry["ses_definition"].eq(primary_ses)
        & registry["eligibility_status"].eq("eligible")
    ]
    diagnostics.update(
        {
            "minimum_population_kish_effective_n": float(
                primary_registry["kish_effective_n"].min()
            ),
            "median_population_kish_effective_n": float(
                primary_registry["kish_effective_n"].median()
            ),
            "populations_with_kish_effective_n_below_500": int(
                primary_registry["kish_effective_n"].lt(500).sum()
            ),
        }
    )
    go = all(criteria.values())
    rule_profiles = []
    for row in calibration[
        calibration["ses_definition"] == primary_ses
    ].itertuples(index=False):
        rule_similarity = similarities[
            (similarities["ses_definition"] == primary_ses)
            & (similarities["rule_id"] == row.rule_id)
            & (similarities["graph_id_a"] < similarities["graph_id_b"])
        ]
        rule_support = edge_support[
            (edge_support["ses_definition"] == primary_ses)
            & (edge_support["rule_id"] == row.rule_id)
        ]
        rule_profiles.append(
            {
                "rule_id": row.rule_id,
                "median_edge_count": row.edge_count_median,
                "maximum_density": row.density_maximum,
                "median_raw_jaccard": float(
                    rule_similarity["edge_jaccard"].median()
                ),
                "median_density_adjusted_jaccard_z": float(
                    rule_similarity["density_adjusted_jaccard_z"].median()
                ),
                "variable_recurrent_edge_count": int(
                    rule_support["support_fraction"].between(0.10, 0.90).sum()
                ),
                "meets_sparsity_targets": bool(
                    row.median_edge_target_met
                    and row.maximum_density_target_met
                ),
                "bootstrap_stability_assessed": row.rule_id == candidate_rule,
            }
        )
    eligibility = (
        registry.groupby(["ses_definition", "eligibility_status"])
        .size()
        .rename("population_count")
        .reset_index()
        .to_dict(orient="records")
    )
    return {
        "decision": "GO" if go else "HOLD",
        "decision_scope": (
            f"{candidate_rule} with {primary_ses}; alternative rules are "
            "reported but not promoted without their own stability calibration"
        ),
        "candidate_rule": candidate_rule,
        "primary_ses_definition": primary_ses,
        "criteria": criteria,
        "diagnostics": diagnostics,
        "bootstrap_diagnostics": bootstrap_diagnostics,
        "rule_profiles": rule_profiles,
        "decision_basis": (
            "Technical sparsity, stability, recurrence, and graph comparability only. "
            "SES permutation significance is excluded from the GO/HOLD decision."
        ),
        "eligibility": eligibility,
        "bootstrap_scope": (
            f"{config.graph['phase25']['bootstrap']['replicates']} replicates "
            "for calibration; confirm paper-relevant edges with "
            f"{config.graph['phase25']['bootstrap']['confirmation_replicates_minimum']}"
            "–"
            f"{config.graph['phase25']['bootstrap']['confirmation_replicates_maximum']}"
            " replicates."
        ),
        "phase_boundary": "Frequent subgraph mining is not implemented.",
    }


def run_phase25(
    config: ProjectConfig,
    verbose: bool = False,
    workers: int = 4,
    run_bootstrap: bool = True,
) -> Path:
    source = raw_xpt_path(config)
    if not source.exists():
        raise FileNotFoundError(f"BRFSS XPT file not found: {source}")
    phase25 = config.analysis["phase25"]
    graph_phase25 = config.graph["phase25"]
    source_sha256 = _sha256(source)
    source_tree_sha256 = _source_tree_sha256(config.project_root)
    harmonized_cache_key = hashlib.sha256(
        (
            f"{config.digest}\n{source_sha256}\n{source_tree_sha256}"
        ).encode()
    ).hexdigest()[:16]
    run_bootstrap = run_bootstrap and bool(
        graph_phase25["bootstrap"]["enabled"]
    )
    base_output = output_directory(config, "phase25")
    output = base_output / ("full" if run_bootstrap else "point_only")
    model_output = base_output / "r"
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    logger = configure_logging(output / "run.log", verbose)
    geography_codes = [int(value) for value in phase25["geography_codes"]]
    intermediate = interim_directory(config)
    parquet_path = (
        intermediate / f"phase25_harmonized_{harmonized_cache_key}.parquet"
    )
    r_csv_path = (
        intermediate / f"phase25_harmonized_{harmonized_cache_key}.csv"
    )

    if parquet_path.exists():
        harmonized = pd.read_parquet(parquet_path)
        logger.info("Loaded cached Phase 2.5 harmonized data")
    else:
        raw = read_selected(source, phase25_selected_columns(config))
        harmonized = harmonize_phase25(raw, config, geography_codes)
        del raw
        if phase25["persist_harmonized"]:
            _write_parquet(harmonized, parquet_path)
        logger.info("Harmonized %s state/DC respondents", len(harmonized))
    if not r_csv_path.exists():
        _write_r_csv(harmonized, r_csv_path)

    registries = [
        phase25_population_registry(
            harmonized,
            config.year,
            definition,
            int(phase25["minimum_population_n"]),
            geography_codes,
        )
        for definition in phase25["ses_definitions"]
    ]
    registry = pd.concat(registries, ignore_index=True)
    write_csv(registry, output / "population_registry.csv")
    eligible_registry = registry[registry["eligibility_status"] == "eligible"].copy()
    logger.info(
        "Eligible populations: %s",
        eligible_registry.groupby("ses_definition").size().to_dict(),
    )

    names = condition_names(config.conditions)
    pair_parts = []
    phi_parts = []
    node_parts = []
    for row in eligible_registry.itertuples(index=False):
        population = _population_frame(harmonized, row)
        pair_table, scenario_table = estimate_phi_scenarios(
            population,
            names,
            row.graph_id,
            config.graph["edge_criteria"],
            graph_phase25["phi_rules"],
        )
        pair_table["ses_definition"] = row.ses_definition
        pair_table["ses_category"] = row.ses_category
        scenario_table["ses_definition"] = row.ses_definition
        scenario_table["ses_category"] = row.ses_category
        nodes = node_prevalence_table(population, names, row.graph_id)
        nodes["ses_definition"] = row.ses_definition
        nodes["ses_category"] = row.ses_category
        pair_parts.append(pair_table)
        phi_parts.append(scenario_table)
        node_parts.append(nodes)
    pair_counts = pd.concat(pair_parts, ignore_index=True)
    phi_edges = pd.concat(phi_parts, ignore_index=True)
    nodes = pd.concat(node_parts, ignore_index=True)
    write_csv(pair_counts, output / "pair_estimates_weighted_phi.csv")
    write_csv(nodes, output / "disease_prevalence_by_graph.csv")

    r_versions = r_survey_versions(config.project_root)
    model_jobs = _build_model_jobs(pair_counts, eligible_registry)
    adjusted_settings = graph_phase25["adjusted_rule"]
    point_models = run_survey_models(
        project_root=config.project_root,
        mode="point",
        harmonized_csv=r_csv_path,
        jobs=model_jobs,
        output_directory=model_output,
        confidence_level=float(adjusted_settings["confidence_level"]),
        replicates=0,
        seed=config.random_seed,
        workers=workers,
        runtime_versions=r_versions,
    )
    point_model_cache_key = point_models.attrs.get("model_cache_key")
    adjusted = adjusted_edge_table(
        point_models,
        pair_counts,
        float(adjusted_settings["maximum_q_value"]),
        benjamini_hochberg,
        str(adjusted_settings["rule_id"]),
    )
    write_csv(adjusted, output / "adjusted_edge_table.csv")
    direction_summary = (
        adjusted.groupby("ses_definition", as_index=False)
        .agg(
            pair_count=("direction_concordant", "size"),
            direction_concordant_count=(
                "direction_concordant",
                lambda values: values.fillna(False).sum(),
            ),
            forward_model_failure_count=(
                "model_status",
                lambda values: (~values.eq("ok")).sum(),
            ),
            reverse_model_failure_count=(
                "reverse_model_status",
                lambda values: (~values.eq("ok")).sum(),
            ),
            reverse_nonconverged_count=(
                "reverse_converged",
                lambda values: (~values.fillna(False)).sum(),
            ),
        )
    )
    direction_summary["direction_discordant_count"] = (
        direction_summary["pair_count"]
        - direction_summary["direction_concordant_count"]
    )
    write_csv(
        direction_summary,
        output / "adjusted_direction_diagnostic.csv",
    )

    bootstrap_settings = graph_phase25["bootstrap"]
    primary_ses = phase25["primary_ses_definition"]
    bootstrap_candidates = adjusted[
        adjusted["edge_present"] & adjusted["ses_definition"].eq(primary_ses)
    ]
    bootstrap_jobs = bootstrap_candidates[
        [
            "graph_id",
            "state_code",
            "ses_definition",
            "ses_category",
            "source_condition",
            "target_condition",
        ]
    ]
    if run_bootstrap:
        bootstrap_results = run_survey_models(
            project_root=config.project_root,
            mode="bootstrap",
            harmonized_csv=r_csv_path,
            jobs=bootstrap_jobs,
            output_directory=model_output,
            confidence_level=float(adjusted_settings["confidence_level"]),
            replicates=int(bootstrap_settings["replicates"]),
            seed=int(bootstrap_settings["seed"]),
            workers=workers,
            runtime_versions=r_versions,
        )
        bootstrap_model_cache_key = bootstrap_results.attrs.get(
            "model_cache_key"
        )
        bootstrap_results = _add_stability_intervals(bootstrap_results)
        stable = _bootstrap_stable_table(
            adjusted,
            bootstrap_results,
            primary_ses,
            float(bootstrap_settings["positive_stability_threshold"]),
            str(bootstrap_settings["rule_id"]),
        )
        write_csv(
            bootstrap_results,
            output / "bootstrap_edge_stability.csv",
        )
        bootstrap_summary = (
            bootstrap_results.groupby("graph_id", as_index=False)
            .agg(
                candidate_edge_count=("positive_stability", "size"),
                valid_edge_count=("bootstrap_status", lambda values: values.eq("ok").sum()),
                minimum_positive_stability=("positive_stability", "min"),
                median_positive_stability=("positive_stability", "median"),
                mean_positive_stability=("positive_stability", "mean"),
            )
        )
        bootstrap_summary["stable_edge_count"] = bootstrap_summary[
            "graph_id"
        ].map(
            stable[stable["edge_present"]]
            .groupby("graph_id")
            .size()
            .to_dict()
        ).fillna(0).astype(int)
        write_csv(
            bootstrap_summary,
            output / "bootstrap_graph_summary.csv",
        )
        bootstrap_complete = (
            len(bootstrap_results) == len(bootstrap_jobs)
            and bootstrap_results["bootstrap_status"].eq("ok").all()
            and bootstrap_results["bootstrap_replicates_valid"]
            .eq(int(bootstrap_settings["replicates"]))
            .all()
        )
    else:
        bootstrap_results = pd.DataFrame()
        stable = pd.DataFrame()
        bootstrap_complete = False
        bootstrap_model_cache_key = None

    edge_parts = [phi_edges, adjusted]
    if not stable.empty:
        edge_parts.append(stable)
    all_edges = pd.concat(edge_parts, ignore_index=True, sort=False)
    write_csv(all_edges, output / "edge_table_all_rules.csv")

    (
        graph_collections,
        edge_collections,
        statistics,
        graph_index,
    ) = _construct_graphs(
        harmonized,
        eligible_registry,
        nodes,
        all_edges,
        config,
        output,
    )
    write_csv(graph_index, output / "graph_database.csv")
    write_csv(statistics, output / "graph_statistics.csv")

    similarity_parts = []
    classified_parts = []
    summary_parts = []
    permutation_parts = []
    for (ses_definition, rule_id), graphs in graph_collections.items():
        similarities = pairwise_similarity(
            graphs,
            edge_collections[(ses_definition, rule_id)],
            names,
        )
        similarities["ses_definition"] = ses_definition
        similarities["rule_id"] = rule_id
        similarity_parts.append(similarities)
        classified = classify_similarity_pairs(similarities)
        classified["ses_definition"] = ses_definition
        classified["rule_id"] = rule_id
        classified_parts.append(classified)
        summary = similarity_summary(similarities)
        summary["ses_definition"] = ses_definition
        summary["rule_id"] = rule_id
        summary_parts.append(summary)
        permutation = ses_label_permutation_test(
            graphs,
            int(graph_phase25["similarity"]["permutations"]),
            int(graph_phase25["similarity"]["seed"]),
        )
        permutation["ses_definition"] = ses_definition
        permutation["rule_id"] = rule_id
        permutation_parts.append(permutation)
        draw_similarity_heatmap(
            similarities,
            list(graphs),
            output
            / "figures"
            / f"{ses_definition}_{rule_id}_edge_jaccard_heatmap.png",
            int(config.graph["visualization"]["dpi"]),
            show_labels=False,
            title=f"{ses_definition} — {rule_id} edge-set similarity",
        )

    similarities = pd.concat(similarity_parts, ignore_index=True)
    classified_similarities = pd.concat(classified_parts, ignore_index=True)
    similarity_summaries = pd.concat(summary_parts, ignore_index=True)
    permutation_results = pd.concat(permutation_parts, ignore_index=True)
    write_csv(similarities, output / "structural_similarity.csv")
    write_csv(
        classified_similarities,
        output / "similarity_comparison_pairs.csv",
    )
    write_csv(
        similarity_summaries,
        output / "similarity_comparison_summary.csv",
    )
    write_csv(
        permutation_results,
        output / "within_state_ses_label_permutation.csv",
    )

    calibration = _calibration_summary(
        statistics,
        graph_phase25["calibration"],
    )
    support = _edge_support(graph_collections, names)
    sensitivity = _ses_definition_sensitivity(
        graph_collections,
        names,
        primary_ses,
        [
            definition
            for definition in phase25["ses_definitions"]
            if definition != primary_ses
        ],
    )
    write_csv(calibration, output / "graph_rule_comparison.csv")
    write_csv(support, output / "edge_support_by_rule.csv")
    write_csv(sensitivity, output / "ses_definition_sensitivity.csv")

    outlier_parts = []
    education_registry = eligible_registry[
        eligible_registry["ses_definition"] == primary_ses
    ]
    for (ses_definition, rule_id), graphs in graph_collections.items():
        if ses_definition != primary_ses:
            continue
        scores = graph_outlier_scores(graphs, education_registry)
        scores["ses_definition"] = ses_definition
        scores["rule_id"] = rule_id
        outlier_parts.append(scores)
    outlier_scores = pd.concat(outlier_parts, ignore_index=True)
    write_csv(outlier_scores, output / "graph_outlier_scores.csv")
    california_graph_id = f"{config.year}_06_{primary_ses}_lower"
    (
        california_pairs,
        california_prevalence,
        california_summary,
    ) = _california_diagnostics(
        pair_counts,
        all_edges,
        bootstrap_results[
            bootstrap_results["graph_id"] == california_graph_id
        ]
        if not bootstrap_results.empty
        else bootstrap_results,
        outlier_scores,
        nodes,
        registry,
        config.year,
        primary_ses,
        graph_phase25["phi_rules"],
        str(adjusted_settings["rule_id"]),
        str(bootstrap_settings["rule_id"]),
        float(bootstrap_settings["positive_stability_threshold"]),
    )
    write_csv(
        california_pairs,
        output / "california_lower_edge_diagnostics.csv",
    )
    write_csv(
        california_prevalence,
        output / "california_lower_prevalence_diagnostics.csv",
    )
    write_json(
        _json_safe(california_summary),
        output / "california_lower_summary.json",
    )

    draw_rule_density_distributions(
        statistics,
        output / "figures" / "graph_density_by_rule.png",
    )
    candidate_pairs = classified_similarities[
        (classified_similarities["ses_definition"] == primary_ses)
        & (
            classified_similarities["rule_id"]
            == str(bootstrap_settings["rule_id"])
        )
    ]
    if not candidate_pairs.empty:
        draw_similarity_distributions(
            candidate_pairs,
            output
            / "figures"
            / f"{bootstrap_settings['rule_id']}_similarity_comparison.png",
        )

    if not bootstrap_results.empty:
        stability_threshold = float(
            bootstrap_settings["positive_stability_threshold"]
        )
        retained = bootstrap_results["positive_stability"].ge(
            stability_threshold
        )
        bootstrap_diagnostics = {
            "candidate_edge_count": len(bootstrap_results),
            "retained_edge_count": int(retained.sum()),
            "retained_fraction": float(retained.mean()),
            "minimum_positive_stability": float(
                bootstrap_results["positive_stability"].min()
            ),
            "bootstrap_intervals_including_zero": int(
                (
                    bootstrap_results["bootstrap_low_log"].le(0)
                    & bootstrap_results["bootstrap_high_log"].ge(0)
                ).sum()
            ),
            "wilson_lower_bound_meets_threshold_count": int(
                bootstrap_results["stability_wilson_low"]
                .ge(stability_threshold)
                .sum()
            ),
            "selection_uses_point_estimate_not_interval_bound": True,
        }
    else:
        bootstrap_diagnostics = {
            "candidate_edge_count": 0,
            "retained_edge_count": 0,
            "retained_fraction": None,
            "selection_uses_point_estimate_not_interval_bound": True,
        }
    viability = _neutral_viability_report(
        calibration,
        support,
        similarities,
        registry,
        config,
        bootstrap_complete,
        str(bootstrap_settings["rule_id"]),
        bootstrap_diagnostics,
    )
    write_json(_json_safe(viability), output / "viability_report.json")
    write_json(
        _json_safe(
            manifest(
                config,
                "phase25",
                {
                    "baseline_run_id": phase25["baseline_run_id"],
                    "source_data_sha256": source_sha256,
                    "source_tree_sha256": source_tree_sha256,
                    "harmonized_cache_key": harmonized_cache_key,
                    "git_dirty": _git_dirty(config.project_root),
                    "population_definition": (
                        "available US state or DC × SES; education primary, "
                        "income sensitivity"
                    ),
                    "geography_scope": phase25["geography_scope"],
                    "territories_included": False,
                    "eligible_population_graphs": {
                        key: int(value)
                        for key, value in eligible_registry.groupby(
                            "ses_definition"
                        ).size().items()
                    },
                    "r_runtime": r_versions,
                    "point_model_cache_key": point_model_cache_key,
                    "bootstrap_model_cache_key": bootstrap_model_cache_key,
                    "survey_lonely_psu": "adjust",
                    "survey_adjust_domain_lonely": True,
                    "edge_rules": sorted(all_edges["rule_id"].unique()),
                    "edge_threshold": (
                        "phi >= 0.08; phi >= 0.12; or adjusted OR > 1 with "
                        "95% CI lower > 1 and within-graph BH q <= 0.05"
                    ),
                    "multiple_testing_rule": (
                        "BH FDR within each adjusted graph"
                    ),
                    "edge_semantics": (
                        "Undirected statistical association; fixed logistic "
                        "direction is an estimation convention, not a causal "
                        "direction"
                    ),
                    "bootstrap_replicates": (
                        int(bootstrap_settings["replicates"])
                        if run_bootstrap
                        else 0
                    ),
                    "bootstrap_complete": bool(bootstrap_complete),
                    "bootstrap_purpose": (
                        "calibration only; confirm paper-relevant results with "
                        f"{bootstrap_settings['confirmation_replicates_minimum']}"
                        "–"
                        f"{bootstrap_settings['confirmation_replicates_maximum']}"
                        " replicates"
                    ),
                    "similarity_density_adjustment": (
                        "Exact fixed-edge-count hypergeometric null"
                    ),
                    "permutation_scheme": (
                        "within_state_lower_higher_label_swap"
                    ),
                    "permutation_null": (
                        "Graph structure is unrelated to SES label conditional "
                        "on state"
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
    logger.info("Phase 2.5 complete: %s", output)
    return output
