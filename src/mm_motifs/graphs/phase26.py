from __future__ import annotations

import math
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd


def point_phi_stability_diagnostic(
    bootstrap: pd.DataFrame,
    point_threshold: float,
    stability_threshold: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = bootstrap[
        bootstrap["pair_eligible"].fillna(False)
        & bootstrap["point_phi"].ge(point_threshold)
    ].copy()
    raw["stable"] = (
        raw["p_phi_ge_012"].ge(stability_threshold)
        & raw["bootstrap_status"].eq("ok")
        & raw["bootstrap_replicates_valid"].eq(
            raw["bootstrap_replicates_requested"]
        )
    )
    bounds = [point_threshold, 0.13, 0.14, 0.15, 0.16, 0.18, 0.20, 0.25, 0.30, math.inf]
    labels = [
        ".12–.13",
        ".13–.14",
        ".14–.15",
        ".15–.16",
        ".16–.18",
        ".18–.20",
        ".20–.25",
        ".25–.30",
        "≥.30",
    ]
    raw["point_phi_band"] = pd.cut(
        raw["point_phi"],
        bins=bounds,
        labels=labels,
        right=False,
    )
    summary = (
        raw.groupby("point_phi_band", observed=True)
        .agg(
            raw_edges=("stable", "size"),
            stable_edges=("stable", "sum"),
            median_point_phi=("point_phi", "median"),
            median_selection_stability=("p_phi_ge_012", "median"),
            minimum_selection_stability=("p_phi_ge_012", "min"),
        )
        .reset_index()
    )
    summary["unstable_edges"] = (
        summary["raw_edges"] - summary["stable_edges"]
    )
    summary["stable_fraction"] = (
        summary["stable_edges"] / summary["raw_edges"]
    )
    summary = summary[
        [
            "point_phi_band",
            "raw_edges",
            "stable_edges",
            "unstable_edges",
            "stable_fraction",
            "median_point_phi",
            "median_selection_stability",
            "minimum_selection_stability",
        ]
    ]
    unstable = raw[~raw["stable"]]
    cutoff_diagnostics = {}
    for cutoff in (0.14, 0.15, 0.16, 0.18, 0.20):
        above = raw[raw["point_phi"].ge(cutoff)]
        key = f"phi_ge_{int(round(cutoff * 100)):03d}"
        cutoff_diagnostics[key] = {
            "raw_edge_count": int(len(above)),
            "stable_edge_count": int(above["stable"].sum()),
            "stable_fraction_against_phi_012": float(
                above["stable"].mean()
            ),
            "unstable_edges_below_cutoff": int(
                unstable["point_phi"].lt(cutoff).sum()
            ),
            "fraction_of_all_unstable_edges_below_cutoff": float(
                unstable["point_phi"].lt(cutoff).mean()
            ),
        }
    diagnostic = {
        "raw_phi_012_edge_count": int(len(raw)),
        "stable_phi_012_edge_count": int(raw["stable"].sum()),
        "unstable_phi_012_edge_count": int((~raw["stable"]).sum()),
        "raw_edge_stable_fraction": float(raw["stable"].mean()),
        "point_phi_selection_stability_spearman": float(
            raw["point_phi"].corr(raw["p_phi_ge_012"], method="spearman")
        ),
        "cutoff_diagnostics": cutoff_diagnostics,
        "interpretation_boundary": (
            "All stability fractions remain relative to P_boot(phi >= 0.12). "
            "A new 0.15 rule requires P_boot(phi >= 0.15) and cannot be "
            "validated from this diagnostic."
        ),
    }
    return summary, diagnostic


def build_phi_edge_scenarios(
    pair_counts: pd.DataFrame,
    bootstrap_results: pd.DataFrame,
    thresholds: list[dict[str, Any]],
    primary_rule_id: str,
    stable_rule_id: str,
    stability_threshold: float,
) -> pd.DataFrame:
    primary_effect = next(
        float(rule["minimum_effect"])
        for rule in thresholds
        if str(rule["rule_id"]) == primary_rule_id
    )
    bootstrap = bootstrap_results.rename(
        columns={
            "disease_a": "source_condition",
            "disease_b": "target_condition",
        }
    )
    keys = ["graph_id", "source_condition", "target_condition"]
    enriched = pair_counts.merge(
        bootstrap[
            [
                *keys,
                "point_phi",
                "p_phi_gt_zero",
                "p_phi_ge_012",
                "bootstrap_phi_median",
                "bootstrap_phi_025",
                "bootstrap_phi_975",
                "selection_stability_standard_error",
                "selection_stability_wilson_low",
                "selection_stability_wilson_high",
                "bootstrap_replicates_requested",
                "bootstrap_replicates_valid",
                "bootstrap_status",
                "bootstrap_message",
            ]
        ],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    mismatch = (
        enriched["point_phi"].notna()
        & enriched["association"].notna()
        & ~np.isclose(
            enriched["point_phi"],
            enriched["association"],
            rtol=1e-10,
            atol=1e-12,
        )
    )
    if mismatch.any():
        raise ValueError("Bootstrap point phi does not match the Python estimate")
    scenario_tables = []
    for rule in thresholds:
        scenario = enriched.copy()
        scenario["rule_id"] = str(rule["rule_id"])
        scenario["minimum_effect"] = float(rule["minimum_effect"])
        scenario["association_scale"] = "weighted_phi"
        scenario["edge_present"] = (
            scenario["pair_eligible"]
            & scenario["association"].ge(float(rule["minimum_effect"]))
        )
        scenario_tables.append(scenario)
    stable = enriched.copy()
    stable["rule_id"] = stable_rule_id
    stable["minimum_effect"] = primary_effect
    stable["association_scale"] = "weighted_phi"
    stable["estimator"] = "survey_bootstrap_stable_phi"
    stable["inference_scope"] = "design_bootstrap"
    stable["edge_present"] = (
        stable["pair_eligible"]
        & stable["association"].ge(primary_effect)
        & stable["bootstrap_status"].eq("ok")
        & stable["bootstrap_replicates_valid"].eq(
            stable["bootstrap_replicates_requested"]
        )
        & stable["p_phi_ge_012"].ge(stability_threshold)
    )
    scenario_tables.append(stable)
    return pd.concat(scenario_tables, ignore_index=True, sort=False)


def threshold_topology(
    graph_collections: dict[tuple[str, str], dict[str, nx.Graph]],
    ses_definition: str,
    threshold_rules: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered_rules = sorted(
        (
            (str(rule["rule_id"]), float(rule["minimum_effect"]))
            for rule in threshold_rules
        ),
        key=lambda value: value[1],
    )
    graphs_by_rule = {
        rule_id: graph_collections[(ses_definition, rule_id)]
        for rule_id, _ in ordered_rules
    }
    graph_ids = set(next(iter(graphs_by_rule.values())))
    if any(set(graphs) != graph_ids for graphs in graphs_by_rule.values()):
        raise ValueError("Threshold graph collections do not share graph IDs")

    statistic_rows = []
    transition_rows = []
    comparisons = [
        (*ordered_rules[0], *ordered_rules[1], "adjacent"),
        (*ordered_rules[1], *ordered_rules[2], "adjacent"),
        (*ordered_rules[0], *ordered_rules[2], "outer"),
    ]
    for graph_id in sorted(graph_ids):
        edge_sets = {}
        for rule_id, threshold in ordered_rules:
            graph = graphs_by_rule[rule_id][graph_id]
            edge_sets[rule_id] = {tuple(sorted(edge)) for edge in graph.edges()}
            statistic_rows.append(
                {
                    "graph_id": graph_id,
                    "geography_code": graph.graph["state_code"],
                    "ses_category": graph.graph["ses_category"],
                    "rule_id": rule_id,
                    "threshold": threshold,
                    "edge_count": graph.number_of_edges(),
                    "density": nx.density(graph),
                    "component_count": nx.number_connected_components(graph),
                    "isolate_count": len(list(nx.isolates(graph))),
                    "transitivity": nx.transitivity(graph),
                }
            )
        for (
            from_rule,
            from_threshold,
            to_rule,
            to_threshold,
            transition_type,
        ) in comparisons:
            from_graph = graphs_by_rule[from_rule][graph_id]
            to_graph = graphs_by_rule[to_rule][graph_id]
            from_edges = edge_sets[from_rule]
            to_edges = edge_sets[to_rule]
            union = from_edges | to_edges
            intersection = from_edges & to_edges
            nodes = sorted(from_graph.nodes() | to_graph.nodes())
            from_degrees = pd.Series(
                [from_graph.degree(node) for node in nodes],
                dtype=float,
            )
            to_degrees = pd.Series(
                [to_graph.degree(node) for node in nodes],
                dtype=float,
            )
            transition_rows.append(
                {
                    "graph_id": graph_id,
                    "geography_code": from_graph.graph["state_code"],
                    "ses_category": from_graph.graph["ses_category"],
                    "transition_type": transition_type,
                    "from_rule_id": from_rule,
                    "to_rule_id": to_rule,
                    "from_threshold": from_threshold,
                    "to_threshold": to_threshold,
                    "from_edge_count": len(from_edges),
                    "to_edge_count": len(to_edges),
                    "edges_removed": len(from_edges - to_edges),
                    "edge_retention_fraction": (
                        len(intersection) / len(from_edges)
                        if from_edges
                        else 1.0
                    ),
                    "edge_jaccard": (
                        len(intersection) / len(union) if union else 1.0
                    ),
                    "nested": to_edges.issubset(from_edges),
                    "degree_spearman": from_degrees.corr(
                        to_degrees,
                        method="spearman",
                    ),
                    "component_count_change": (
                        nx.number_connected_components(to_graph)
                        - nx.number_connected_components(from_graph)
                    ),
                    "isolate_count_change": (
                        len(list(nx.isolates(to_graph)))
                        - len(list(nx.isolates(from_graph)))
                    ),
                }
            )
    transitions = pd.DataFrame(transition_rows)
    summaries = []
    for keys, frame in transitions.groupby(
        [
            "transition_type",
            "from_rule_id",
            "to_rule_id",
            "from_threshold",
            "to_threshold",
        ],
        sort=True,
    ):
        summaries.append(
            {
                "transition_type": keys[0],
                "from_rule_id": keys[1],
                "to_rule_id": keys[2],
                "from_threshold": keys[3],
                "to_threshold": keys[4],
                "graph_count": len(frame),
                "median_edge_jaccard": float(frame["edge_jaccard"].median()),
                "minimum_edge_jaccard": float(frame["edge_jaccard"].min()),
                "q10_edge_jaccard": float(frame["edge_jaccard"].quantile(0.10)),
                "fraction_edge_jaccard_below_050": float(
                    frame["edge_jaccard"].lt(0.50).mean()
                ),
                "median_edges_removed": float(frame["edges_removed"].median()),
                "total_edges_removed": int(frame["edges_removed"].sum()),
                "median_degree_spearman": float(
                    frame["degree_spearman"].median()
                ),
                "fraction_component_count_changed": float(
                    frame["component_count_change"].ne(0).mean()
                ),
                "all_graphs_nested": bool(frame["nested"].all()),
            }
        )
    return (
        pd.DataFrame(statistic_rows),
        transitions,
        pd.DataFrame(summaries),
    )


def adjusted_concordance(
    edge_scenarios: pd.DataFrame,
    adjusted_edges: pd.DataFrame,
    rule_ids: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    adjusted = adjusted_edges.copy()
    adjusted["adjusted_positive_forward"] = (
        adjusted["model_status"].eq("ok")
        & adjusted["converged"].fillna(False)
        & adjusted["log_odds_ratio"].gt(0)
    )
    adjusted["adjusted_positive_reverse"] = (
        adjusted["reverse_model_status"].eq("ok")
        & adjusted["reverse_converged"].fillna(False)
        & adjusted["reverse_log_odds_ratio"].gt(0)
    )
    adjusted["adjusted_positive_both_directions"] = (
        adjusted["adjusted_positive_forward"]
        & adjusted["adjusted_positive_reverse"]
    )
    adjusted = adjusted.rename(
        columns={"edge_present": "adjusted_fdr_supported"}
    )
    keys = ["graph_id", "source_condition", "target_condition"]
    validation_columns = [
        *keys,
        "model_status",
        "log_odds_ratio",
        "odds_ratio",
        "confidence_low",
        "confidence_high",
        "q_value",
        "reverse_model_status",
        "reverse_log_odds_ratio",
        "reverse_odds_ratio",
        "direction_concordant",
        "adjusted_positive_forward",
        "adjusted_positive_reverse",
        "adjusted_positive_both_directions",
        "adjusted_fdr_supported",
    ]
    selected = edge_scenarios[
        edge_scenarios["rule_id"].isin(rule_ids)
        & edge_scenarios["edge_present"]
    ][
        [
            *keys,
            "rule_id",
            "association",
            "p_phi_ge_012",
        ]
    ]
    edge_level = selected.merge(
        adjusted[validation_columns],
        on=keys,
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    rows = []
    for rule_id, frame in edge_level.groupby("rule_id", sort=True):
        matched = frame["_merge"].eq("both")
        denominator = len(frame)
        rows.append(
            {
                "rule_id": rule_id,
                "selected_edge_count": denominator,
                "adjusted_model_match_count": int(matched.sum()),
                "adjusted_positive_forward_count": int(
                    frame["adjusted_positive_forward"].fillna(False).sum()
                ),
                "adjusted_positive_forward_fraction": float(
                    frame["adjusted_positive_forward"].fillna(False).mean()
                ),
                "adjusted_positive_both_directions_count": int(
                    frame["adjusted_positive_both_directions"]
                    .fillna(False)
                    .sum()
                ),
                "adjusted_positive_both_directions_fraction": float(
                    frame["adjusted_positive_both_directions"]
                    .fillna(False)
                    .mean()
                ),
                "adjusted_fdr_supported_count": int(
                    frame["adjusted_fdr_supported"].fillna(False).sum()
                ),
                "adjusted_fdr_supported_fraction": float(
                    frame["adjusted_fdr_supported"].fillna(False).mean()
                ),
            }
        )
    return edge_level.drop(columns="_merge"), pd.DataFrame(rows)


def phase26_viability_report(
    graph_statistics: pd.DataFrame,
    similarities: pd.DataFrame,
    edge_support: pd.DataFrame,
    bootstrap_results: pd.DataFrame,
    continuity_summary: pd.DataFrame,
    concordance_summary: pd.DataFrame,
    stable_rule_id: str,
    primary_rule_id: str,
    settings: dict[str, Any],
) -> dict[str, Any]:
    stable_stats = graph_statistics[
        graph_statistics["rule_id"] == stable_rule_id
    ]
    stable_similarity = similarities[
        (similarities["rule_id"] == stable_rule_id)
        & (similarities["graph_id_a"] < similarities["graph_id_b"])
    ]
    stable_support = edge_support[
        edge_support["rule_id"] == stable_rule_id
    ]
    primary_concordance = concordance_summary[
        concordance_summary["rule_id"] == primary_rule_id
    ].iloc[0]
    raw_selected = bootstrap_results["point_phi"].ge(0.12)
    if "pair_eligible" in bootstrap_results:
        raw_selected &= bootstrap_results["pair_eligible"].fillna(False)
    stable_selected = raw_selected & bootstrap_results["p_phi_ge_012"].ge(
        float(settings["selection_stability_threshold"])
    )
    stable_selected &= (
        bootstrap_results["bootstrap_status"].eq("ok")
        & bootstrap_results["bootstrap_replicates_valid"].eq(
            bootstrap_results["bootstrap_replicates_requested"]
        )
    )
    adjacent = continuity_summary[
        continuity_summary["transition_type"] == "adjacent"
    ]
    outer = continuity_summary[
        continuity_summary["transition_type"] == "outer"
    ]
    median_raw_jaccard = float(stable_similarity["edge_jaccard"].median())
    median_adjusted_z = float(
        stable_similarity["density_adjusted_jaccard_z"].median()
    )
    variable_recurrent_edges = int(
        stable_support["support_fraction"].between(0.10, 0.90).sum()
    )
    raw_edge_stable_fraction = float(
        stable_selected.sum() / raw_selected.sum()
    )
    criteria = {
        "bootstrap_complete": bool(
            bootstrap_results["bootstrap_status"].eq("ok").all()
            and bootstrap_results["bootstrap_replicates_valid"]
            .eq(bootstrap_results["bootstrap_replicates_requested"])
            .all()
        ),
        "enough_graphs": len(stable_stats)
        >= int(settings["minimum_eligible_graphs"]),
        "moderate_density": bool(
            float(stable_stats["edge_count"].median())
            <= float(settings["preferred_edge_count_maximum"])
            and float(stable_stats["density"].max())
            < float(settings["maximum_density_caution"])
        ),
        "meaningful_edge_count": float(stable_stats["edge_count"].median())
        >= float(settings["preferred_edge_count_minimum"]),
        "most_raw_edges_stable": raw_edge_stable_fraction
        >= float(settings["minimum_raw_edge_stable_fraction"]),
        "threshold_continuity": bool(
            adjacent["median_edge_jaccard"]
            .ge(float(settings["minimum_median_adjacent_jaccard"]))
            .all()
            and adjacent["fraction_edge_jaccard_below_050"]
            .le(
                float(
                    settings[
                        "maximum_fraction_adjacent_jaccard_below_050"
                    ]
                )
            )
            .all()
            and outer["median_edge_jaccard"]
            .ge(float(settings["minimum_median_outer_jaccard"]))
            .all()
            and continuity_summary["all_graphs_nested"].all()
        ),
        "adjustment_concordance": float(
            primary_concordance[
                "adjusted_positive_both_directions_fraction"
            ]
        )
        >= float(settings["minimum_positive_concordance"]),
        "heterogeneous_but_comparable": bool(
            median_raw_jaccard
            < float(settings["maximum_median_raw_jaccard"])
            and median_adjusted_z
            > float(settings["minimum_median_density_adjusted_z"])
        ),
        "recurrent_edge_backbone": variable_recurrent_edges
        >= int(settings["minimum_variable_recurrent_edges"]),
    }
    return {
        "decision": "GO" if all(criteria.values()) else "HOLD",
        "candidate_rule": stable_rule_id,
        "criteria": criteria,
        "diagnostics": {
            "graph_count": len(stable_stats),
            "median_edge_count": float(stable_stats["edge_count"].median()),
            "minimum_edge_count": int(stable_stats["edge_count"].min()),
            "maximum_edge_count": int(stable_stats["edge_count"].max()),
            "median_density": float(stable_stats["density"].median()),
            "maximum_density": float(stable_stats["density"].max()),
            "raw_phi_012_edge_count": int(raw_selected.sum()),
            "stable_phi_012_edge_count": int(stable_selected.sum()),
            "raw_edge_stable_fraction": raw_edge_stable_fraction,
            "minimum_selection_stability_among_raw_edges": float(
                bootstrap_results.loc[raw_selected, "p_phi_ge_012"].min()
            ),
            "median_raw_jaccard": median_raw_jaccard,
            "median_density_adjusted_jaccard_z": median_adjusted_z,
            "variable_recurrent_edge_count": variable_recurrent_edges,
            "adjusted_positive_both_directions_fraction": float(
                primary_concordance[
                    "adjusted_positive_both_directions_fraction"
                ]
            ),
            "adjusted_fdr_supported_fraction": float(
                primary_concordance["adjusted_fdr_supported_fraction"]
            ),
        },
        "decision_basis": (
            "Sparsity, selection stability, local threshold continuity, "
            "adjustment concordance, heterogeneity, and recurrence only."
        ),
        "selection_threshold_fixed": True,
        "stability_threshold_fixed": True,
        "ses_significance_used_for_selection": False,
        "phase_boundary": "Frequent subgraph mining is not implemented.",
    }
