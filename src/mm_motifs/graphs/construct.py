from __future__ import annotations

from typing import Any

import networkx as nx
import pandas as pd

from mm_motifs.statistics.survey import weighted_prevalence


def node_prevalence_table(
    frame: pd.DataFrame,
    condition_names: list[str],
    graph_id: str,
) -> pd.DataFrame:
    rows = []
    for condition in condition_names:
        rows.append(
            {
                "graph_id": graph_id,
                "condition": condition,
                **weighted_prevalence(frame[condition], frame["survey_weight"]),
            }
        )
    return pd.DataFrame(rows)


def construct_graph(
    graph_id: str,
    year: int,
    state_code: int,
    geography: str,
    ses_category: str,
    population: pd.DataFrame,
    node_table: pd.DataFrame,
    edge_table: pd.DataFrame,
    graph_config: dict[str, Any],
    population_graph_id: str | None = None,
    rule_id: str | None = None,
    estimator_name: str | None = None,
) -> nx.Graph:
    weights = pd.to_numeric(population["survey_weight"], errors="coerce")
    valid_weights = weights[weights.notna() & weights.gt(0)]
    weight_sum = float(valid_weights.sum())
    weight_squared_sum = float((valid_weights**2).sum())
    inference_scope = (
        str(edge_table["inference_scope"].iloc[0])
        if "inference_scope" in edge_table and not edge_table.empty
        else graph_config["estimator"]["inference_scope"]
    )
    association_scale = (
        str(edge_table["association_scale"].iloc[0])
        if "association_scale" in edge_table and not edge_table.empty
        else estimator_name or graph_config["estimator"]["name"]
    )
    graph = nx.Graph(
        graph_id=graph_id,
        year=int(year),
        state_code=int(state_code),
        geography=geography,
        ses_category=ses_category,
        n_unweighted=int(len(population)),
        weighted_population_estimate=weight_sum,
        kish_effective_n=(
            weight_sum**2 / weight_squared_sum
            if weight_squared_sum > 0
            else float("nan")
        ),
        estimator=estimator_name or graph_config["estimator"]["name"],
        inference_scope=inference_scope,
        association_scale=association_scale,
    )
    if population_graph_id is not None:
        graph.graph["population_graph_id"] = population_graph_id
    if rule_id is not None:
        graph.graph["rule_id"] = rule_id
    criteria = graph_config["node_criteria"]
    for row in node_table.itertuples(index=False):
        eligible = (
            row.n_valid >= int(criteria["minimum_valid_n"])
            and row.n_cases >= int(criteria["minimum_cases"])
        )
        if eligible:
            graph.add_node(
                row.condition,
                label=row.condition.replace("_", " ").title(),
                prevalence=float(row.weighted_prevalence),
                n_valid=int(row.n_valid),
                n_cases=int(row.n_cases),
            )

    for row in edge_table.loc[edge_table["edge_present"]].itertuples(index=False):
        if row.source_condition in graph and row.target_condition in graph:
            attributes = {
                "association": float(row.association),
                "estimator": row.estimator,
                "n_complete": int(row.n_complete),
                "cooccurring_cases": int(row.cooccurring_cases),
            }
            for name in (
                "odds_ratio",
                "reverse_odds_ratio",
                "confidence_low",
                "confidence_high",
                "q_value",
                "positive_stability",
                "stability_wilson_low",
                "stability_wilson_high",
                "p_phi_gt_zero",
                "p_phi_ge_012",
                "bootstrap_phi_median",
                "bootstrap_phi_025",
                "bootstrap_phi_975",
                "selection_stability_wilson_low",
                "selection_stability_wilson_high",
            ):
                value = getattr(row, name, None)
                if value is not None and pd.notna(value):
                    attributes[name] = float(value)
            graph.add_edge(
                row.source_condition,
                row.target_condition,
                **attributes,
            )
    return graph
