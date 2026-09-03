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
) -> nx.Graph:
    graph = nx.Graph(
        graph_id=graph_id,
        year=int(year),
        state_code=int(state_code),
        geography=geography,
        ses_category=ses_category,
        n_unweighted=int(len(population)),
        weighted_population_estimate=float(population["survey_weight"].sum()),
        estimator=graph_config["estimator"]["name"],
        inference_scope=graph_config["estimator"]["inference_scope"],
    )
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
            graph.add_edge(
                row.source_condition,
                row.target_condition,
                association=float(row.association),
                estimator=row.estimator,
                n_complete=int(row.n_complete),
                cooccurring_cases=int(row.cooccurring_cases),
            )
    return graph
