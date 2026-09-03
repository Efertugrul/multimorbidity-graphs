from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from mm_motifs.graphs.qc import pairwise_similarity, viability_report


def _graph(graph_id: str, state: int, ses: str, edges: list[tuple[str, str]]) -> nx.Graph:
    graph = nx.Graph(
        graph_id=graph_id,
        year=2024,
        state_code=state,
        geography=str(state),
        ses_category=ses,
        n_unweighted=1000,
        weighted_population_estimate=10000.0,
    )
    graph.add_nodes_from(["A", "B", "C"])
    graph.add_edges_from(edges)
    return graph


def _edges(graph_id: str, values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "graph_id": graph_id,
            "source_condition": ["A", "A", "B"],
            "target_condition": ["B", "C", "C"],
            "association": values,
        }
    )


def test_pairwise_similarity_uses_labeled_edge_sets() -> None:
    graphs = {
        "low": _graph("low", 1, "lower", [("A", "B"), ("A", "C")]),
        "high": _graph("high", 1, "higher", [("A", "B"), ("B", "C")]),
    }
    tables = {
        "low": _edges("low", [0.2, 0.3, 0.1]),
        "high": _edges("high", [0.2, 0.1, 0.3]),
    }
    result = pairwise_similarity(graphs, tables, ["A", "B", "C"])
    comparison = result[
        (result["graph_id_a"] == "low") & (result["graph_id_b"] == "high")
    ].iloc[0]
    assert comparison["edge_jaccard"] == 1 / 3
    assert np.isclose(comparison["edge_agreement"], 1 / 3)
    assert np.isfinite(comparison["density_adjusted_jaccard_z"])
    assert comparison["density_null_expected_jaccard"] > 0
    diagonal = result[
        (result["graph_id_a"] == "low") & (result["graph_id_b"] == "low")
    ].iloc[0]
    assert np.isnan(diagonal["density_adjusted_jaccard_z"])


def test_viability_report_defers_motif_criteria() -> None:
    graphs = {
        "low": _graph("low", 1, "lower", [("A", "B")]),
        "high": _graph("high", 1, "higher", [("A", "B"), ("B", "C")]),
    }
    statistics = pd.DataFrame(
        {
            "edge_count": [1, 2],
            "density": [1 / 3, 2 / 3],
        }
    )
    similarities = pd.DataFrame(
        {
            "graph_id_a": ["high"],
            "graph_id_b": ["low"],
            "same_state": [True],
            "same_ses": [False],
            "edge_jaccard": [0.5],
        }
    )
    settings = {
        "minimum_eligible_graphs": 2,
        "minimum_unique_edge_fraction": 0.5,
        "preferred_median_density_minimum": 0.05,
        "preferred_median_density_maximum": 0.75,
    }
    report = viability_report(graphs, statistics, similarities, settings)
    assert report["criteria"]["A_structural_heterogeneity"]["passed"]
    assert report["criteria"]["C_recurrent_nontrivial_motifs"]["passed"] is None
