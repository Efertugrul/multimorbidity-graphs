from __future__ import annotations

import itertools
import math
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd


def graph_statistics(graph: nx.Graph) -> dict[str, Any]:
    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    degrees = [degree for _, degree in graph.degree()]
    return {
        "graph_id": graph.graph["graph_id"],
        "year": graph.graph["year"],
        "geography_code": graph.graph["state_code"],
        "geography": graph.graph["geography"],
        "ses_category": graph.graph["ses_category"],
        "n_unweighted": graph.graph["n_unweighted"],
        "weighted_population_estimate": graph.graph["weighted_population_estimate"],
        "node_count": node_count,
        "edge_count": edge_count,
        "density": nx.density(graph) if node_count > 1 else 0.0,
        "component_count": nx.number_connected_components(graph) if node_count else 0,
        "isolate_count": len(list(nx.isolates(graph))),
        "mean_degree": float(np.mean(degrees)) if degrees else 0.0,
        "maximum_degree": max(degrees, default=0),
        "transitivity": nx.transitivity(graph) if node_count > 2 else 0.0,
    }


def _association_map(table: pd.DataFrame) -> dict[tuple[str, str], float]:
    values = {}
    for row in table.itertuples(index=False):
        key = tuple(sorted((row.source_condition, row.target_condition)))
        values[key] = float(row.association)
    return values


def pairwise_similarity(
    graphs: dict[str, nx.Graph],
    edge_tables: dict[str, pd.DataFrame],
    condition_names: list[str],
) -> pd.DataFrame:
    possible_edges = math.comb(len(condition_names), 2)
    rows = []
    association_maps = {
        graph_id: _association_map(table) for graph_id, table in edge_tables.items()
    }
    for graph_id_a, graph_id_b in itertools.product(graphs, repeat=2):
        graph_a = graphs[graph_id_a]
        graph_b = graphs[graph_id_b]
        edges_a = {tuple(sorted(edge)) for edge in graph_a.edges()}
        edges_b = {tuple(sorted(edge)) for edge in graph_b.edges()}
        union = edges_a | edges_b
        intersection = edges_a & edges_b
        jaccard = len(intersection) / len(union) if union else 1.0
        agreement = (
            1 - len(edges_a ^ edges_b) / possible_edges if possible_edges else 1.0
        )
        values_a = association_maps[graph_id_a]
        values_b = association_maps[graph_id_b]
        common = [
            key
            for key in values_a.keys() & values_b.keys()
            if np.isfinite(values_a[key]) and np.isfinite(values_b[key])
        ]
        if (
            len(common) >= 2
            and np.std([values_a[key] for key in common]) > 0
            and np.std([values_b[key] for key in common]) > 0
        ):
            correlation = float(
                np.corrcoef(
                    [values_a[key] for key in common],
                    [values_b[key] for key in common],
                )[0, 1]
            )
        else:
            correlation = math.nan
        rows.append(
            {
                "graph_id_a": graph_id_a,
                "graph_id_b": graph_id_b,
                "state_a": graph_a.graph["state_code"],
                "state_b": graph_b.graph["state_code"],
                "ses_a": graph_a.graph["ses_category"],
                "ses_b": graph_b.graph["ses_category"],
                "same_state": graph_a.graph["state_code"] == graph_b.graph["state_code"],
                "same_ses": graph_a.graph["ses_category"] == graph_b.graph["ses_category"],
                "edge_jaccard": jaccard,
                "edge_agreement": agreement,
                "association_correlation": correlation,
                "common_association_count": len(common),
            }
        )
    return pd.DataFrame(rows)


def viability_report(
    graphs: dict[str, nx.Graph],
    statistics: pd.DataFrame,
    similarities: pd.DataFrame,
    settings: dict[str, Any],
) -> dict[str, Any]:
    graph_count = len(graphs)
    signatures = {
        frozenset(tuple(sorted(edge)) for edge in graph.edges())
        for graph in graphs.values()
    }
    unique_fraction = len(signatures) / graph_count if graph_count else 0.0
    edge_min = int(statistics["edge_count"].min()) if graph_count else 0
    edge_max = int(statistics["edge_count"].max()) if graph_count else 0
    edge_median = float(statistics["edge_count"].median()) if graph_count else 0.0
    density_min = float(statistics["density"].min()) if graph_count else 0.0
    density_max = float(statistics["density"].max()) if graph_count else 0.0
    density_median = float(statistics["density"].median()) if graph_count else 0.0
    paired = similarities[
        (similarities["graph_id_a"] < similarities["graph_id_b"])
        & similarities["same_state"]
        & (~similarities["same_ses"])
    ]
    off_diagonal = similarities[
        similarities["graph_id_a"] < similarities["graph_id_b"]
    ]
    within_state_mean = (
        float(paired["edge_jaccard"].mean()) if not paired.empty else None
    )
    jaccard_summary = {
        "minimum": (
            float(off_diagonal["edge_jaccard"].min()) if not off_diagonal.empty else None
        ),
        "median": (
            float(off_diagonal["edge_jaccard"].median())
            if not off_diagonal.empty
            else None
        ),
        "maximum": (
            float(off_diagonal["edge_jaccard"].max()) if not off_diagonal.empty else None
        ),
    }

    criterion_a = (
        graph_count >= int(settings["minimum_eligible_graphs"])
        and unique_fraction >= float(settings["minimum_unique_edge_fraction"])
        and edge_max > edge_min
    )
    criterion_b = (
        float(settings["preferred_median_density_minimum"])
        <= density_median
        <= float(settings["preferred_median_density_maximum"])
    )
    density_caution = density_max >= float(settings.get("density_caution_threshold", 0.7))
    proceed = criterion_a and criterion_b
    return {
        "scope": "Phase 2 descriptive viability checkpoint",
        "eligible_graph_count": graph_count,
        "unique_edge_sets": len(signatures),
        "unique_edge_set_fraction": unique_fraction,
        "edge_count": {"minimum": edge_min, "median": edge_median, "maximum": edge_max},
        "density": {
            "minimum": density_min,
            "median": density_median,
            "maximum": density_max,
        },
        "off_diagonal_edge_jaccard": jaccard_summary,
        "mean_within_state_low_high_edge_jaccard": within_state_mean,
        "criteria": {
            "A_structural_heterogeneity": {
                "passed": criterion_a,
                "interpretation": "Edge sets vary across eligible population graphs.",
            },
            "B_usable_density": {
                "passed": criterion_b,
                "interpretation": "Median density lies inside the configured exploratory range.",
                "caution": (
                    "At least one graph meets the configured high-density caution threshold; "
                    "review thresholding or adjustment before motif mining."
                    if density_caution
                    else None
                ),
            },
            "C_recurrent_nontrivial_motifs": {
                "passed": None,
                "interpretation": "Not assessed before the Phase 2 stop.",
            },
            "D_ses_differentiation": {
                "passed": None,
                "interpretation": "Not tested inferentially in Phase 2.",
            },
            "E_modest_sensitivity": {
                "passed": None,
                "interpretation": "Deferred until the representation passes this checkpoint.",
            },
        },
        "phase_3_recommendation": (
            "proceed_to_motif_pipeline_design_with_density_review"
            if proceed and density_caution
            else "representation_ready_for_motif_pipeline_review"
            if proceed
            else "review_graph_representation_before_motif_mining"
        ),
        "interpretation_constraint": (
            "Edges are descriptive weighted associations, not causal links or "
            "design-based inferential results."
        ),
    }
