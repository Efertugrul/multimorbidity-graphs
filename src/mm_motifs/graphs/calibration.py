from __future__ import annotations

import math
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from mm_motifs.graphs.qc import fixed_edge_null_jaccard


def classify_similarity_pairs(similarities: pd.DataFrame) -> pd.DataFrame:
    pairs = similarities[
        similarities["graph_id_a"] < similarities["graph_id_b"]
    ].copy()
    conditions = [
        pairs["same_state"] & ~pairs["same_ses"],
        ~pairs["same_state"] & pairs["same_ses"],
        ~pairs["same_state"] & ~pairs["same_ses"],
    ]
    labels = [
        "within_state_different_ses",
        "same_ses_cross_state",
        "cross_ses_cross_state",
    ]
    pairs["comparison_type"] = np.select(conditions, labels, default="other")
    return pairs


def similarity_summary(similarities: pd.DataFrame) -> pd.DataFrame:
    pairs = classify_similarity_pairs(similarities)
    return (
        pairs[pairs["comparison_type"] != "other"]
        .groupby("comparison_type", as_index=False)
        .agg(
            pair_count=("edge_jaccard", "size"),
            mean_jaccard=("edge_jaccard", "mean"),
            median_jaccard=("edge_jaccard", "median"),
            standard_deviation=("edge_jaccard", "std"),
            minimum_jaccard=("edge_jaccard", "min"),
            maximum_jaccard=("edge_jaccard", "max"),
            mean_density_adjusted_z=("density_adjusted_jaccard_z", "mean"),
            median_density_adjusted_z=("density_adjusted_jaccard_z", "median"),
        )
    )


def _edge_jaccard(graph_a: nx.Graph, graph_b: nx.Graph) -> float:
    edges_a = {tuple(sorted(edge)) for edge in graph_a.edges()}
    edges_b = {tuple(sorted(edge)) for edge in graph_b.edges()}
    union = edges_a | edges_b
    return len(edges_a & edges_b) / len(union) if union else 1.0


def _batched_swap_statistics(
    similarity: np.ndarray,
    swaps: np.ndarray,
    batch_size: int = 500,
) -> np.ndarray:
    state_count = swaps.shape[1]
    state_a, state_b = np.triu_indices(state_count, k=1)
    lower_a = 2 * state_a
    higher_a = lower_a + 1
    lower_b = 2 * state_b
    higher_b = lower_b + 1
    ll = similarity[lower_a, lower_b]
    lh = similarity[lower_a, higher_b]
    hl = similarity[higher_a, lower_b]
    hh = similarity[higher_a, higher_b]
    total = ll + lh + hl + hh
    output = np.empty((len(swaps), 7), dtype=float)
    for start in range(0, len(swaps), batch_size):
        stop = min(len(swaps), start + batch_size)
        swap_a = swaps[start:stop, state_a]
        swap_b = swaps[start:stop, state_b]
        same_orientation = swap_a == swap_b
        lower_values = np.where(
            ~swap_a,
            np.where(~swap_b, ll, lh),
            np.where(~swap_b, hl, hh),
        )
        higher_values = np.where(
            ~swap_a,
            np.where(~swap_b, hh, hl),
            np.where(~swap_b, lh, ll),
        )
        same_sum = np.where(same_orientation, ll + hh, lh + hl)
        cross_sum = total - same_sum
        lower_mean = lower_values.mean(axis=1)
        higher_mean = higher_values.mean(axis=1)
        same_mean = same_sum.mean(axis=1) / 2
        cross_mean = cross_sum.mean(axis=1) / 2
        output[start:stop] = np.column_stack(
            [
                same_mean,
                lower_mean,
                higher_mean,
                cross_mean,
                same_mean - cross_mean,
                lower_mean - cross_mean,
                higher_mean - cross_mean,
            ]
        )
    return output


def ses_label_permutation_test(
    graphs: dict[str, nx.Graph],
    permutations: int,
    seed: int,
) -> pd.DataFrame:
    paired: dict[int, dict[str, nx.Graph]] = {}
    for graph in graphs.values():
        state = int(graph.graph["state_code"])
        paired.setdefault(state, {})[str(graph.graph["ses_category"])] = graph
    paired = {
        state: values
        for state, values in paired.items()
        if {"lower", "higher"} <= set(values)
    }
    if len(paired) < 2:
        raise ValueError("At least two complete state SES pairs are required")

    ordered_states = sorted(paired)
    ordered_graphs = [
        paired[state][category]
        for state in ordered_states
        for category in ("lower", "higher")
    ]
    size = len(ordered_graphs)
    possible_edges = math.comb(
        max(graph.number_of_nodes() for graph in ordered_graphs),
        2,
    )
    raw_similarity = np.eye(size)
    adjusted_similarity = np.zeros((size, size), dtype=float)
    for row in range(size):
        for column in range(row + 1, size):
            raw = _edge_jaccard(ordered_graphs[row], ordered_graphs[column])
            null = fixed_edge_null_jaccard(
                ordered_graphs[row].number_of_edges(),
                ordered_graphs[column].number_of_edges(),
                possible_edges,
                raw,
            )
            raw_similarity[row, column] = raw
            raw_similarity[column, row] = raw
            adjusted_similarity[row, column] = null["z_score"]
            adjusted_similarity[column, row] = null["z_score"]

    names = [
        "same_ses_mean",
        "lower_ses_mean",
        "higher_ses_mean",
        "cross_ses_mean",
        "same_minus_cross",
        "lower_minus_cross",
        "higher_minus_cross",
    ]
    rows: list[dict[str, Any]] = []
    generator = np.random.default_rng(seed)
    swaps = generator.integers(
        0,
        2,
        size=(permutations, len(ordered_states)),
        dtype=np.int8,
    ).astype(bool)
    observed_swaps = np.zeros((1, len(ordered_states)), dtype=bool)
    for metric_name, similarity in (
        ("edge_jaccard", raw_similarity),
        ("density_adjusted_jaccard_z", adjusted_similarity),
    ):
        observed = _batched_swap_statistics(
            similarity,
            observed_swaps,
        )[0]
        null_statistics = _batched_swap_statistics(similarity, swaps)
        for column, name in enumerate(names):
            null_values = null_statistics[:, column]
            rows.append(
                {
                    "similarity_metric": metric_name,
                    "metric": name,
                    "observed": observed[column],
                    "null_mean": float(null_values.mean()),
                    "null_standard_deviation": float(null_values.std(ddof=1)),
                    "null_q025": float(np.quantile(null_values, 0.025)),
                    "null_q975": float(np.quantile(null_values, 0.975)),
                    "p_value_one_sided_greater": float(
                        (1 + np.count_nonzero(null_values >= observed[column]))
                        / (permutations + 1)
                    ),
                    "permutations": permutations,
                    "paired_states": len(ordered_states),
                    "permutation_scheme": "within_state_lower_higher_label_swap",
                    "null_hypothesis": (
                        "Graph structure is unrelated to SES label conditional on state"
                    ),
                }
            )
    return pd.DataFrame(rows)


def graph_outlier_scores(
    graphs: dict[str, nx.Graph],
    population_registry: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    values = list(graphs.values())
    possible_edges = math.comb(
        max(graph.number_of_nodes() for graph in values),
        2,
    )
    for graph in values:
        peers = [
            other
            for other in values
            if other.graph["graph_id"] != graph.graph["graph_id"]
            and other.graph["ses_category"] == graph.graph["ses_category"]
        ]
        similarities = []
        adjusted_similarities = []
        for peer in peers:
            raw = _edge_jaccard(graph, peer)
            similarities.append(raw)
            adjusted_similarities.append(
                fixed_edge_null_jaccard(
                    graph.number_of_edges(),
                    peer.number_of_edges(),
                    possible_edges,
                    raw,
                )["z_score"]
            )
        rows.append(
            {
                "graph_id": graph.graph["population_graph_id"],
                "geography_code": graph.graph["state_code"],
                "geography": graph.graph["geography"],
                "ses_category": graph.graph["ses_category"],
                "edge_count": graph.number_of_edges(),
                "density": nx.density(graph),
                "mean_same_ses_jaccard": (
                    float(np.mean(similarities)) if similarities else math.nan
                ),
                "median_same_ses_jaccard": (
                    float(np.median(similarities)) if similarities else math.nan
                ),
                "mean_same_ses_density_adjusted_z": (
                    float(np.mean(adjusted_similarities))
                    if adjusted_similarities
                    else math.nan
                ),
            }
        )
    result = pd.DataFrame(rows)
    for _, indices in result.groupby("ses_category").groups.items():
        raw = result.loc[indices, "mean_same_ses_jaccard"]
        raw_median = float(raw.median())
        raw_deviation = float(np.median(np.abs(raw - raw_median)))
        result.loc[indices, "robust_similarity_z"] = (
            0.67448975 * (raw - raw_median) / raw_deviation
            if raw_deviation > 0
            else 0.0
        )
        adjusted = result.loc[indices, "mean_same_ses_density_adjusted_z"]
        adjusted_median = float(adjusted.median())
        adjusted_deviation = float(
            np.median(np.abs(adjusted - adjusted_median))
        )
        result.loc[indices, "robust_density_adjusted_similarity_z"] = (
            0.67448975
            * (adjusted - adjusted_median)
            / adjusted_deviation
            if adjusted_deviation > 0
            else 0.0
        )
    registry_columns = [
        "graph_id",
        "n_unweighted",
        "weighted_population_estimate",
        "kish_effective_n",
        "n_strata",
        "n_psu",
    ]
    result = result.merge(
        population_registry[registry_columns],
        on="graph_id",
        how="left",
    )
    return result.sort_values("mean_same_ses_jaccard")
