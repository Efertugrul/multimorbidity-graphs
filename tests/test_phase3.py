from __future__ import annotations

import json

import networkx as nx
import numpy as np
import pandas as pd

from mm_motifs.config import load_config
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
    support_counts,
)
from mm_motifs.motifs.robustness import (
    bootstrap_graph_masks,
    bootstrap_motif_support,
    degree_preserving_null_masks,
    fixed_edge_occurrence_probability,
    motif_shape_summary,
    paired_ses_permutation,
    restrict_to_frozen_edges,
)


def _edge_table() -> pd.DataFrame:
    rows = []
    graph_edges = {
        "1_lower": {("A", "B"), ("B", "C")},
        "1_higher": {("A", "B"), ("B", "C")},
        "2_lower": {("A", "B"), ("B", "C"), ("A", "C")},
        "2_higher": {("A", "B"), ("A", "C")},
    }
    for graph_id, selected in graph_edges.items():
        for source, target in (("A", "B"), ("A", "C"), ("B", "C")):
            rows.append(
                {
                    "graph_id": graph_id,
                    "rule_id": "stable",
                    "source_condition": source,
                    "target_condition": target,
                    "edge_present": (source, target) in selected,
                }
            )
    return pd.DataFrame(rows)


def test_phase3_configuration_is_frozen() -> None:
    config = load_config(
        "configs/analysis_phase25.yaml",
        "configs/conditions.yaml",
        "configs/graph_phase3.yaml",
    )
    settings = config.graph["phase3"]
    assert settings["graph_definition"] == {
        "point_phi_threshold": 0.12,
        "bootstrap_selection_threshold": 0.12,
        "minimum_selection_stability": 0.90,
    }
    assert settings["mining"]["support_fractions"] == [0.10, 0.20, 0.30]
    assert settings["mining"]["minimum_nodes"] == 3
    assert settings["mining"]["maximum_nodes"] == 5
    assert settings["mining"]["minimum_paired_states"] == 40
    assert settings["bootstrap"]["bank_role"] == "independent_evaluation"
    assert (
        settings["bootstrap"]["reporting_cutpoints_role"]
        == "descriptive_only"
    )
    assert (
        settings["bootstrap"]["seed"]
        != settings["bootstrap"]["selection_bank_seed"]
    )
    assert settings["decision"]["phase26_retention"] == "HOLD"
    assert settings["decision"]["structural_feasibility"] == "GO"


def test_gspan_mines_connected_non_induced_labeled_motifs() -> None:
    graphs, labels = graph_database_from_edges(_edge_table(), "stable")
    motifs, metadata = mine_frequent_motifs(
        graphs,
        labels,
        [0.50],
        0.50,
        3,
        3,
        1,
        "0.1.3",
    )
    target_edges = json.dumps(
        [["A", "B"], ["B", "C"]],
        separators=(",", ":"),
    )
    target = motifs[motifs["edge_list"].eq(target_edges)]
    assert len(target) == 1
    assert target.iloc[0]["support_count"] == 3
    assert len(motifs) == 2
    assert metadata["algorithm_backend"] == "gBolt gSpan"
    assert motifs["node_count"].eq(3).all()


def test_gspan_support_is_independently_reconstructed() -> None:
    graphs, labels = graph_database_from_edges(_edge_table(), "stable")
    motifs, _ = mine_frequent_motifs(
        graphs,
        labels,
        [0.50],
        0.50,
        3,
        3,
        1,
        "0.1.3",
    )
    universe = edge_universe(labels["condition"].tolist())
    edge_index = {edge: index for index, edge in enumerate(universe)}
    graph_masks = graph_edge_masks(graphs, edge_index)
    motif_masks = motif_edge_masks(motifs, edge_index)
    support = occurrence_matrix(graph_masks, motif_masks).sum(axis=0)
    assert np.array_equal(support, motifs["support_count"].to_numpy())


def test_occurrence_redundancy_identifies_closed_supergraph() -> None:
    graph_ids = json.dumps(["g1", "g2"], separators=(",", ":"))
    motifs = pd.DataFrame(
        {
            "motif_id": ["small", "large"],
            "node_count": [3, 4],
            "edge_count": [2, 3],
            "node_labels": [
                json.dumps(["A", "B", "C"]),
                json.dumps(["A", "B", "C", "D"]),
            ],
            "edge_list": [
                json.dumps([["A", "B"], ["B", "C"]]),
                json.dumps(
                    [["A", "B"], ["B", "C"], ["C", "D"]]
                ),
            ],
            "graph_ids": [graph_ids, graph_ids],
        }
    )
    result = annotate_redundancy(motifs)
    assert not bool(result.set_index("motif_id").loc["small", "is_closed"])
    assert bool(result.set_index("motif_id").loc["large", "is_closed"])
    assert result["occurrence_class_size"].eq(2).all()


def test_bootstrap_masks_preserve_joint_edge_realizations() -> None:
    bootstrap = pd.DataFrame(
        {
            "graph_id": ["g1", "g1", "g2", "g2"],
            "disease_a": ["A", "A", "A", "A"],
            "disease_b": ["B", "C", "B", "C"],
            "bootstrap_selection_mask": [
                "b1110",
                "b1100",
                "b1010",
                "b1011",
            ],
            "pair_eligible": [True] * 4,
        }
    )
    edge_index = {("A", "B"): 0, ("A", "C"): 1}
    graph_masks = bootstrap_graph_masks(
        bootstrap,
        ["g1", "g2"],
        edge_index,
        4,
    )
    motifs = pd.DataFrame(
        {
            "motif_id": ["m"],
            "node_count": [3],
            "edge_count": [2],
            "support_count": [1],
            "paired_state_count": [1],
            "evaluable_graph_count": [2],
            "support_count_50pct": [1],
            "support_50pct": [True],
            "primary_support": [True],
            "edge_list": [
                json.dumps([["A", "B"], ["A", "C"]])
            ],
        }
    )
    summary, long, support = bootstrap_motif_support(
        graph_masks,
        motifs,
        edge_index,
        [0.50],
        np.ones((2, 1), dtype=bool),
    )
    assert support[:, 0].tolist() == [2, 1, 1, 0]
    assert summary.loc[0, "p_boot_support_50pct"] == 0.75
    assert len(long) == 4


def test_primary_bootstrap_prevents_unselected_edges_from_entering() -> None:
    threshold_masks = np.asarray(
        [[0b111, 0b110], [0b101, 0b011]],
        dtype=np.uint64,
    )
    frozen_masks = np.asarray([0b011, 0b010], dtype=np.uint64)
    restricted = restrict_to_frozen_edges(
        threshold_masks,
        frozen_masks,
    )
    assert restricted.tolist() == [[0b011, 0b010], [0b001, 0b010]]


def test_opportunity_support_uses_pair_complete_states() -> None:
    graph_ids = ["1_lower", "1_higher", "2_lower", "2_higher"]
    graph_masks = np.asarray([0b001, 0b011, 0b011, 0b001], dtype=np.uint64)
    opportunity_masks = np.asarray(
        [0b001, 0b111, 0b111, 0b111],
        dtype=np.uint64,
    )
    motifs = pd.DataFrame(
        {
            "motif_id": ["m"],
            "node_count": [3],
            "edge_count": [2],
            "edge_list": [json.dumps([["A", "B"], ["A", "C"]])],
            "support_count": [2],
            "support_fraction": [0.5],
            "graph_ids": [json.dumps(["1_higher", "2_lower"])],
        }
    )
    registry = pd.DataFrame(
        {
            "graph_id": graph_ids,
            "geography_code": [1, 1, 2, 2],
            "ses_category": ["lower", "higher", "lower", "higher"],
        }
    )
    result, evaluable = annotate_opportunity_support(
        motifs,
        graph_masks,
        opportunity_masks,
        graph_ids,
        registry,
        {("A", "B"): 0, ("A", "C"): 1, ("B", "C"): 2},
        [0.50],
        0.50,
        1,
    )
    assert result.loc[0, "paired_state_count"] == 1
    assert result.loc[0, "evaluable_graph_count"] == 2
    assert result.loc[0, "support_count"] == 1
    assert result.loc[0, "distinct_state_support_count"] == 1
    assert json.loads(result.loc[0, "graph_ids"]) == ["2_lower"]
    assert evaluable[:, 0].tolist() == [False, False, True, True]
    swapped, _ = annotate_opportunity_support(
        motifs,
        graph_masks[[1, 0, 3, 2]],
        opportunity_masks[[1, 0, 3, 2]],
        graph_ids,
        registry,
        {("A", "B"): 0, ("A", "C"): 1, ("B", "C"): 2},
        [0.50],
        0.50,
        1,
    )
    assert swapped.loc[0, "support_count"] == result.loc[0, "support_count"]
    assert bool(swapped.loc[0, "support_50pct"])


def test_graph_opportunity_masks_distinguish_missing_from_absent() -> None:
    rows = []
    for graph_id in ["g1", "g2"]:
        for source, target in [("A", "B"), ("A", "C"), ("B", "C")]:
            rows.append(
                {
                    "graph_id": graph_id,
                    "rule_id": "stable",
                    "source_condition": source,
                    "target_condition": target,
                    "pair_eligible": not (
                        graph_id == "g1" and (source, target) == ("A", "C")
                    ),
                }
            )
    masks = graph_opportunity_masks(
        pd.DataFrame(rows),
        "stable",
        ["g1", "g2"],
        {("A", "B"): 0, ("A", "C"): 1, ("B", "C"): 2},
    )
    assert masks.tolist() == [0b101, 0b111]


def test_fixed_edge_density_probability_is_exact() -> None:
    probability = fixed_edge_occurrence_probability(6, 3, 2)
    assert np.isclose(probability, 0.2)


def test_degree_null_preserves_labeled_degree_sequence() -> None:
    universe = edge_universe(["A", "B", "C", "D"])
    edge_index = {edge: index for index, edge in enumerate(universe)}
    baseline_edges = {
        ("A", "B"),
        ("A", "D"),
        ("B", "C"),
        ("C", "D"),
    }
    baseline_mask = sum(1 << edge_index[edge] for edge in baseline_edges)
    masks, diagnostics = degree_preserving_null_masks(
        np.asarray([baseline_mask], dtype=np.uint64),
        universe,
        10,
        4,
        7,
    )
    baseline = nx.Graph()
    baseline.add_nodes_from(["A", "B", "C", "D"])
    baseline.add_edges_from(baseline_edges)
    expected = dict(baseline.degree())
    for mask in masks[:, 0]:
        graph = nx.Graph()
        graph.add_nodes_from(expected)
        graph.add_edges_from(
            edge
            for index, edge in enumerate(universe)
            if int(mask) & (1 << index)
        )
        assert dict(graph.degree()) == expected
    assert diagnostics["changed"].any()


def test_degree_null_never_uses_ineligible_dyads() -> None:
    universe = edge_universe(["A", "B", "C", "D", "E"])
    edge_index = {edge: index for index, edge in enumerate(universe)}
    baseline_edges = {
        ("A", "B"),
        ("A", "D"),
        ("A", "E"),
        ("B", "C"),
        ("C", "D"),
        ("D", "E"),
    }
    baseline_mask = np.uint64(
        sum(1 << edge_index[edge] for edge in baseline_edges)
    )
    disallowed = ("A", "C")
    opportunity_mask = np.uint64(
        sum(
            1 << index
            for index, edge in enumerate(universe)
            if edge != disallowed
        )
    )
    masks, _ = degree_preserving_null_masks(
        np.asarray([baseline_mask], dtype=np.uint64),
        universe,
        20,
        4,
        11,
        np.asarray([opportunity_mask], dtype=np.uint64),
    )
    disallowed_bit = np.uint64(1) << np.uint64(edge_index[disallowed])
    assert not np.bitwise_and(masks[:, 0], disallowed_bit).any()


def test_paired_ses_permutation_uses_density_residuals() -> None:
    graph_ids = [
        "1_lower",
        "1_higher",
        "2_lower",
        "2_higher",
        "3_lower",
        "3_higher",
    ]
    graph_masks = np.asarray(
        [
            0b011,
            0b001,
            0b011,
            0b001,
            0b001,
            0b001,
        ],
        dtype=np.uint64,
    )
    motifs = pd.DataFrame(
        {
            "motif_id": ["m"],
            "edge_count": [2],
            "edge_list": [
                json.dumps([["A", "B"], ["A", "C"]])
            ],
        }
    )
    registry = pd.DataFrame(
        {
            "graph_id": graph_ids,
            "geography_code": [1, 1, 2, 2, 3, 3],
            "ses_category": [
                "lower",
                "higher",
                "lower",
                "higher",
                "lower",
                "higher",
            ],
        }
    )
    result = paired_ses_permutation(
        graph_ids,
        graph_masks,
        np.asarray(
            [0b111, 0b111, 0b111, 0b111, 0b101, 0b101],
            dtype=np.uint64,
        ),
        motifs,
        registry,
        {("A", "B"): 0, ("A", "C"): 1, ("B", "C"): 2},
        99,
        7,
        2,
    )
    assert result.loc[0, "state_pair_count"] == 2
    assert result.loc[0, "permutation_scheme"] == (
        "within_state_lower_higher_label_swap"
    )
    assert 0 < result.loc[0, "permutation_p_value"] <= 1
    assert result.loc[0, "permutation_max_t_p_value_mcse"] >= 0


def test_support_percentages_use_ceiling_counts() -> None:
    assert support_counts(94, [0.10, 0.20, 0.30]) == {
        0.10: 10,
        0.20: 19,
        0.30: 29,
    }


def test_shape_summary_ignores_support_threshold_columns() -> None:
    motifs = pd.DataFrame(
        {
            "node_count": [3],
            "edge_count": [2],
            "motif_density": [2 / 3],
            "is_tree": [True],
            "is_clique": [False],
            "is_closed": [True],
            "is_maximal": [True],
            "occurrence_class_id": ["c1"],
            "support_10pct": [True],
            "support_count_10pct": [10],
        }
    )
    summary = motif_shape_summary(motifs)
    assert summary["support_level"].tolist() == ["support_10pct"]
