from __future__ import annotations

import json
import shutil
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from mm_motifs.config import load_config
from mm_motifs.graphs.phase26 import (
    adjusted_concordance,
    build_phi_edge_scenarios,
    phase26_viability_report,
    point_phi_stability_diagnostic,
    threshold_topology,
)
from mm_motifs.statistics.disease_association import weighted_phi
from mm_motifs.statistics.phi_bootstrap import run_phi_bootstrap


THRESHOLDS = [
    {"rule_id": "phi_010", "minimum_effect": 0.10},
    {"rule_id": "phi_012", "minimum_effect": 0.12},
    {"rule_id": "phi_014", "minimum_effect": 0.14},
]


def _pair_counts() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "graph_id": ["g", "g", "g"],
            "source_condition": ["A", "A", "B"],
            "target_condition": ["B", "C", "C"],
            "association": [0.15, 0.13, 0.11],
            "pair_eligible": [True, True, True],
            "estimator": ["weighted_phi"] * 3,
            "inference_scope": ["descriptive"] * 3,
            "n_complete": [1000] * 3,
            "cooccurring_cases": [50] * 3,
        }
    )


def _bootstrap_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "graph_id": ["g", "g", "g"],
            "disease_a": ["A", "A", "B"],
            "disease_b": ["B", "C", "C"],
            "point_phi": [0.15, 0.13, 0.11],
            "p_phi_gt_zero": [1.0, 1.0, 1.0],
            "p_phi_ge_012": [0.95, 0.89, 0.99],
            "bootstrap_phi_median": [0.15, 0.13, 0.11],
            "bootstrap_phi_025": [0.12, 0.10, 0.08],
            "bootstrap_phi_975": [0.18, 0.16, 0.14],
            "selection_stability_standard_error": [0.01] * 3,
            "selection_stability_wilson_low": [0.92, 0.85, 0.97],
            "selection_stability_wilson_high": [0.98, 0.92, 1.0],
            "bootstrap_replicates_requested": [500] * 3,
            "bootstrap_replicates_valid": [500] * 3,
            "bootstrap_status": ["ok"] * 3,
            "bootstrap_message": [""] * 3,
            "pair_eligible": [True] * 3,
        }
    )


def test_phase26_configuration_is_fixed() -> None:
    config = load_config(
        "configs/analysis_phase25.yaml",
        "configs/conditions.yaml",
        "configs/graph_phase26.yaml",
    )
    settings = config.graph["phase26"]
    assert [rule["minimum_effect"] for rule in settings["thresholds"]] == [
        0.10,
        0.12,
        0.14,
    ]
    assert settings["bootstrap"]["replicates"] == 500
    assert settings["bootstrap"]["selection_stability_threshold"] == 0.90


def test_phase26_archived_result_is_frozen() -> None:
    archive = Path("results/baselines/phase26_172713cfdaf6")
    manifest = json.loads(
        (archive / "manifest.json").read_text(encoding="utf-8")
    )
    report = json.loads(
        (archive / "viability_report.json").read_text(encoding="utf-8")
    )
    diagnostic = json.loads(
        (archive / "point_phi_stability_diagnostic.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["config_digest"] == "172713cfdaf6"
    assert manifest["git_commit"].startswith("6663f67")
    assert manifest["git_dirty"] is False
    assert report["decision"] == "HOLD"
    assert report["criteria"]["most_raw_edges_stable"] is False
    assert report["diagnostics"]["raw_phi_012_edge_count"] == 1733
    assert report["diagnostics"]["stable_phi_012_edge_count"] == 1057
    assert np.isclose(
        diagnostic["point_phi_selection_stability_spearman"],
        0.891697162147299,
    )
    assert (
        diagnostic["cutoff_diagnostics"]["phi_ge_015"][
            "unstable_edges_below_cutoff"
        ]
        == 560
    )


def test_stable_phi_rule_uses_point_and_selection_thresholds() -> None:
    result = build_phi_edge_scenarios(
        _pair_counts(),
        _bootstrap_results(),
        THRESHOLDS,
        "phi_012",
        "phi_012_stable",
        0.90,
    )
    assert result.groupby("rule_id")["edge_present"].sum().to_dict() == {
        "phi_010": 3,
        "phi_012": 2,
        "phi_012_stable": 1,
        "phi_014": 1,
    }


def test_point_phi_diagnostic_does_not_relabel_stability_threshold() -> None:
    bootstrap = _bootstrap_results()
    bootstrap.loc[2, "point_phi"] = 0.20
    summary, diagnostic = point_phi_stability_diagnostic(
        bootstrap,
        0.12,
        0.90,
    )
    assert summary["raw_edges"].sum() == 3
    assert summary["stable_edges"].sum() == 2
    assert diagnostic["cutoff_diagnostics"]["phi_ge_015"][
        "stable_fraction_against_phi_012"
    ] == 1.0
    assert "P_boot(phi >= 0.15)" in diagnostic["interpretation_boundary"]


def _graph(graph_id: str, edges: list[tuple[str, str]]) -> nx.Graph:
    graph = nx.Graph(
        graph_id=graph_id,
        population_graph_id=graph_id,
        state_code=1,
        ses_category="lower",
    )
    graph.add_nodes_from(["A", "B", "C", "D"])
    graph.add_edges_from(edges)
    return graph


def test_threshold_topology_quantifies_nested_sparsification() -> None:
    collections = {
        ("education_binary", "phi_010"): {
            "g": _graph("g", [("A", "B"), ("A", "C"), ("B", "C")])
        },
        ("education_binary", "phi_012"): {
            "g": _graph("g", [("A", "B"), ("A", "C")])
        },
        ("education_binary", "phi_014"): {
            "g": _graph("g", [("A", "B")])
        },
    }
    _, transitions, summary = threshold_topology(
        collections,
        "education_binary",
        THRESHOLDS,
    )
    assert transitions["nested"].all()
    adjacent = summary[summary["transition_type"] == "adjacent"]
    assert np.allclose(
        adjacent["median_edge_jaccard"].sort_values().to_numpy(),
        [0.5, 2 / 3],
    )
    assert np.isclose(
        summary[summary["transition_type"] == "outer"][
            "median_edge_jaccard"
        ].iloc[0],
        1 / 3,
    )


def test_adjusted_models_are_validation_only() -> None:
    scenarios = build_phi_edge_scenarios(
        _pair_counts(),
        _bootstrap_results(),
        THRESHOLDS,
        "phi_012",
        "phi_012_stable",
        0.90,
    )
    adjusted = pd.DataFrame(
        {
            "graph_id": ["g", "g", "g"],
            "source_condition": ["A", "A", "B"],
            "target_condition": ["B", "C", "C"],
            "model_status": ["ok"] * 3,
            "converged": [True] * 3,
            "log_odds_ratio": [0.5, -0.1, 0.2],
            "odds_ratio": [1.6, 0.9, 1.2],
            "confidence_low": [1.2, 0.7, 0.9],
            "confidence_high": [2.0, 1.2, 1.5],
            "q_value": [0.01, 0.2, 0.1],
            "reverse_model_status": ["ok"] * 3,
            "reverse_converged": [True] * 3,
            "reverse_log_odds_ratio": [0.4, -0.2, 0.1],
            "reverse_odds_ratio": [1.5, 0.8, 1.1],
            "direction_concordant": [True] * 3,
            "edge_present": [True, False, False],
        }
    )
    _, summary = adjusted_concordance(
        scenarios,
        adjusted,
        ["phi_012", "phi_012_stable"],
    )
    raw = summary[summary["rule_id"] == "phi_012"].iloc[0]
    assert raw["selected_edge_count"] == 2
    assert raw["adjusted_positive_both_directions_count"] == 1
    stable = summary[
        summary["rule_id"] == "phi_012_stable"
    ].iloc[0]
    assert stable["adjusted_fdr_supported_fraction"] == 1


def _viability_inputs() -> tuple:
    statistics = pd.DataFrame(
        {
            "rule_id": ["phi_012_stable"] * 80,
            "edge_count": [15] * 80,
            "density": [1 / 3] * 80,
        }
    )
    similarities = pd.DataFrame(
        {
            "rule_id": ["phi_012_stable"],
            "graph_id_a": ["a"],
            "graph_id_b": ["b"],
            "edge_jaccard": [0.7],
            "density_adjusted_jaccard_z": [3.0],
        }
    )
    support = pd.DataFrame(
        {
            "rule_id": ["phi_012_stable"] * 5,
            "support_fraction": [0.5] * 5,
        }
    )
    bootstrap = _bootstrap_results()
    bootstrap["p_phi_ge_012"] = [0.95, 0.95, 0.95]
    continuity = pd.DataFrame(
        {
            "transition_type": ["adjacent", "adjacent", "outer"],
            "median_edge_jaccard": [0.8, 0.8, 0.6],
            "fraction_edge_jaccard_below_050": [0.0, 0.0, 0.0],
            "all_graphs_nested": [True, True, True],
        }
    )
    concordance = pd.DataFrame(
        {
            "rule_id": ["phi_012"],
            "adjusted_positive_both_directions_fraction": [0.95],
            "adjusted_fdr_supported_fraction": [0.92],
        }
    )
    settings = {
        "minimum_eligible_graphs": 80,
        "preferred_edge_count_minimum": 10,
        "preferred_edge_count_maximum": 20,
        "maximum_density_caution": 0.70,
        "selection_stability_threshold": 0.90,
        "minimum_raw_edge_stable_fraction": 0.80,
        "minimum_median_adjacent_jaccard": 0.70,
        "maximum_fraction_adjacent_jaccard_below_050": 0.25,
        "minimum_median_outer_jaccard": 0.50,
        "minimum_positive_concordance": 0.90,
        "maximum_median_raw_jaccard": 0.95,
        "minimum_median_density_adjusted_z": 0.0,
        "minimum_variable_recurrent_edges": 5,
    }
    return (
        statistics,
        similarities,
        support,
        bootstrap,
        continuity,
        concordance,
        settings,
    )


def test_phase26_go_requires_all_neutral_criteria() -> None:
    inputs = _viability_inputs()
    report = phase26_viability_report(
        *inputs[:-1],
        "phi_012_stable",
        "phi_012",
        inputs[-1],
    )
    assert report["decision"] == "GO"
    inputs[0].loc[0, "density"] = 0.8
    report = phase26_viability_report(
        *inputs[:-1],
        "phi_012_stable",
        "phi_012",
        inputs[-1],
    )
    assert report["decision"] == "HOLD"


@pytest.mark.skipif(
    not Path(".conda/r-survey/bin/Rscript").exists()
    and shutil.which("Rscript") is None,
    reason="R survey runtime is not installed",
)
def test_phi_bootstrap_r_smoke(tmp_path: Path) -> None:
    generator = np.random.default_rng(13)
    size = 240
    disease_a = generator.binomial(1, 0.35, size)
    disease_b = generator.binomial(1, 0.15 + 0.5 * disease_a, size)
    data = pd.DataFrame(
        {
            "state_code": 1,
            "survey_weight": generator.uniform(0.5, 2.0, size),
            "survey_strata": np.repeat(np.arange(1, 5), size // 4),
            "survey_psu": np.arange(size),
            "A": disease_a,
            "B": disease_b,
            "ses_education_binary": np.tile(
                ["lower", "higher"],
                size // 2,
            ),
        }
    )
    data_path = tmp_path / "data.csv"
    data.to_csv(data_path, index=False)
    point_phi = {}
    for category in ("lower", "higher"):
        domain = data[data["ses_education_binary"] == category]
        point_phi[category] = weighted_phi(
            domain["A"],
            domain["B"],
            domain["survey_weight"],
        )["association"]
    jobs = pd.DataFrame(
        [
            {
                "graph_id": "g_lower",
                "state_code": 1,
                "ses_definition": "education_binary",
                "ses_category": "lower",
                "disease_a": "A",
                "disease_b": "B",
                "point_phi": point_phi["lower"],
            },
            {
                "graph_id": "g_higher",
                "state_code": 1,
                "ses_definition": "education_binary",
                "ses_category": "higher",
                "disease_a": "A",
                "disease_b": "B",
                "point_phi": point_phi["higher"],
            },
        ]
    )
    result = run_phi_bootstrap(
        Path.cwd(),
        data_path,
        jobs,
        tmp_path / "r",
        20,
        0.12,
        7,
        1,
    )
    assert len(result) == 2
    assert result["bootstrap_status"].eq("ok").all()
    assert result["bootstrap_replicates_valid"].eq(20).all()
    assert result["bootstrap_selection_mask"].str.match(
        r"^b[01]{20}$"
    ).all()
