from __future__ import annotations

import json
from pathlib import Path
import shutil

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from mm_motifs.config import load_config
from mm_motifs.features.populations import phase25_population_registry
from mm_motifs.graphs.calibration import (
    _batched_swap_statistics,
    ses_label_permutation_test,
)
from mm_motifs.graphs.construct import construct_graph
from mm_motifs.graphs.qc import fixed_edge_null_jaccard
from mm_motifs.statistics.multiple_testing import benjamini_hochberg
from mm_motifs.statistics.r_survey import (
    _model_cache_key,
    adjusted_edge_table,
    run_survey_models,
)
from mm_motifs.workflows.phase25 import (
    _add_stability_intervals,
    _bootstrap_stable_table,
)


def test_phase25_registry_reports_eligibility_and_effective_n() -> None:
    frame = pd.DataFrame(
        {
            "state_code": [1, 1, 1, 1, 3, 3, 3],
            "ses_education_binary": [
                "lower",
                "lower",
                "higher",
                "higher",
                "lower",
                "lower",
                "higher",
            ],
            "survey_weight": [1.0, 3.0, 2.0, 2.0, 1.0, 1.0, 1.0],
            "survey_strata": [1, 1, 2, 2, 3, 3, 4],
            "survey_psu": [1, 2, 3, 4, 5, 6, 7],
        }
    )
    registry = phase25_population_registry(
        frame,
        2024,
        "education_binary",
        minimum_n=2,
        geography_codes=[1, 2, 3],
    )
    state_one = registry[registry["geography_code"] == 1]
    assert state_one["eligibility_status"].eq("eligible").all()
    lower = state_one[state_one["ses_category"] == "lower"].iloc[0]
    assert np.isclose(lower["kish_effective_n"], 1.6)
    assert registry[registry["geography_code"] == 2][
        "eligibility_status"
    ].eq("no reporting data").all()
    state_three = registry[registry["geography_code"] == 3]
    assert (
        state_three[state_three["ses_category"] == "lower"][
            "eligibility_status"
        ].iloc[0]
        == "paired SES stratum ineligible"
    )


def test_phase25_configuration_honors_year_override() -> None:
    config = load_config(
        "configs/analysis_phase25.yaml",
        "configs/conditions.yaml",
        "configs/graph_phase25.yaml",
        year=2023,
    )
    assert config.year == 2023


def test_fixed_edge_null_standardizes_for_density() -> None:
    result = fixed_edge_null_jaccard(30, 30, 45, observed_jaccard=0.75)
    assert result["expected"] > 0.4
    assert result["z_score"] > 0
    assert 0 <= result["p_value_upper"] <= 1


def _paired_graphs() -> dict[str, nx.Graph]:
    graphs = {}
    for state in range(1, 5):
        for category, edges in (
            ("lower", [("A", "B"), ("A", "C")]),
            ("higher", [("B", "D"), ("C", "D")]),
        ):
            graph_id = f"{state}_{category}"
            graph = nx.Graph(
                graph_id=graph_id,
                population_graph_id=graph_id,
                state_code=state,
                geography=str(state),
                ses_category=category,
            )
            graph.add_nodes_from(["A", "B", "C", "D"])
            graph.add_edges_from(edges)
            graphs[graph_id] = graph
    return graphs


def test_permutation_is_explicit_within_state_label_swap() -> None:
    result = ses_label_permutation_test(_paired_graphs(), permutations=500, seed=7)
    assert set(result["similarity_metric"]) == {
        "edge_jaccard",
        "density_adjusted_jaccard_z",
    }
    assert result["permutation_scheme"].eq(
        "within_state_lower_higher_label_swap"
    ).all()
    raw_contrast = result[
        (result["similarity_metric"] == "edge_jaccard")
        & (result["metric"] == "same_minus_cross")
    ].iloc[0]
    assert raw_contrast["observed"] > raw_contrast["null_mean"]


def test_paired_swap_keeps_state_graphs_together() -> None:
    similarity = np.eye(4)
    similarity[0, 2] = similarity[2, 0] = 0.9
    similarity[0, 3] = similarity[3, 0] = 0.1
    similarity[1, 2] = similarity[2, 1] = 0.2
    similarity[1, 3] = similarity[3, 1] = 0.8
    result = _batched_swap_statistics(
        similarity,
        np.asarray([[False, False], [False, True]]),
    )
    assert np.allclose(result[0, :4], [0.85, 0.9, 0.8, 0.15])
    assert np.allclose(result[1, :4], [0.15, 0.1, 0.2, 0.85])


def test_adjusted_edge_rule_and_reverse_or_are_recorded() -> None:
    models = pd.DataFrame(
        {
            "graph_id": ["g", "g"],
            "source_condition": ["A", "A"],
            "target_condition": ["B", "C"],
            "model_status": ["ok", "ok"],
            "log_odds_ratio": [np.log(2), np.log(1.1)],
            "reverse_log_odds_ratio": [np.log(1.8), np.log(1.05)],
            "confidence_low_log": [np.log(1.2), np.log(0.9)],
            "confidence_high_log": [np.log(3), np.log(1.3)],
            "p_value": [0.001, 0.2],
            "converged": [True, True],
        }
    )
    counts = pd.DataFrame(
        {
            "graph_id": ["g", "g"],
            "source_condition": ["A", "A"],
            "target_condition": ["B", "C"],
            "n_complete": [1000, 1000],
            "source_cases": [100, 100],
            "target_cases": [100, 100],
            "cooccurring_cases": [50, 50],
            "weighted_n": [10000.0, 10000.0],
            "kish_effective_n": [800.0, 800.0],
            "pair_eligible": [True, True],
            "exclusion_reason": ["", ""],
        }
    )
    result = adjusted_edge_table(models, counts, 0.05, benjamini_hochberg)
    assert bool(result.loc[0, "edge_present"])
    assert not bool(result.loc[1, "edge_present"])
    assert np.isclose(result.loc[0, "reverse_odds_ratio"], 1.8)


def test_empty_bootstrap_candidates_have_typed_schema(tmp_path: Path) -> None:
    jobs = pd.DataFrame(
        columns=[
            "graph_id",
            "state_code",
            "ses_definition",
            "ses_category",
            "source_condition",
            "target_condition",
        ]
    )
    result = run_survey_models(
        tmp_path,
        "bootstrap",
        tmp_path / "missing.csv",
        jobs,
        tmp_path / "output",
        0.95,
        200,
        7,
        1,
    )
    assert result.empty
    assert "positive_stability" in result
    assert result.attrs["model_cache_key"] == "empty"


def test_model_cache_key_tracks_code_data_and_parameters(tmp_path: Path) -> None:
    script = tmp_path / "model.R"
    data = tmp_path / "data.csv"
    script.write_text("fit <- 1\n")
    data.write_text("x\n1\n")
    jobs = pd.DataFrame(
        {
            "graph_id": ["g"],
            "source_condition": ["A"],
            "target_condition": ["B"],
        }
    )
    arguments = (
        script,
        data,
        jobs,
        "point",
        0.95,
        0,
        7,
        {"r": "4.4", "survey": "4.5"},
    )
    first = _model_cache_key(*arguments)
    script.write_text("fit <- 2\n")
    assert _model_cache_key(*arguments) != first
    script.write_text("fit <- 1\n")
    data.write_text("x\n2\n")
    assert _model_cache_key(*arguments) != first
    data.write_text("x\n1\n")
    changed_parameters = (*arguments[:4], 0.90, *arguments[5:])
    assert _model_cache_key(*changed_parameters) != first


def test_graph_metadata_uses_rule_specific_inference_scope() -> None:
    population = pd.DataFrame({"survey_weight": [1.0, 2.0]})
    nodes = pd.DataFrame(
        {
            "condition": ["A", "B"],
            "n_valid": [1000, 1000],
            "n_cases": [100, 100],
            "weighted_prevalence": [0.1, 0.1],
        }
    )
    edges = pd.DataFrame(
        {
            "source_condition": ["A"],
            "target_condition": ["B"],
            "association": [0.2],
            "estimator": ["weighted_phi"],
            "inference_scope": ["descriptive"],
            "association_scale": ["weighted_phi"],
            "n_complete": [1000],
            "cooccurring_cases": [50],
            "edge_present": [True],
        }
    )
    graph = construct_graph(
        "g",
        2024,
        1,
        "Alabama",
        "lower",
        population,
        nodes,
        edges,
        {
            "estimator": {
                "name": "phase25_scenarios",
                "inference_scope": "descriptive_and_design_based",
            },
            "node_criteria": {"minimum_valid_n": 500, "minimum_cases": 30},
        },
    )
    assert graph.graph["inference_scope"] == "descriptive"
    assert graph.graph["association_scale"] == "weighted_phi"


def test_stability_interval_exposes_calibration_uncertainty() -> None:
    result = _add_stability_intervals(
        pd.DataFrame(
            {
                "bootstrap_replicates_valid": [200],
                "positive_stability": [0.9],
            }
        )
    )
    assert result.loc[0, "stability_wilson_low"] < 0.9
    assert result.loc[0, "stability_wilson_high"] > 0.9


def test_stable_edge_requires_all_requested_replicates() -> None:
    adjusted = pd.DataFrame(
        {
            "graph_id": ["g"],
            "source_condition": ["A"],
            "target_condition": ["B"],
            "ses_definition": ["education_binary"],
            "edge_present": [True],
        }
    )
    bootstrap = pd.DataFrame(
        {
            "graph_id": ["g"],
            "source_condition": ["A"],
            "target_condition": ["B"],
            "bootstrap_status": ["ok"],
            "bootstrap_message": [""],
            "bootstrap_log_odds_ratio": [0.5],
            "bootstrap_replicates_requested": [200],
            "bootstrap_replicates_valid": [199],
            "positive_stability": [1.0],
            "stability_standard_error": [0.0],
            "stability_wilson_low": [0.98],
            "stability_wilson_high": [1.0],
            "bootstrap_low_log": [0.2],
            "bootstrap_high_log": [0.8],
        }
    )
    stable = _bootstrap_stable_table(
        adjusted,
        bootstrap,
        "education_binary",
        0.9,
        "adjusted_stable",
    )
    assert not bool(stable.loc[0, "edge_present"])


def test_frozen_baseline_scientific_values() -> None:
    config = load_config()
    assert config.digest == "365b9697fd28"
    baseline = Path("results/baselines/phase2_365b9697fd28")
    statistics = pd.read_csv(baseline / "graph_statistics.csv")
    similarities = pd.read_csv(baseline / "structural_similarity.csv")
    viability = pd.read_json(baseline / "viability_report.json", typ="series")
    assert len(statistics) == 10
    assert statistics["edge_count"].min() == 21
    assert statistics["edge_count"].max() == 33
    assert len(similarities) == 100
    assert viability["unique_edge_sets"] == 10


def test_frozen_phase25_scientific_values() -> None:
    config = load_config(
        "configs/analysis_phase25.yaml",
        "configs/conditions.yaml",
        "configs/graph_phase25.yaml",
    )
    assert config.digest == "fff54b040741"
    baseline = Path("results/baselines/phase25_fff54b040741")
    manifest = json.loads((baseline / "manifest.json").read_text())
    viability = json.loads((baseline / "viability_report.json").read_text())
    statistics = pd.read_csv(baseline / "graph_statistics.csv")
    edges = pd.read_csv(baseline / "edge_table_all_rules.csv.gz", low_memory=False)
    similarities = pd.read_csv(baseline / "structural_similarity.csv.gz")
    bootstrap = pd.read_csv(baseline / "bootstrap_edge_stability.csv")

    assert manifest["config_digest"] == "fff54b040741"
    assert manifest["phase_boundary"] == "Stopped before frequent subgraph mining"
    assert viability["decision"] == "HOLD"
    assert viability["diagnostics"]["median_edge_count"] == 31.0
    assert np.isclose(viability["diagnostics"]["maximum_density"], 8 / 9)
    stable = statistics[
        (statistics["ses_definition"] == "education_binary")
        & (statistics["rule_id"] == "adjusted_stable")
    ]
    assert len(stable) == 94
    assert stable["edge_count"].median() == 31
    assert len(edges) == 29880
    assert len(similarities) == 62992
    off_diagonal = similarities["graph_id_a"].ne(similarities["graph_id_b"])
    assert similarities.loc[
        off_diagonal, "density_adjusted_jaccard_z"
    ].notna().all()
    assert len(bootstrap) == 2807
    assert bootstrap["bootstrap_replicates_valid"].eq(200).all()
    assert bootstrap["positive_stability"].ge(0.90).all()


@pytest.mark.skipif(
    not Path(".conda/r-survey/bin/Rscript").exists() and shutil.which("Rscript") is None,
    reason="R survey runtime is not installed",
)
def test_r_survey_point_model_smoke(tmp_path: Path) -> None:
    size = 240
    generator = np.random.default_rng(11)
    source = generator.binomial(1, 0.35, size)
    target = generator.binomial(1, 0.15 + 0.5 * source, size)
    data = pd.DataFrame(
        {
            "state_code": 1,
            "survey_weight": 1.0,
            "survey_strata": np.repeat(np.arange(1, 5), size // 4),
            "survey_psu": np.arange(size),
            "age_group": np.tile(np.arange(1, 7), size // 6),
            "sex": generator.integers(1, 3, size),
            "A": source,
            "B": target,
            "ses_education_binary": np.tile(["lower", "higher"], size // 2),
        }
    )
    data_path = tmp_path / "data.csv"
    data.to_csv(data_path, index=False)
    jobs = pd.DataFrame(
        [
            {
                "graph_id": "g",
                "state_code": 1,
                "ses_definition": "education_binary",
                "ses_category": "lower",
                "source_condition": "A",
                "target_condition": "B",
            }
        ]
    )
    result = run_survey_models(
        Path.cwd(),
        "point",
        data_path,
        jobs,
        tmp_path / "r",
        0.95,
        0,
        7,
        1,
    )
    assert result.loc[0, "model_status"] == "ok"
    assert np.isfinite(result.loc[0, "log_odds_ratio"])
