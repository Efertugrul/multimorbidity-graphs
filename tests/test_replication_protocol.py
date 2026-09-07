from __future__ import annotations

import hashlib
import json
import shutil
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from mm_motifs.cli import build_parser
from mm_motifs.config import load_config
from mm_motifs.replication import (
    evaluate_confirmatory_replication,
    evaluate_methodological_vocabulary,
    holm_adjust,
    summarize_occurrence_families,
    validate_replication_protocol,
)
from mm_motifs.replication.confirmatory import (
    directional_sign_flip_test,
    rademacher_sign_bank,
)
from mm_motifs.motifs.robustness import (
    fixed_edge_occurrence_probability,
)
from mm_motifs.statistics.phi_bootstrap import (
    RESULT_COLUMNS,
    _valid_results,
)
from mm_motifs.workflows import phase4 as phase4_workflow
from mm_motifs.workflows.phase4 import (
    _complete_bootstrap_graphs,
    _phase4_bootstrap_jobs,
    _paired_usable_registry,
    _validate_source_input,
)


PROTOCOL = Path("protocols/phase4_2023_replication")


def test_replication_protocol_is_frozen_and_valid() -> None:
    result = validate_replication_protocol(PROTOCOL)
    assert result == {
        "protocol_id": "phase4_2023_cc5bbcc08283",
        "protocol_sha256": (
            "cc5bbcc0828396fe028317e9901f76cf5ab36dc12fff7d7a429333485245ccff"
        ),
        "methodological_motif_count": 484,
        "occurrence_family_count": 299,
        "confirmatory_hypothesis_count": 3,
        "target_jurisdiction_count": 51,
        "discovery_jurisdiction_count": 47,
    }


def test_replication_never_reruns_2023_motif_discovery() -> None:
    protocol = yaml.safe_load((PROTOCOL / "protocol.yaml").read_text())
    assert protocol["motifs"]["rerun_gspan_in_2023"] is False
    assert protocol["motifs"]["permit_2023_motif_discovery"] is False
    replication = protocol["methodological_replication"]
    assert replication["frozen_motif_count"] == 484
    assert (
        replication["primary_2023_reproduction"][
            "no_2023_screen_before_evaluation"
        ]
        is True
    )


def test_phase4_target_configuration_is_executable_and_2023_only() -> None:
    config = load_config(
        "configs/analysis_phase4.yaml",
        "configs/conditions.yaml",
        "configs/graph_phase4.yaml",
    )
    assert config.year == 2023
    assert set(config.analysis["socioeconomic"]["definitions"]) == {
        "education_binary"
    }
    assert config.graph["estimator"]["name"] == (
        "phase4_confirmatory_replication"
    )
    assert config.graph["phase4"]["motif_discovery"] == {
        "rerun_gspan": False,
        "permit_new_motifs": False,
    }
    command_action = next(
        action
        for action in build_parser()._actions
        if action.dest == "command"
    )
    phase4_parser = command_action.choices["phase4"]
    option_strings = {
        option
        for action in phase4_parser._actions
        for option in action.option_strings
    }
    assert option_strings == {"-h", "--help", "--verbose", "--workers"}


def test_confirmatory_family_has_three_frozen_directions() -> None:
    hypotheses = pd.read_csv(PROTOCOL / "confirmatory_hypotheses.csv")
    assert hypotheses[
        ["hypothesis_id", "motif_id", "expected_enrichment", "direction_sign"]
    ].to_dict(orient="records") == [
        {
            "hypothesis_id": "H1",
            "motif_id": "motif_a79cab0e1b2c2fbf",
            "expected_enrichment": "higher",
            "direction_sign": -1,
        },
        {
            "hypothesis_id": "H2",
            "motif_id": "motif_eeb4d747c66beaa4",
            "expected_enrichment": "higher",
            "direction_sign": -1,
        },
        {
            "hypothesis_id": "H3",
            "motif_id": "motif_07019d963bc595b3",
            "expected_enrichment": "lower",
            "direction_sign": 1,
        },
    ]
    protocol = yaml.safe_load((PROTOCOL / "protocol.yaml").read_text())
    confirmatory = protocol["confirmatory_replication"]
    assert confirmatory["alternatives"] == "one_sided_frozen_direction"
    assert confirmatory["permutations"] == 99999
    assert {
        key: confirmatory["multiplicity"][key]
        for key in ["method", "family_size", "alpha"]
    } == {
        "method": "holm",
        "family_size": 3,
        "alpha": 0.05,
    }


def test_graph_and_density_rules_are_exactly_frozen() -> None:
    protocol = yaml.safe_load((PROTOCOL / "protocol.yaml").read_text())
    assert protocol["edges"]["point_phi_threshold"] == 0.12
    assert protocol["edges"]["minimum_bootstrap_selection_probability"] == 0.90
    assert protocol["edge_bootstrap"]["replicates"] == 500
    assert protocol["population"]["minimum_unweighted_population_n"] == 1000
    assert protocol["density_standardization"]["residual"] == (
        "observed_minus_expected"
    )
    assert protocol["ses"]["definition"] == "education_binary"
    assert protocol["ses"]["income_analysis"] == "prohibited"


def test_protocol_checksum_rejects_in_place_amendment(
    tmp_path: Path,
) -> None:
    copied = tmp_path / "protocol"
    shutil.copytree(PROTOCOL, copied)
    with (copied / "confirmatory_hypotheses.csv").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_replication_protocol(copied)


def test_holm_adjustment_is_fixed_to_three_hypotheses() -> None:
    adjusted = holm_adjust(np.asarray([0.01, 0.04, 0.03]))
    assert np.allclose(adjusted, [0.03, 0.06, 0.06])


def test_confirmatory_runner_uses_primary_decisions_only() -> None:
    state_codes = np.arange(1, 41)
    graph_ids = []
    graph_masks = []
    registry_rows = []
    for state_code in state_codes:
        for category, mask in [("lower", 0b110), ("higher", 0b011)]:
            graph_id = f"{state_code}_{category}"
            graph_ids.append(graph_id)
            graph_masks.append(mask)
            registry_rows.append(
                {
                    "graph_id": graph_id,
                    "geography_code": state_code,
                    "ses_category": category,
                }
            )
    hypotheses = pd.DataFrame(
        {
            "family_order": [1, 2, 3],
            "hypothesis_id": ["H1", "H2", "H3"],
            "motif_id": ["m1", "m2", "m3"],
            "expected_enrichment": ["higher", "higher", "lower"],
            "direction_sign": [-1, -1, 1],
            "edge_count": [2, 2, 2],
            "edge_list": [
                '[["A","B"],["A","C"]]',
                '[["A","B"],["B","C"]]',
                '[["A","C"],["B","C"]]',
            ],
        }
    )
    result = evaluate_confirmatory_replication(
        graph_ids,
        np.asarray(graph_masks, dtype=np.uint64),
        np.full(len(graph_ids), 0b111, dtype=np.uint64),
        hypotheses,
        pd.DataFrame(registry_rows),
        {("A", "B"): 0, ("A", "C"): 1, ("B", "C"): 2},
        set(state_codes[:-1]),
        9999,
        20230904,
        40,
        0.05,
    )
    primary = result[result["analysis_set"].eq("primary_2023_eligible")]
    sensitivity = result[
        result["analysis_set"].eq("matched_2024_jurisdictions")
    ]
    assert primary["state_pair_count"].eq(40).all()
    assert primary["decision_eligible"].all()
    assert primary.set_index("hypothesis_id").loc["H1", "replicated"]
    assert primary.set_index("hypothesis_id").loc["H3", "replicated"]
    assert sensitivity["state_pair_count"].eq(39).all()
    assert sensitivity["one_sided_p_value"].eq(1.0).all()
    assert not sensitivity["decision_eligible"].any()
    assert not sensitivity["replicated"].any()


def test_methodological_runner_retains_all_frozen_motifs() -> None:
    graph_ids = ["1_lower", "1_higher", "2_lower", "2_higher"]
    graph_masks = np.asarray([0b011, 0b001, 0b011, 0b000], dtype=np.uint64)
    opportunity_masks = np.asarray(
        [0b111, 0b111, 0b111, 0b001],
        dtype=np.uint64,
    )
    vocabulary = pd.DataFrame(
        {
            "motif_id": ["m1", "m2"],
            "edge_count": [1, 2],
            "canonical_edges": ['[["A","B"]]', '[["A","B"],["A","C"]]'],
            "occurrence_class_id": ["c1", "c1"],
            "discovery_support_fraction": [0.75, 0.50],
        }
    )
    registry = pd.DataFrame(
        {
            "graph_id": graph_ids,
            "geography_code": [1, 1, 2, 2],
            "ses_category": ["lower", "higher", "lower", "higher"],
        }
    )
    result, summary = evaluate_methodological_vocabulary(
        graph_ids,
        graph_masks,
        opportunity_masks,
        vocabulary,
        registry,
        {("A", "B"): 0, ("A", "C"): 1, ("B", "C"): 2},
        0.20,
        1,
    )
    assert result["motif_id"].tolist() == ["m1", "m2"]
    assert result["replication_paired_state_count"].tolist() == [2, 1]
    assert result["replication_support_count"].tolist() == [3, 1]
    assert result["replicated_at_20pct"].tolist() == [True, True]
    assert summary["frozen_motif_count"] == 2
    assert summary["replicated_motif_count"] == 2
    family = summarize_occurrence_families(
        result,
        pd.DataFrame(
            {
                "occurrence_class_id": ["c1"],
                "representative_motif_id": ["m1"],
            }
        ),
    )
    assert family.loc[0, "replication_member_count"] == 2
    assert family.loc[0, "representative_replicated_at_20pct"]


def test_bootstrap_failure_excludes_entire_state_pair() -> None:
    graph_ids = [
        "1_lower",
        "1_higher",
        "2_lower",
        "2_higher",
    ]
    bootstrap = pd.DataFrame(
        {
            "graph_id": graph_ids,
            "pair_eligible": [True] * 4,
            "bootstrap_status": ["ok", "ok", "ok", "partial"],
            "bootstrap_replicates_requested": [500] * 4,
            "bootstrap_replicates_valid": [500, 500, 500, 499],
        }
    )
    quality = _complete_bootstrap_graphs(bootstrap, 500)
    registry = pd.DataFrame(
        {
            "graph_id": graph_ids,
            "geography_code": [1, 1, 2, 2],
            "ses_category": ["lower", "higher", "lower", "higher"],
            "eligibility_status": ["eligible"] * 4,
        }
    )
    usable = _paired_usable_registry(registry, quality)
    assert usable["graph_id"].tolist() == ["1_lower", "1_higher"]


def test_phase4_bootstraps_only_point_eligible_dyads() -> None:
    pairs = pd.DataFrame(
        {
            "graph_id": ["1_lower", "1_lower"],
            "source_condition": ["A", "A"],
            "target_condition": ["B", "C"],
            "association": [0.2, 0.1],
            "pair_eligible": [True, False],
        }
    )
    registry = pd.DataFrame(
        {
            "graph_id": ["1_lower"],
            "geography_code": [1],
            "ses_definition": ["education_binary"],
            "ses_category": ["lower"],
        }
    )
    jobs = _phase4_bootstrap_jobs(pairs, registry)
    assert jobs[["disease_a", "disease_b"]].values.tolist() == [["A", "B"]]


def test_zero_valid_bootstrap_failure_rows_are_preserved() -> None:
    jobs = pd.DataFrame(
        {
            "graph_id": ["1_lower", "2_lower"],
            "disease_a": ["A", "A"],
            "disease_b": ["B", "B"],
        }
    )
    rows = []
    for graph_id, status in [
        ("1_lower", "partial"),
        ("2_lower", "state_error"),
    ]:
        row = {column: np.nan for column in RESULT_COLUMNS}
        row.update(
            {
                "graph_id": graph_id,
                "disease_a": "A",
                "disease_b": "B",
                "bootstrap_selection_mask": "bxxx",
                "bootstrap_replicates_requested": 3,
                "bootstrap_replicates_valid": 0,
                "bootstrap_status": status,
            }
        )
        rows.append(row)
    assert _valid_results(pd.DataFrame(rows), jobs, 3)
    rows[0]["bootstrap_status"] = "ok"
    assert not _valid_results(pd.DataFrame(rows), jobs, 3)


def test_exact_sign_draw_ties_and_zero_variance_rules() -> None:
    states, generated = rademacher_sign_bank(
        np.asarray([2, 1]),
        4,
        20230904,
    )
    assert states.tolist() == [1, 2]
    assert generated.tolist() == [
        [1, -1],
        [-1, -1],
        [1, -1],
        [-1, 1],
    ]
    differences = pd.DataFrame(
        {
            "H1": np.zeros(4),
            "H2": np.ones(4),
            "H3": -np.ones(4),
        },
        index=[1, 2, 3, 4],
    )
    hypotheses = pd.DataFrame(
        {
            "family_order": [1, 2, 3],
            "hypothesis_id": ["H1", "H2", "H3"],
            "motif_id": ["m1", "m2", "m3"],
            "expected_enrichment": ["lower", "lower", "higher"],
            "direction_sign": [1, 1, -1],
        }
    )
    signs = np.asarray(list(product([-1, 1], repeat=4)), dtype=np.int8)
    result = directional_sign_flip_test(
        differences,
        hypotheses,
        signs,
        4,
        0.05,
        True,
    ).set_index("hypothesis_id")
    assert result.loc["H1", "studentized_statistic"] == 0
    assert result.loc["H1", "one_sided_p_value"] == 1
    assert np.isposinf(result.loc["H2", "studentized_statistic"])
    assert result.loc["H2", "one_sided_p_value"] == pytest.approx(2 / 17)
    assert np.isneginf(result.loc["H3", "studentized_statistic"])
    assert result.loc["H3", "one_sided_p_value"] == pytest.approx(2 / 17)


def test_density_boundary_rules_are_executable() -> None:
    assert fixed_edge_occurrence_probability(10, 2, 3) == 0
    with pytest.raises(ValueError, match="exceeds eligible"):
        fixed_edge_occurrence_probability(2, 3, 1)


def test_phase4_input_provenance_is_enforced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(
        "configs/analysis_phase4.yaml",
        "configs/conditions.yaml",
        "configs/graph_phase4.yaml",
    )
    source = tmp_path / "LLCP2023.XPT"
    source.write_bytes(b"frozen-input")
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    metadata = {
        "year": 2023,
        "source_url": config.analysis["data"]["urls"]["2023"],
        "archive_sha256": "a" * 64,
        "xpt_sha256": source_sha256,
        "xpt_bytes": source.stat().st_size,
    }
    source.with_suffix(".download.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    required = phase4_workflow.phase25_selected_columns(config)
    monkeypatch.setattr(
        phase4_workflow,
        "xport_dimensions",
        lambda _: (433323, required),
    )
    observed_sha256, provenance = _validate_source_input(config, source)
    assert observed_sha256 == source_sha256
    assert provenance["xpt_row_count"] == 433323
    monkeypatch.setattr(
        phase4_workflow,
        "xport_dimensions",
        lambda _: (1, required),
    )
    with pytest.raises(ValueError, match="Expected 433323"):
        _validate_source_input(config, source)
