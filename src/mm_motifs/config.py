from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    analysis: dict[str, Any]
    conditions: dict[str, Any]
    graph: dict[str, Any]
    project_root: Path
    digest: str

    @property
    def year(self) -> int:
        return int(self.analysis["data"]["year"])

    @property
    def random_seed(self) -> int:
        return int(self.analysis["project"]["random_seed"])

    @property
    def active_ses(self) -> str:
        return str(self.analysis["socioeconomic"]["active_definition"])

    def resolve(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.project_root / path


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return value


def _config_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:12]


def _validate(analysis: dict[str, Any], conditions: dict[str, Any], graph: dict[str, Any]) -> None:
    year = str(analysis["data"]["year"])
    active_ses = analysis["socioeconomic"]["active_definition"]
    definitions = analysis["socioeconomic"]["definitions"]
    if active_ses not in definitions:
        raise ValueError(f"Unknown active SES definition: {active_ses}")
    if year not in definitions[active_ses]["source_by_year"]:
        raise ValueError(f"SES definition {active_ses} has no source for {year}")
    estimator_name = graph["estimator"]["name"]
    if estimator_name not in {
        "weighted_phi",
        "phase25_scenarios",
        "phase26_phi_stability",
        "phase3_gspan",
    }:
        raise ValueError(f"Unsupported estimator configuration: {estimator_name}")
    if estimator_name == "phase25_scenarios":
        phase25 = analysis.get("phase25")
        if not phase25:
            raise ValueError("Phase 2.5 analysis settings are required")
        unknown_ses = set(phase25["ses_definitions"]) - set(definitions)
        if unknown_ses:
            raise ValueError(f"Unknown Phase 2.5 SES definitions: {sorted(unknown_ses)}")
        if phase25["primary_ses_definition"] not in phase25["ses_definitions"]:
            raise ValueError("Phase 2.5 primary SES must be one of the SES definitions")
        if any(int(code) >= 60 for code in phase25["geography_codes"]):
            raise ValueError("Phase 2.5 primary geography cannot include territories")
        graph_phase25 = graph.get("phase25")
        if not graph_phase25:
            raise ValueError("Phase 2.5 graph settings are required")
        if int(graph_phase25["bootstrap"]["replicates"]) < 2:
            raise ValueError("At least two bootstrap replicates are required")
        if (
            graph_phase25["bootstrap"]["ses_definition"]
            != phase25["primary_ses_definition"]
        ):
            raise ValueError("Bootstrap SES definition must match the primary SES")
        if graph_phase25["bootstrap"]["type"] != "bootstrap":
            raise ValueError("Phase 2.5 supports bootstrap replicate weights only")
        if graph_phase25["bootstrap"]["mse"] is not True:
            raise ValueError("Phase 2.5 bootstrap requires mse: true")
        rule_ids = [
            *(rule["rule_id"] for rule in graph_phase25["phi_rules"]),
            graph_phase25["adjusted_rule"]["rule_id"],
            graph_phase25["bootstrap"]["rule_id"],
        ]
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("Phase 2.5 rule IDs must be unique")
    if estimator_name == "phase26_phi_stability":
        phase25 = analysis.get("phase25")
        phase26 = graph.get("phase26")
        if not phase25 or not phase26:
            raise ValueError("Phase 2.6 requires Phase 2.5 population settings")
        primary_ses = phase26["primary_ses_definition"]
        if primary_ses != phase25["primary_ses_definition"]:
            raise ValueError("Phase 2.6 SES must match the Phase 2.5 primary SES")
        thresholds = phase26["thresholds"]
        effects = [float(rule["minimum_effect"]) for rule in thresholds]
        if effects != sorted(effects) or effects != [0.10, 0.12, 0.14]:
            raise ValueError("Phase 2.6 thresholds must be 0.10, 0.12, and 0.14")
        rule_ids = [str(rule["rule_id"]) for rule in thresholds]
        if phase26["primary_rule_id"] not in rule_ids:
            raise ValueError("Phase 2.6 primary rule must be a threshold rule")
        if phase26["stable_rule_id"] in rule_ids:
            raise ValueError("Phase 2.6 stable rule ID must be unique")
        bootstrap = phase26["bootstrap"]
        if int(bootstrap["replicates"]) < 2:
            raise ValueError("At least two Phase 2.6 replicates are required")
        if bootstrap["type"] != "bootstrap" or bootstrap["mse"] is not True:
            raise ValueError("Phase 2.6 requires bootstrap replicate weights with mse")
        stability = float(bootstrap["selection_stability_threshold"])
        if not 0 < stability <= 1:
            raise ValueError("Phase 2.6 stability threshold must be in (0, 1]")
    if estimator_name == "phase3_gspan":
        phase25 = analysis.get("phase25")
        phase3 = graph.get("phase3")
        if not phase25 or not phase3:
            raise ValueError("Phase 3 requires all-state population settings")
        if (
            phase3["primary_ses_definition"]
            != phase25["primary_ses_definition"]
        ):
            raise ValueError("Phase 3 SES must match the Phase 2.6 primary SES")
        definition = phase3["graph_definition"]
        if (
            float(definition["point_phi_threshold"]) != 0.12
            or float(definition["bootstrap_selection_threshold"]) != 0.12
            or float(definition["minimum_selection_stability"]) != 0.90
        ):
            raise ValueError("Phase 3 must use the frozen stable 0.12 graph")
        mining = phase3["mining"]
        fractions = [
            float(value) for value in mining["support_fractions"]
        ]
        if fractions != [0.10, 0.20, 0.30]:
            raise ValueError("Phase 3 support levels must be 10%, 20%, 30%")
        if float(mining["primary_support_fraction"]) not in fractions:
            raise ValueError("Primary motif support must be in the spectrum")
        if (
            int(mining["minimum_nodes"]) != 3
            or int(mining["maximum_nodes"]) != 5
        ):
            raise ValueError("Phase 3 motifs must contain 3–5 nodes")
        if (
            mining["backend"] != "fast_gspan"
            or str(mining["backend_version"]) != "0.1.3"
            or mining["connected"] is not True
            or mining["induced"] is not False
            or mining["undirected"] is not True
        ):
            raise ValueError("Phase 3 requires undirected non-induced gSpan")
        bootstrap = phase3["bootstrap"]
        if (
            int(bootstrap["replicates"]) != 500
            or float(bootstrap["threshold"]) != 0.12
            or bootstrap["discovery_reruns"] is not True
            or bootstrap["raw_threshold_sensitivity_reruns"] is not True
        ):
            raise ValueError("Phase 3 requires 500 phi 0.12 realizations")
        density_null = phase3["density_null"]
        if (
            int(density_null["fixed_edge_replicates"]) < 2
            or int(density_null["degree_preserving_replicates"]) < 2
        ):
            raise ValueError("Phase 3 density nulls require replication")
        if int(phase3["ses_permutation"]["permutations"]) < 2:
            raise ValueError("Phase 3 SES permutation requires replication")

    names: set[str] = set()
    required_years = {
        str(value)
        for value in conditions.get("selection_policy", {}).get("required_years", [])
    }
    for condition in conditions["conditions"]:
        name = condition["canonical_name"]
        if name in names:
            raise ValueError(f"Duplicate condition name: {name}")
        names.add(name)
        if condition.get("primary", True) and year not in condition["source_by_year"]:
            raise ValueError(f"Condition {name} has no source for {year}")
        missing_years = required_years - set(condition["source_by_year"])
        if missing_years:
            raise ValueError(
                f"Condition {name} has no source for required years {sorted(missing_years)}"
            )
        coding = condition["coding"]
        if coding["type"] != "binary":
            raise ValueError(f"Unsupported coding type for {name}: {coding['type']}")
        positive = set(coding["positive_values"])
        negative = set(coding["negative_values"])
        if positive & negative:
            raise ValueError(f"Overlapping positive and negative codes for {name}")

    states = analysis["prototype"]["states"]
    if not states:
        raise ValueError("At least one prototype state is required")
    if len(names) < 2:
        raise ValueError("At least two conditions are required")


def load_config(
    analysis_path: str | Path = "configs/analysis.yaml",
    conditions_path: str | Path = "configs/conditions.yaml",
    graph_path: str | Path = "configs/graph.yaml",
    year: int | None = None,
) -> ProjectConfig:
    analysis_file = Path(analysis_path).resolve()
    conditions_file = Path(conditions_path).resolve()
    graph_file = Path(graph_path).resolve()
    analysis = copy.deepcopy(_read_yaml(analysis_file))
    conditions = _read_yaml(conditions_file)
    graph = _read_yaml(graph_file)
    if year is not None:
        analysis["data"]["year"] = int(year)
    _validate(analysis, conditions, graph)
    payload = {"analysis": analysis, "conditions": conditions, "graph": graph}
    return ProjectConfig(
        analysis=analysis,
        conditions=conditions,
        graph=graph,
        project_root=analysis_file.parent.parent,
        digest=_config_digest(payload),
    )
