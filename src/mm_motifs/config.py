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
    if graph["estimator"]["name"] != "weighted_phi":
        raise ValueError("Phase 2 supports only the weighted_phi estimator")

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
