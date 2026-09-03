from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

import pandas as pd


def primary_conditions(registry: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in registry["conditions"] if item.get("primary", True)]


def condition_names(registry: dict[str, Any]) -> list[str]:
    return [item["canonical_name"] for item in primary_conditions(registry)]


def condition_sources(registry: dict[str, Any], year: int) -> list[str]:
    year_key = str(year)
    sources: list[str] = []
    for item in primary_conditions(registry):
        source = item["source_by_year"][year_key]
        values = source if isinstance(source, list) else [source]
        sources.extend(values)
    return list(dict.fromkeys(sources))


def _binary_indicator(series: pd.Series, coding: dict[str, Any]) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    result = pd.Series(pd.NA, index=series.index, dtype="Float64")
    result.loc[numeric.isin(coding["positive_values"])] = 1.0
    result.loc[numeric.isin(coding["negative_values"])] = 0.0
    return result


def derive_conditions(
    frame: pd.DataFrame,
    registry: dict[str, Any],
    year: int,
) -> pd.DataFrame:
    derived: dict[str, pd.Series] = {}
    year_key = str(year)
    for item in primary_conditions(registry):
        source = item["source_by_year"][year_key]
        if isinstance(source, list):
            raise ValueError(f"Multi-variable coding is not implemented for {item['canonical_name']}")
        if source not in frame.columns:
            raise KeyError(f"Missing source variable {source} for {item['canonical_name']}")
        derived[item["canonical_name"]] = _binary_indicator(frame[source], item["coding"])
    return pd.DataFrame(derived, index=frame.index)


def condition_registry_table(
    registry: dict[str, Any],
    year: int,
    available_variables: Iterable[str],
    labels: dict[str, str | None],
) -> pd.DataFrame:
    available = set(available_variables)
    rows: list[dict[str, Any]] = []
    year_key = str(year)
    selection_policy = json.dumps(registry.get("selection_policy", {}), sort_keys=True)
    for item in registry["conditions"]:
        source = item["source_by_year"].get(year_key)
        sources = source if isinstance(source, list) else [source] if source else []
        present = bool(sources) and all(value in available for value in sources)
        rows.append(
            {
                "canonical_condition_name": item["canonical_name"],
                "condition_label": item["label"],
                "year": year,
                "source_variables": "|".join(sources),
                "source_labels": "|".join(str(labels.get(value) or "") for value in sources),
                "coding_rules": json.dumps(item["coding"], sort_keys=True),
                "selection_policy": selection_policy,
                "primary": bool(item.get("primary", True)),
                "source_available": present,
                "audit_status": "candidate" if present else "missing source",
            }
        )
    return pd.DataFrame(rows)
