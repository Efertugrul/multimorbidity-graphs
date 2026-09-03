from __future__ import annotations

import json
from typing import Any

import pandas as pd


def ses_source(analysis: dict[str, Any], definition_name: str, year: int) -> str:
    definition = analysis["socioeconomic"]["definitions"][definition_name]
    return str(definition["source_by_year"][str(year)])


def derive_ses(
    frame: pd.DataFrame,
    analysis: dict[str, Any],
    year: int,
    definition_name: str | None = None,
) -> pd.Series:
    name = definition_name or analysis["socioeconomic"]["active_definition"]
    definition = analysis["socioeconomic"]["definitions"][name]
    source = ses_source(analysis, name, year)
    if source not in frame.columns:
        raise KeyError(f"Missing SES source variable: {source}")
    values = pd.to_numeric(frame[source], errors="coerce")
    result = pd.Series(pd.NA, index=frame.index, dtype="string", name="ses_category")
    for category, codes in definition["groups"].items():
        result.loc[values.isin(codes)] = category
    return result


def ses_registry_table(
    analysis: dict[str, Any],
    year: int,
    labels: dict[str, str | None],
    available_variables: set[str],
) -> pd.DataFrame:
    active = analysis["socioeconomic"]["active_definition"]
    rows = []
    for name, definition in analysis["socioeconomic"]["definitions"].items():
        source = ses_source(analysis, name, year)
        rows.append(
            {
                "definition_name": name,
                "definition_label": definition["label"],
                "active": name == active,
                "year": year,
                "source_variable": source,
                "source_label": labels.get(source),
                "groups": json.dumps(definition["groups"], sort_keys=True),
                "missing_values": json.dumps(definition.get("missing_values", [])),
                "source_available": source in available_variables,
            }
        )
    return pd.DataFrame(rows)
