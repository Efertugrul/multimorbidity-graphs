from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from mm_motifs.config import ProjectConfig
from mm_motifs.features.conditions import condition_sources, derive_conditions
from mm_motifs.features.populations import primary_geography_codes, state_name
from mm_motifs.features.socioeconomic import derive_ses, ses_source


def phase25_selected_columns(config: ProjectConfig) -> list[str]:
    design = config.analysis["survey_design"]
    columns = list(design.values())
    columns.extend(condition_sources(config.conditions, config.year))
    for definition_name in config.analysis["phase25"]["ses_definitions"]:
        columns.append(ses_source(config.analysis, definition_name, config.year))
    return list(dict.fromkeys(columns))


def harmonize_phase25(
    raw: pd.DataFrame,
    config: ProjectConfig,
    geography_codes: Iterable[int] | None = None,
) -> pd.DataFrame:
    design = config.analysis["survey_design"]
    codes = set(geography_codes or primary_geography_codes())
    state_values = pd.to_numeric(raw[design["state"]], errors="coerce")
    selected = raw.loc[state_values.isin(codes)].copy()
    state_values = pd.to_numeric(selected[design["state"]], errors="coerce").astype(
        "Int64"
    )
    conditions = derive_conditions(selected, config.conditions, config.year)
    frame = pd.DataFrame(index=selected.index)
    frame["state_code"] = state_values
    frame["geography"] = state_values.map(
        lambda value: state_name(int(value)) if pd.notna(value) else pd.NA
    )
    frame["survey_weight"] = pd.to_numeric(
        selected[design["weight"]], errors="coerce"
    )
    frame["survey_strata"] = pd.to_numeric(
        selected[design["strata"]], errors="coerce"
    )
    frame["survey_psu"] = pd.to_numeric(selected[design["psu"]], errors="coerce")
    frame["age_group"] = pd.to_numeric(
        selected[design["age_group"]], errors="coerce"
    )
    frame["sex"] = pd.to_numeric(selected[design["sex"]], errors="coerce")
    for name in conditions:
        frame[name] = conditions[name]
    for definition_name in config.analysis["phase25"]["ses_definitions"]:
        frame[f"ses_{definition_name}"] = derive_ses(
            selected,
            config.analysis,
            config.year,
            definition_name,
        )
    return frame.reset_index(drop=True)
