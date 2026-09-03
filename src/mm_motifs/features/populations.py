from __future__ import annotations

from typing import Any

import pandas as pd


STATE_NAMES = {
    1: "Alabama",
    2: "Alaska",
    4: "Arizona",
    5: "Arkansas",
    6: "California",
    8: "Colorado",
    9: "Connecticut",
    10: "Delaware",
    11: "District of Columbia",
    12: "Florida",
    13: "Georgia",
    15: "Hawaii",
    16: "Idaho",
    17: "Illinois",
    18: "Indiana",
    19: "Iowa",
    20: "Kansas",
    21: "Kentucky",
    22: "Louisiana",
    23: "Maine",
    24: "Maryland",
    25: "Massachusetts",
    26: "Michigan",
    27: "Minnesota",
    28: "Mississippi",
    29: "Missouri",
    30: "Montana",
    31: "Nebraska",
    32: "Nevada",
    33: "New Hampshire",
    34: "New Jersey",
    35: "New Mexico",
    36: "New York",
    37: "North Carolina",
    38: "North Dakota",
    39: "Ohio",
    40: "Oklahoma",
    41: "Oregon",
    42: "Pennsylvania",
    44: "Rhode Island",
    45: "South Carolina",
    46: "South Dakota",
    47: "Tennessee",
    48: "Texas",
    49: "Utah",
    50: "Vermont",
    51: "Virginia",
    53: "Washington",
    54: "West Virginia",
    55: "Wisconsin",
    56: "Wyoming",
    66: "Guam",
    72: "Puerto Rico",
    78: "U.S. Virgin Islands",
}


def state_name(code: int | float) -> str:
    normalized = int(code)
    return STATE_NAMES.get(normalized, f"Unknown ({normalized:02d})")


def make_graph_id(year: int, state_code: int, ses_category: str) -> str:
    return f"{year}_{state_code:02d}_{ses_category}"


def primary_geography_codes() -> list[int]:
    return sorted(code for code in STATE_NAMES if code < 60)


def make_phase25_graph_id(
    year: int,
    state_code: int,
    ses_definition: str,
    ses_category: str,
) -> str:
    return f"{year}_{state_code:02d}_{ses_definition}_{ses_category}"


def phase25_population_registry(
    frame: pd.DataFrame,
    year: int,
    ses_definition: str,
    minimum_n: int,
    geography_codes: list[int] | None = None,
) -> pd.DataFrame:
    codes = geography_codes or primary_geography_codes()
    ses_column = f"ses_{ses_definition}"
    if ses_column not in frame:
        raise KeyError(f"Missing harmonized SES column: {ses_column}")
    reported_codes = set(
        pd.to_numeric(frame["state_code"], errors="coerce").dropna().astype(int)
    )
    rows = []
    for state_code in codes:
        state_frame = frame[frame["state_code"] == state_code]
        for category in ("lower", "higher"):
            group = state_frame[state_frame[ses_column] == category]
            weights = pd.to_numeric(group["survey_weight"], errors="coerce")
            valid_weight = weights.notna() & (weights > 0)
            valid_weights = weights.loc[valid_weight]
            weight_sum = float(valid_weights.sum())
            weight_squared_sum = float((valid_weights**2).sum())
            n = int(len(group))
            if state_code not in reported_codes:
                status = "no reporting data"
            elif n < minimum_n:
                status = "insufficient sample"
            else:
                status = "eligible"
            rows.append(
                {
                    "graph_id": make_phase25_graph_id(
                        year,
                        state_code,
                        ses_definition,
                        category,
                    ),
                    "year": year,
                    "geography_code": state_code,
                    "geography": state_name(state_code),
                    "ses_definition": ses_definition,
                    "ses_category": category,
                    "n_unweighted": n,
                    "weighted_population_estimate": weight_sum,
                    "kish_effective_n": (
                        weight_sum**2 / weight_squared_sum
                        if weight_squared_sum > 0
                        else float("nan")
                    ),
                    "n_strata": int(group["survey_strata"].nunique(dropna=True)),
                    "n_psu": int(group["survey_psu"].nunique(dropna=True)),
                    "eligibility_status": status,
                }
            )
    registry = pd.DataFrame(rows)
    registry["stratum_eligibility_status"] = registry["eligibility_status"]
    for state_code, indices in registry.groupby("geography_code").groups.items():
        statuses = registry.loc[indices, "stratum_eligibility_status"]
        if statuses.eq("eligible").all():
            continue
        eligible_indices = statuses[statuses == "eligible"].index
        registry.loc[eligible_indices, "eligibility_status"] = (
            "paired SES stratum ineligible"
        )
    return registry


def population_registry(
    frame: pd.DataFrame,
    analysis: dict[str, Any],
    year: int,
) -> pd.DataFrame:
    minimum_n = int(analysis["prototype"]["minimum_population_n"])
    states = [int(value) for value in analysis["prototype"]["states"]]
    categories = list(
        analysis["socioeconomic"]["definitions"][
            analysis["socioeconomic"]["active_definition"]
        ]["groups"]
    )
    rows = []
    for state_code in states:
        for category in categories:
            group = frame[
                (frame["state_code"] == state_code)
                & (frame["ses_category"] == category)
            ]
            weights = pd.to_numeric(group["survey_weight"], errors="coerce")
            valid_weight = weights.notna() & (weights > 0)
            valid_weights = weights.loc[valid_weight]
            weight_sum = float(valid_weights.sum())
            weight_squared_sum = float((valid_weights**2).sum())
            n = int(len(group))
            rows.append(
                {
                    "graph_id": make_graph_id(year, state_code, category),
                    "year": year,
                    "geography_code": state_code,
                    "geography": state_name(state_code),
                    "ses_category": category,
                    "n_unweighted": n,
                    "weighted_population_estimate": weight_sum,
                    "kish_effective_n": (
                        weight_sum**2 / weight_squared_sum
                        if weight_squared_sum > 0
                        else float("nan")
                    ),
                    "n_strata": int(group["survey_strata"].nunique(dropna=True)),
                    "n_psu": int(group["survey_psu"].nunique(dropna=True)),
                    "eligibility_status": "eligible" if n >= minimum_n else "insufficient sample",
                }
            )
    return pd.DataFrame(rows)
