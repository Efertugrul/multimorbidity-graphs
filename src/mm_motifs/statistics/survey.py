from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def valid_weights(weights: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(weights, errors="coerce")
    return numeric.notna() & np.isfinite(numeric) & (numeric > 0)


def weighted_prevalence(values: pd.Series, weights: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce")
    weight_values = pd.to_numeric(weights, errors="coerce")
    valid = numeric.notna() & valid_weights(weight_values)
    cases = valid & (numeric == 1)
    valid_weight = weight_values.loc[valid]
    denominator = float(valid_weight.sum())
    numerator = float(weight_values.loc[cases].sum())
    effective_n = (
        denominator**2 / float(np.square(valid_weight).sum())
        if denominator > 0 and float(np.square(valid_weight).sum()) > 0
        else math.nan
    )
    return {
        "n_valid": int(valid.sum()),
        "n_cases": int(cases.sum()),
        "unweighted_prevalence": float(numeric.loc[valid].mean()) if valid.any() else math.nan,
        "weighted_prevalence": numerator / denominator if denominator > 0 else math.nan,
        "weighted_population_denominator": denominator,
        "kish_effective_n": effective_n,
    }
