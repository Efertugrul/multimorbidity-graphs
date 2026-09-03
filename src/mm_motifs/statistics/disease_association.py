from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np
import pandas as pd

from mm_motifs.statistics.multiple_testing import benjamini_hochberg
from mm_motifs.statistics.survey import valid_weights


def weighted_phi(
    source: pd.Series,
    target: pd.Series,
    weights: pd.Series,
) -> dict[str, Any]:
    x = pd.to_numeric(source, errors="coerce")
    y = pd.to_numeric(target, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")
    valid = x.notna() & y.notna() & valid_weights(w)
    x = x.loc[valid].astype(float)
    y = y.loc[valid].astype(float)
    w = w.loc[valid].astype(float)
    total_weight = float(w.sum())
    if total_weight <= 0:
        association = math.nan
        source_prevalence = math.nan
        target_prevalence = math.nan
        joint_prevalence = math.nan
        effective_n = math.nan
    else:
        source_prevalence = float(np.average(x, weights=w))
        target_prevalence = float(np.average(y, weights=w))
        joint_prevalence = float(np.average(x * y, weights=w))
        denominator = math.sqrt(
            source_prevalence
            * (1 - source_prevalence)
            * target_prevalence
            * (1 - target_prevalence)
        )
        association = (
            (joint_prevalence - source_prevalence * target_prevalence) / denominator
            if denominator > 0
            else math.nan
        )
        squared_weight_sum = float(np.square(w).sum())
        effective_n = total_weight**2 / squared_weight_sum if squared_weight_sum > 0 else math.nan

    return {
        "association": association,
        "standard_error": math.nan,
        "p_value": math.nan,
        "n_complete": int(valid.sum()),
        "source_cases": int((x == 1).sum()),
        "target_cases": int((y == 1).sum()),
        "cooccurring_cases": int(((x == 1) & (y == 1)).sum()),
        "weighted_n": total_weight,
        "kish_effective_n": effective_n,
        "weighted_source_prevalence": source_prevalence,
        "weighted_target_prevalence": target_prevalence,
        "weighted_joint_prevalence": joint_prevalence,
    }


def estimate_edge_table(
    frame: pd.DataFrame,
    condition_names: list[str],
    graph_id: str,
    criteria: dict[str, Any],
) -> pd.DataFrame:
    rows = []
    for source_name, target_name in itertools.combinations(condition_names, 2):
        estimate = weighted_phi(
            frame[source_name],
            frame[target_name],
            frame["survey_weight"],
        )
        reasons = []
        if estimate["n_complete"] < int(criteria["minimum_complete_n"]):
            reasons.append("complete_n")
        if estimate["source_cases"] < int(criteria["minimum_condition_cases"]):
            reasons.append("source_cases")
        if estimate["target_cases"] < int(criteria["minimum_condition_cases"]):
            reasons.append("target_cases")
        if estimate["cooccurring_cases"] < int(criteria["minimum_cooccurring_cases"]):
            reasons.append("cooccurring_cases")
        if not np.isfinite(estimate["association"]):
            reasons.append("undefined_association")
        rows.append(
            {
                "graph_id": graph_id,
                "source_condition": source_name,
                "target_condition": target_name,
                **estimate,
                "estimator": "weighted_phi",
                "inference_scope": "descriptive",
                "pair_eligible": not reasons,
                "exclusion_reason": "|".join(reasons),
            }
        )

    result = pd.DataFrame(rows)
    result["q_value"] = benjamini_hochberg(result["p_value"])
    effect_pass = result["association"] >= float(criteria["minimum_effect"])
    fdr_pass = (
        result["q_value"] <= float(criteria["maximum_q_value"])
        if criteria["require_fdr"]
        else pd.Series(True, index=result.index)
    )
    result["edge_present"] = result["pair_eligible"] & effect_pass & fdr_pass
    return result
