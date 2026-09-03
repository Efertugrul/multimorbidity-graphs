from __future__ import annotations

import numpy as np
import pandas as pd

from mm_motifs.statistics.disease_association import (
    estimate_edge_table,
    weighted_phi,
)
from mm_motifs.statistics.multiple_testing import benjamini_hochberg


def test_weighted_phi_recovers_perfect_association() -> None:
    values = pd.Series([0, 0, 1, 1])
    estimate = weighted_phi(values, values, pd.Series([1, 2, 3, 4]))
    assert np.isclose(estimate["association"], 1.0)
    assert estimate["n_complete"] == 4
    assert estimate["cooccurring_cases"] == 2


def test_edge_threshold_and_case_criteria() -> None:
    frame = pd.DataFrame(
        {
            "A": [0, 0, 1, 1] * 20,
            "B": [0, 0, 1, 1] * 20,
            "survey_weight": 1.0,
        }
    )
    criteria = {
        "minimum_effect": 0.5,
        "minimum_complete_n": 50,
        "minimum_condition_cases": 20,
        "minimum_cooccurring_cases": 20,
        "require_fdr": False,
        "maximum_q_value": 0.05,
    }
    table = estimate_edge_table(frame, ["A", "B"], "graph", criteria)
    assert bool(table.loc[0, "pair_eligible"])
    assert bool(table.loc[0, "edge_present"])
    assert pd.isna(table.loc[0, "p_value"])


def test_benjamini_hochberg_preserves_missing_values() -> None:
    adjusted = benjamini_hochberg(pd.Series([0.01, 0.04, 0.03, np.nan]))
    assert np.allclose(adjusted.iloc[:3], [0.03, 0.04, 0.04])
    assert np.isnan(adjusted.iloc[3])
