from __future__ import annotations

import pandas as pd

from mm_motifs.config import load_config
from mm_motifs.features.conditions import derive_conditions
from mm_motifs.features.socioeconomic import derive_ses


def test_condition_coding_distinguishes_negative_and_missing() -> None:
    config = load_config()
    frame = pd.DataFrame(
        {
            "_MICHD": [1, 2, 7, None],
            "CVDSTRK3": [2, 1, 9, None],
            "_CASTHM1": [2, 1, 1, 9],
            "CHCSCNC1": [1, 2, 7, 9],
            "CHCOCNC1": [1, 2, 7, 9],
            "CHCCOPD3": [1, 2, 7, 9],
            "ADDEPEV3": [1, 2, 7, 9],
            "CHCKDNY2": [1, 2, 7, 9],
            "HAVARTH4": [1, 2, 7, 9],
            "DIABETE4": [1, 2, 3, 4],
        }
    )
    result = derive_conditions(frame, config.conditions, 2024)
    assert result["ISCHEMIC_HEART_DISEASE"].tolist() == [1.0, 0.0, pd.NA, pd.NA]
    assert result["CURRENT_ASTHMA"].tolist() == [1.0, 0.0, 0.0, pd.NA]
    assert result["DIABETES"].tolist() == [1.0, 0.0, 0.0, 0.0]


def test_ses_definitions_are_configurable() -> None:
    config = load_config()
    frame = pd.DataFrame(
        {
            "_EDUCAG": [1, 2, 3, 4, 9],
            "_INCOMG1": [1, 4, 5, 7, 9],
        }
    )
    education = derive_ses(frame, config.analysis, 2024)
    income = derive_ses(frame, config.analysis, 2024, "income_binary")
    assert education.tolist() == ["lower", "lower", "higher", "higher", pd.NA]
    assert income.tolist() == ["lower", "lower", "higher", "higher", pd.NA]
