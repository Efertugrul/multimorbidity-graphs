from __future__ import annotations

import numpy as np
import pandas as pd


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(p_values, errors="coerce")
    result = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = numeric.notna() & np.isfinite(numeric) & numeric.between(0, 1)
    if not valid.any():
        return result
    values = numeric.loc[valid]
    order = np.argsort(values.to_numpy())
    ranked = values.to_numpy()[order]
    count = len(ranked)
    adjusted = ranked * count / np.arange(1, count + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    original_positions = values.index.to_numpy()[order]
    result.loc[original_positions] = adjusted
    return result
