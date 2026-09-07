from __future__ import annotations

import numpy as np
import pandas as pd

from mm_motifs.motifs.mining import (
    motif_edge_masks,
    paired_graph_indices,
    paired_motif_evaluability,
)
from mm_motifs.motifs.robustness import density_residuals


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or np.any(~np.isfinite(values)):
        raise ValueError("Holm adjustment requires finite p-values")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("P-values must be in [0, 1]")
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    adjusted_ranked = np.minimum(
        1.0,
        np.maximum.accumulate(
            (len(values) - np.arange(len(values))) * ranked
        ),
    )
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return adjusted


def rademacher_sign_bank(
    state_codes: np.ndarray,
    permutations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    states = np.asarray(state_codes, dtype=int)
    if len(states) != len(np.unique(states)):
        raise ValueError("State codes must be unique")
    order = np.argsort(states, kind="stable")
    states = states[order]
    generator = np.random.Generator(np.random.PCG64(seed))
    bits = generator.integers(
        0,
        2,
        size=(permutations, len(states)),
        dtype=np.int8,
    )
    return states, bits * 2 - 1


def _studentized_mean(values: np.ndarray) -> float:
    count = len(values)
    if count < 2:
        return np.nan
    mean = float(values.mean())
    variance = max(
        float((np.square(values).sum() - count * mean**2) / (count - 1)),
        0.0,
    )
    standard_error = float(np.sqrt(variance / count))
    if standard_error > 0:
        return mean / standard_error
    if mean == 0:
        return 0.0
    return float(np.sign(mean) * np.inf)


def _permuted_studentized_means(
    values: np.ndarray,
    signs: np.ndarray,
) -> np.ndarray:
    count = len(values)
    means = signs @ values / count
    sum_of_squares = float(np.square(values).sum())
    variances = np.maximum(
        (sum_of_squares - count * np.square(means)) / (count - 1),
        0.0,
    )
    standard_errors = np.sqrt(variances / count)
    statistics = np.divide(
        means,
        standard_errors,
        out=np.zeros_like(means),
        where=standard_errors > 0,
    )
    degenerate = (standard_errors == 0) & (means != 0)
    statistics[degenerate] = np.sign(means[degenerate]) * np.inf
    return statistics


def directional_sign_flip_test(
    differences: pd.DataFrame,
    hypotheses: pd.DataFrame,
    signs: np.ndarray,
    minimum_state_pairs: int,
    alpha: float,
    decision_eligible: bool,
) -> pd.DataFrame:
    if list(differences.columns) != list(hypotheses["hypothesis_id"]):
        raise ValueError("Hypothesis differences are misordered")
    if signs.ndim != 2 or signs.shape[1] != len(differences):
        raise ValueError("Sign bank and state differences are misaligned")
    rows = []
    for column_index, hypothesis in enumerate(
        hypotheses.itertuples(index=False)
    ):
        values = differences.iloc[:, column_index].to_numpy(dtype=float)
        valid = np.isfinite(values)
        count = int(valid.sum())
        effect = float(np.nanmean(values)) if count else np.nan
        direction = int(hypothesis.direction_sign)
        if count < minimum_state_pairs:
            statistic = np.nan
            oriented_statistic = np.nan
            p_value = 1.0
            status = "insufficient_opportunity"
        else:
            selected = values[valid]
            statistic = _studentized_mean(selected)
            oriented_statistic = direction * statistic
            permuted = _permuted_studentized_means(
                selected,
                signs[:, valid],
            )
            oriented_permuted = direction * permuted
            p_value = float(
                (
                    1
                    + np.count_nonzero(
                        oriented_permuted >= oriented_statistic
                    )
                )
                / (len(signs) + 1)
            )
            status = "tested"
        rows.append(
            {
                "family_order": int(hypothesis.family_order),
                "hypothesis_id": hypothesis.hypothesis_id,
                "motif_id": hypothesis.motif_id,
                "expected_enrichment": hypothesis.expected_enrichment,
                "direction_sign": direction,
                "state_pair_count": count,
                "effect_lower_minus_higher": effect,
                "studentized_statistic": statistic,
                "oriented_studentized_statistic": oriented_statistic,
                "one_sided_p_value": p_value,
                "test_status": status,
            }
        )
    result = pd.DataFrame(rows).sort_values("family_order").reset_index(
        drop=True
    )
    result["holm_adjusted_p_value"] = holm_adjust(
        result["one_sided_p_value"].to_numpy()
    )
    result["p_value_mcse"] = np.sqrt(
        result["one_sided_p_value"]
        * (1 - result["one_sided_p_value"])
        / (len(signs) + 1)
    )
    result["direction_concordant"] = (
        result["direction_sign"] * result["effect_lower_minus_higher"]
    ).gt(0)
    result["meets_statistical_rule"] = (
        result["test_status"].eq("tested")
        & result["direction_concordant"]
        & result["holm_adjusted_p_value"].le(alpha)
    )
    result["decision_eligible"] = decision_eligible
    result["replicated"] = (
        result["meets_statistical_rule"] & decision_eligible
    )
    return result


def hypothesis_state_differences(
    graph_ids: list[str],
    graph_masks: np.ndarray,
    opportunity_masks: np.ndarray,
    hypotheses: pd.DataFrame,
    registry: pd.DataFrame,
    edge_index: dict[tuple[str, str], int],
) -> pd.DataFrame:
    if np.any(
        np.bitwise_and(graph_masks, opportunity_masks) != graph_masks
    ):
        raise ValueError("Replication graph contains an ineligible edge")
    motifs = hypotheses[["motif_id", "edge_count", "edge_list"]].copy()
    motif_masks = motif_edge_masks(motifs, edge_index)
    pairs = paired_graph_indices(graph_ids, registry)
    paired_evaluable, _ = paired_motif_evaluability(
        opportunity_masks,
        motif_masks,
        pairs,
    )
    residuals, _ = density_residuals(
        graph_masks,
        opportunity_masks,
        paired_evaluable,
        motifs,
        edge_index,
    )
    lower = pairs["lower_index"].to_numpy(dtype=int)
    higher = pairs["higher_index"].to_numpy(dtype=int)
    values = residuals[lower] - residuals[higher]
    return pd.DataFrame(
        values,
        index=pairs["geography_code"].to_numpy(dtype=int),
        columns=hypotheses["hypothesis_id"].tolist(),
    ).sort_index()


def evaluate_confirmatory_replication(
    graph_ids: list[str],
    graph_masks: np.ndarray,
    opportunity_masks: np.ndarray,
    hypotheses: pd.DataFrame,
    registry: pd.DataFrame,
    edge_index: dict[tuple[str, str], int],
    discovery_state_codes: set[int],
    permutations: int,
    seed: int,
    minimum_state_pairs: int,
    alpha: float,
) -> pd.DataFrame:
    ordered = hypotheses.sort_values("family_order").reset_index(drop=True)
    differences = hypothesis_state_differences(
        graph_ids,
        graph_masks,
        opportunity_masks,
        ordered,
        registry,
        edge_index,
    )
    state_codes, signs = rademacher_sign_bank(
        differences.index.to_numpy(dtype=int),
        permutations,
        seed,
    )
    differences = differences.loc[state_codes]
    primary = directional_sign_flip_test(
        differences,
        ordered,
        signs,
        minimum_state_pairs,
        alpha,
        True,
    )
    primary.insert(0, "analysis_set", "primary_2023_eligible")
    sensitivity_mask = np.isin(state_codes, sorted(discovery_state_codes))
    sensitivity = directional_sign_flip_test(
        differences.loc[state_codes[sensitivity_mask]],
        ordered,
        signs[:, sensitivity_mask],
        minimum_state_pairs,
        alpha,
        False,
    )
    sensitivity.insert(0, "analysis_set", "matched_2024_jurisdictions")
    return pd.concat([primary, sensitivity], ignore_index=True)
