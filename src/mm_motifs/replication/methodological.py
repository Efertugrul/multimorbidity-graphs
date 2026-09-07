from __future__ import annotations

import json

import numpy as np
import pandas as pd

from mm_motifs.motifs.mining import (
    motif_edge_masks,
    occurrence_matrix,
    paired_graph_indices,
    paired_motif_evaluability,
)


def evaluate_methodological_vocabulary(
    graph_ids: list[str],
    graph_masks: np.ndarray,
    opportunity_masks: np.ndarray,
    vocabulary: pd.DataFrame,
    registry: pd.DataFrame,
    edge_index: dict[tuple[str, str], int],
    minimum_support_fraction: float,
    minimum_state_pairs: int,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    if np.any(
        np.bitwise_and(graph_masks, opportunity_masks) != graph_masks
    ):
        raise ValueError("Replication graph contains an ineligible edge")
    motifs = vocabulary.rename(columns={"canonical_edges": "edge_list"}).copy()
    motif_masks = motif_edge_masks(motifs, edge_index)
    pairs = paired_graph_indices(graph_ids, registry)
    paired_evaluable, state_evaluable = paired_motif_evaluability(
        opportunity_masks,
        motif_masks,
        pairs,
    )
    occurrences = occurrence_matrix(graph_masks, motif_masks)
    valid_occurrences = occurrences & paired_evaluable
    lower = pairs["lower_index"].to_numpy(dtype=int)
    higher = pairs["higher_index"].to_numpy(dtype=int)
    paired_state_count = state_evaluable.sum(axis=0)
    evaluable_graph_count = 2 * paired_state_count
    support_count = valid_occurrences.sum(axis=0)
    state_support_count = (
        (occurrences[lower] | occurrences[higher]) & state_evaluable
    ).sum(axis=0)
    support_fraction = np.divide(
        support_count,
        evaluable_graph_count,
        out=np.full(len(motifs), np.nan, dtype=float),
        where=evaluable_graph_count > 0,
    )
    result = vocabulary.copy()
    result["replication_paired_state_count"] = paired_state_count
    result["replication_evaluable_graph_count"] = evaluable_graph_count
    result["replication_support_count"] = support_count
    result["replication_support_fraction"] = support_fraction
    result["replication_distinct_state_support_count"] = state_support_count
    result["replication_graph_ids"] = [
        json.dumps(
            [
                graph_ids[index]
                for index in np.flatnonzero(
                    valid_occurrences[:, motif_index]
                )
            ],
            separators=(",", ":"),
        )
        for motif_index in range(len(result))
    ]
    result["replicated_at_20pct"] = (
        result["replication_paired_state_count"].ge(minimum_state_pairs)
        & result["replication_support_fraction"].ge(
            minimum_support_fraction
        )
    )
    result["support_fraction_difference_2023_minus_2024"] = (
        result["replication_support_fraction"]
        - result["discovery_support_fraction"]
    )
    correlation_frame = result[
        result["replication_paired_state_count"].ge(minimum_state_pairs)
    ]
    summary: dict[str, float | int] = {
        "frozen_motif_count": int(len(result)),
        "replicated_motif_count": int(result["replicated_at_20pct"].sum()),
        "retention_fraction_all_frozen_motifs": float(
            result["replicated_at_20pct"].mean()
        ),
        "minimum_state_pair_count": int(
            result["replication_paired_state_count"].min()
        ),
        "support_spearman_correlation": float(
            correlation_frame["discovery_support_fraction"].corr(
                correlation_frame["replication_support_fraction"],
                method="spearman",
            )
        ),
        "median_absolute_support_difference": float(
            result[
                "support_fraction_difference_2023_minus_2024"
            ].abs().median()
        ),
    }
    return result, summary


def summarize_occurrence_families(
    motif_results: pd.DataFrame,
    families: pd.DataFrame,
) -> pd.DataFrame:
    grouped = (
        motif_results.groupby("occurrence_class_id", sort=True)
        .agg(
            replication_member_count=("motif_id", "size"),
            replicated_member_count=("replicated_at_20pct", "sum"),
            minimum_replication_support=(
                "replication_support_fraction",
                "min",
            ),
            median_replication_support=(
                "replication_support_fraction",
                "median",
            ),
            maximum_replication_support=(
                "replication_support_fraction",
                "max",
            ),
        )
        .reset_index()
    )
    result = families.merge(
        grouped,
        on="occurrence_class_id",
        how="left",
        validate="one_to_one",
    )
    if result["replication_member_count"].isna().any():
        raise ValueError("A frozen occurrence family has no replication motif")
    representative = motif_results[
        [
            "motif_id",
            "replication_paired_state_count",
            "replication_support_fraction",
            "replicated_at_20pct",
        ]
    ].rename(
        columns={
            "motif_id": "representative_motif_id",
            "replication_paired_state_count": (
                "representative_replication_paired_state_count"
            ),
            "replication_support_fraction": (
                "representative_replication_support_fraction"
            ),
            "replicated_at_20pct": (
                "representative_replicated_at_20pct"
            ),
        }
    )
    return result.merge(
        representative,
        on="representative_motif_id",
        how="left",
        validate="one_to_one",
    )
