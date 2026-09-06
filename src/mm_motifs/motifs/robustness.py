from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from mm_motifs.motifs.mining import (
    annotate_opportunity_support,
    graphs_from_edge_masks,
    mine_frequent_motifs,
    motif_edge_masks,
    occurrence_matrix,
    paired_graph_indices,
    paired_motif_evaluability,
)
from mm_motifs.statistics.multiple_testing import benjamini_hochberg


def bootstrap_graph_masks(
    bootstrap: pd.DataFrame,
    graph_ids: list[str],
    edge_index: dict[tuple[str, str], int],
    replicates: int,
) -> np.ndarray:
    graph_lookup = {graph_id: index for index, graph_id in enumerate(graph_ids)}
    result = np.zeros((replicates, len(graph_ids)), dtype=np.uint64)
    required = {
        "graph_id",
        "disease_a",
        "disease_b",
        "bootstrap_selection_mask",
        "pair_eligible",
    }
    if not required.issubset(bootstrap):
        raise ValueError(
            f"Bootstrap masks require columns {sorted(required)}"
        )
    for row in bootstrap.itertuples(index=False):
        if pd.isna(row.pair_eligible):
            raise ValueError("Bootstrap pair eligibility is missing")
        if not bool(row.pair_eligible):
            continue
        graph_id = str(row.graph_id)
        if graph_id not in graph_lookup:
            raise ValueError(f"Unknown bootstrap graph ID: {graph_id}")
        edge = tuple(sorted((str(row.disease_a), str(row.disease_b))))
        if edge not in edge_index:
            raise ValueError(f"Unknown bootstrap edge: {edge}")
        encoded = str(row.bootstrap_selection_mask)
        if not encoded.startswith("b"):
            raise ValueError("Bootstrap selection mask lacks its string prefix")
        encoded = encoded[1:]
        if len(encoded) != replicates or set(encoded) - {"0", "1", "x"}:
            raise ValueError("Invalid bootstrap selection mask")
        selected = np.frombuffer(encoded.encode(), dtype=np.uint8) == ord("1")
        result[selected, graph_lookup[graph_id]] |= (
            np.uint64(1) << np.uint64(edge_index[edge])
        )
    return result


def restrict_to_frozen_edges(
    threshold_replicate_masks: np.ndarray,
    frozen_graph_masks: np.ndarray,
) -> np.ndarray:
    if (
        threshold_replicate_masks.ndim != 2
        or frozen_graph_masks.ndim != 1
        or threshold_replicate_masks.shape[1] != len(frozen_graph_masks)
    ):
        raise ValueError("Bootstrap and frozen graph masks are misaligned")
    return np.bitwise_and(
        threshold_replicate_masks,
        frozen_graph_masks[np.newaxis, :],
    )


def bootstrap_motif_support(
    replicate_graph_masks: np.ndarray,
    motifs: pd.DataFrame,
    edge_index: dict[tuple[str, str], int],
    support_fractions: list[float],
    paired_evaluable: np.ndarray,
    chunk_size: int = 128,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    masks = motif_edge_masks(motifs, edge_index)
    replicate_count = replicate_graph_masks.shape[0]
    support = np.zeros((replicate_count, len(motifs)), dtype=np.int16)
    for start in range(0, len(motifs), chunk_size):
        stop = min(start + chunk_size, len(motifs))
        occurrences = (
            np.bitwise_and(
                replicate_graph_masks[:, :, np.newaxis],
                masks[np.newaxis, np.newaxis, start:stop],
            )
            == masks[np.newaxis, np.newaxis, start:stop]
        )
        occurrences &= paired_evaluable[
            np.newaxis,
            :,
            start:stop,
        ]
        support[:, start:stop] = occurrences.sum(axis=1)
    summary = motifs[
        ["motif_id", "node_count", "edge_count", "support_count"]
    ].copy()
    summary = summary.rename(
        columns={"support_count": "baseline_support_count"}
    )
    summary["paired_state_count"] = motifs[
        "paired_state_count"
    ].to_numpy()
    summary["evaluable_graph_count"] = motifs[
        "evaluable_graph_count"
    ].to_numpy()
    summary["bootstrap_support_mean"] = support.mean(axis=0)
    summary["bootstrap_support_median"] = np.median(support, axis=0)
    summary["bootstrap_support_025"] = np.quantile(
        support,
        0.025,
        axis=0,
    )
    summary["bootstrap_support_975"] = np.quantile(
        support,
        0.975,
        axis=0,
    )
    for fraction in support_fractions:
        suffix = int(round(fraction * 100))
        count = motifs[
            f"support_count_{suffix:02d}pct"
        ].to_numpy(dtype=int)
        summary[f"support_count_{suffix:02d}pct"] = count
        summary[f"support_{suffix:02d}pct"] = motifs[
            f"support_{suffix:02d}pct"
        ].to_numpy()
        summary[f"p_boot_support_{suffix:02d}pct"] = (
            support >= count[np.newaxis, :]
        ).mean(axis=0)
    summary["primary_support"] = motifs["primary_support"].to_numpy()
    long = pd.DataFrame(
        {
            "bootstrap_replicate": np.repeat(
                np.arange(1, replicate_count + 1),
                len(motifs),
            ),
            "motif_id": np.tile(
                motifs["motif_id"].to_numpy(),
                replicate_count,
            ),
            "support_count": support.reshape(-1),
            "evaluable_graph_count": np.tile(
                motifs["evaluable_graph_count"].to_numpy(),
                replicate_count,
            ),
        }
    )
    return summary, long, support


def frozen_vocabulary_stability(
    motifs: pd.DataFrame,
    bootstrap_support: np.ndarray,
    support_fractions: list[float],
) -> pd.DataFrame:
    rows = []
    for fraction in sorted(support_fractions):
        suffix = int(round(fraction * 100))
        baseline = motifs[f"support_{suffix:02d}pct"].to_numpy(dtype=bool)
        count = motifs[
            f"support_count_{suffix:02d}pct"
        ].to_numpy(dtype=int)
        replicate_selected = bootstrap_support >= count[np.newaxis, :]
        baseline_count = int(baseline.sum())
        intersection = replicate_selected[:, baseline].sum(axis=1)
        replicate_count = replicate_selected.sum(axis=1)
        union = baseline_count + replicate_count - intersection
        retention = intersection / baseline_count
        jaccard = np.divide(
            intersection,
            union,
            out=np.ones_like(intersection, dtype=float),
            where=union > 0,
        )
        for replicate_index in range(len(bootstrap_support)):
            rows.append(
                {
                    "support_fraction": fraction,
                    "minimum_support_count": int(count[baseline].min()),
                    "maximum_support_count": int(count[baseline].max()),
                    "bootstrap_replicate": replicate_index + 1,
                    "baseline_motif_count": baseline_count,
                    "replicate_baseline_vocabulary_count": int(
                        replicate_count[replicate_index]
                    ),
                    "retained_baseline_motif_count": int(
                        intersection[replicate_index]
                    ),
                    "baseline_vocabulary_retention": float(
                        retention[replicate_index]
                    ),
                    "baseline_vocabulary_jaccard": float(
                        jaccard[replicate_index]
                    ),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_discovery_set_stability(
    replicate_graph_masks: np.ndarray,
    graph_ids: list[str],
    condition_labels: pd.DataFrame,
    universe: list[tuple[str, str]],
    edge_index: dict[tuple[str, str], int],
    opportunity_masks: np.ndarray,
    registry: pd.DataFrame,
    baseline_motifs: pd.DataFrame,
    support_fractions: list[float],
    primary_support_fraction: float,
    minimum_paired_states: int,
    candidate_minimum_support: int,
    minimum_nodes: int,
    maximum_nodes: int,
    threads: int,
    backend_version: str,
) -> pd.DataFrame:
    baseline_sets = {}
    for fraction in support_fractions:
        column = f"support_{int(round(fraction * 100)):02d}pct"
        baseline_sets[fraction] = set(
            baseline_motifs.loc[baseline_motifs[column], "motif_id"]
        )
    rows = []
    for replicate_index, masks in enumerate(replicate_graph_masks):
        masks = np.bitwise_and(masks, opportunity_masks)
        graphs = graphs_from_edge_masks(
            masks,
            graph_ids,
            condition_labels,
            universe,
        )
        motifs, _ = mine_frequent_motifs(
            graphs,
            condition_labels,
            support_fractions,
            primary_support_fraction,
            minimum_nodes,
            maximum_nodes,
            threads,
            backend_version,
            include_redundancy=False,
            candidate_minimum_support=candidate_minimum_support,
        )
        motifs, _ = annotate_opportunity_support(
            motifs,
            masks,
            opportunity_masks,
            graph_ids,
            registry,
            edge_index,
            support_fractions,
            primary_support_fraction,
            minimum_paired_states,
        )
        for fraction in support_fractions:
            column = f"support_{int(round(fraction * 100)):02d}pct"
            discovered = set(motifs.loc[motifs[column], "motif_id"])
            baseline = baseline_sets[fraction]
            intersection = baseline & discovered
            union = baseline | discovered
            rows.append(
                {
                    "bootstrap_replicate": replicate_index + 1,
                    "support_fraction": fraction,
                    "baseline_motif_count": len(baseline),
                    "replicate_discovered_motif_count": len(discovered),
                    "retained_baseline_motif_count": len(intersection),
                    "novel_motif_count": len(discovered - baseline),
                    "discovery_set_retention": (
                        len(intersection) / len(baseline)
                        if baseline
                        else 1.0
                    ),
                    "discovery_set_jaccard": (
                        len(intersection) / len(union) if union else 1.0
                    ),
                }
            )
    return pd.DataFrame(rows)


def _sample_fixed_edge_masks(
    edge_counts: np.ndarray,
    opportunity_masks: np.ndarray,
    replicates: int,
    seed: int,
) -> np.ndarray:
    generator = np.random.default_rng(seed)
    masks = np.zeros((replicates, len(edge_counts)), dtype=np.uint64)
    for replicate in range(replicates):
        for graph_index, edge_count in enumerate(edge_counts):
            possible = np.asarray(
                [
                    index
                    for index in range(64)
                    if int(opportunity_masks[graph_index]) & (1 << index)
                ],
                dtype=int,
            )
            if int(edge_count) > len(possible):
                raise ValueError("Graph has more edges than eligible dyads")
            selected = generator.choice(
                possible,
                size=int(edge_count),
                replace=False,
            )
            value = np.uint64(0)
            for edge_index in selected:
                value |= np.uint64(1) << np.uint64(edge_index)
            masks[replicate, graph_index] = value
    return masks


def _mask_edges(
    mask: np.uint64,
    universe: list[tuple[str, str]],
) -> set[tuple[str, str]]:
    return {
        edge
        for index, edge in enumerate(universe)
        if int(mask) & (1 << index)
    }


def _edges_mask(
    edges: set[tuple[str, str]],
    edge_index: dict[tuple[str, str], int],
) -> np.uint64:
    value = np.uint64(0)
    for edge in edges:
        value |= np.uint64(1) << np.uint64(edge_index[edge])
    return value


def _degree_preserving_swap(
    edges: set[tuple[str, str]],
    target_swaps: int,
    generator: np.random.Generator,
    allowed_edges: set[tuple[str, str]] | None = None,
) -> tuple[set[tuple[str, str]], int, int]:
    current = set(edges)
    accepted = 0
    attempts = 0
    maximum_attempts = max(100, target_swaps * 30)
    while accepted < target_swaps and attempts < maximum_attempts:
        attempts += 1
        if len(current) < 2:
            break
        first_index, second_index = generator.choice(
            len(current),
            size=2,
            replace=False,
        )
        edge_values = tuple(sorted(current))
        first = edge_values[int(first_index)]
        second = edge_values[int(second_index)]
        a, b = first
        c, d = second
        if generator.random() < 0.5:
            c, d = d, c
        if len({a, b, c, d}) < 4:
            continue
        proposed = {
            tuple(sorted((a, c))),
            tuple(sorted((b, d))),
        }
        if len(proposed) < 2:
            continue
        if allowed_edges is not None and not proposed.issubset(allowed_edges):
            continue
        remainder = current - {first, second}
        if proposed & remainder:
            continue
        current = remainder | proposed
        accepted += 1
    return current, accepted, attempts


def degree_preserving_null_masks(
    graph_masks: np.ndarray,
    universe: list[tuple[str, str]],
    replicates: int,
    swaps_per_edge: int,
    seed: int,
    opportunity_masks: np.ndarray | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    edge_index = {edge: index for index, edge in enumerate(universe)}
    generator = np.random.default_rng(seed)
    result = np.zeros(
        (replicates, len(graph_masks)),
        dtype=np.uint64,
    )
    diagnostics = []
    baseline_edges = [_mask_edges(mask, universe) for mask in graph_masks]
    allowed_by_graph = (
        [set(universe)] * len(graph_masks)
        if opportunity_masks is None
        else [
            _mask_edges(mask, universe)
            for mask in opportunity_masks
        ]
    )
    for replicate in range(replicates):
        for graph_index, (edges, allowed_edges) in enumerate(
            zip(baseline_edges, allowed_by_graph, strict=True)
        ):
            if not edges.issubset(allowed_edges):
                raise ValueError("Observed graph contains an ineligible dyad")
            target = swaps_per_edge * len(edges)
            randomized, accepted, attempts = _degree_preserving_swap(
                edges,
                target,
                generator,
                allowed_edges,
            )
            result[replicate, graph_index] = _edges_mask(
                randomized,
                edge_index,
            )
            randomized_mask = result[replicate, graph_index]
            diagnostics.append(
                {
                    "null_replicate": replicate + 1,
                    "graph_index": graph_index,
                    "target_swaps": target,
                    "accepted_swaps": accepted,
                    "attempted_swaps": attempts,
                    "acceptance_rate": (
                        accepted / attempts if attempts else np.nan
                    ),
                    "changed": randomized != edges,
                    "randomized_edge_mask": f"{int(randomized_mask):012x}",
                }
            )
    return result, pd.DataFrame(diagnostics)


def _null_support_summary(
    name: str,
    null_graph_masks: np.ndarray,
    observed_occurrences: np.ndarray,
    motifs: pd.DataFrame,
    edge_index: dict[tuple[str, str], int],
    paired_evaluable: np.ndarray,
    chunk_size: int = 128,
) -> tuple[pd.DataFrame, np.ndarray]:
    masks = motif_edge_masks(motifs, edge_index)
    support = np.zeros(
        (null_graph_masks.shape[0], len(motifs)),
        dtype=np.int16,
    )
    for start in range(0, len(motifs), chunk_size):
        stop = min(start + chunk_size, len(motifs))
        values = (
            np.bitwise_and(
                null_graph_masks[:, :, np.newaxis],
                masks[np.newaxis, np.newaxis, start:stop],
            )
            == masks[np.newaxis, np.newaxis, start:stop]
        )
        values &= paired_evaluable[
            np.newaxis,
            :,
            start:stop,
        ]
        support[:, start:stop] = values.sum(axis=1)
    observed = (observed_occurrences & paired_evaluable).sum(axis=0)
    mean = support.mean(axis=0)
    standard_deviation = support.std(axis=0, ddof=1)
    z_score = np.divide(
        observed - mean,
        standard_deviation,
        out=np.full(len(motifs), np.nan),
        where=standard_deviation > 0,
    )
    p_upper = (
        1 + (support >= observed[np.newaxis, :]).sum(axis=0)
    ) / (len(support) + 1)
    summary = pd.DataFrame(
        {
            "motif_id": motifs["motif_id"],
            "null_model": name,
            "evaluable_graph_count": motifs["evaluable_graph_count"],
            "observed_support_count": observed,
            "null_support_mean": mean,
            "null_support_standard_deviation": standard_deviation,
            "null_support_025": np.quantile(support, 0.025, axis=0),
            "null_support_975": np.quantile(support, 0.975, axis=0),
            "density_null_z": z_score,
            "density_null_p_upper": p_upper,
        }
    )
    return summary, support


def density_null_support(
    graph_masks: np.ndarray,
    opportunity_masks: np.ndarray,
    paired_evaluable: np.ndarray,
    motifs: pd.DataFrame,
    edge_index: dict[tuple[str, str], int],
    fixed_edge_replicates: int,
    degree_replicates: int,
    swaps_per_edge: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    motif_masks = motif_edge_masks(motifs, edge_index)
    observed = occurrence_matrix(graph_masks, motif_masks)
    edge_counts = np.asarray(
        [int(int(mask).bit_count()) for mask in graph_masks],
        dtype=int,
    )
    fixed_masks = _sample_fixed_edge_masks(
        edge_counts,
        opportunity_masks,
        fixed_edge_replicates,
        seed,
    )
    fixed_summary, _ = _null_support_summary(
        "fixed_eligible_dyad_edge_count",
        fixed_masks,
        observed,
        motifs,
        edge_index,
        paired_evaluable,
    )
    universe = [
        edge
        for edge, _ in sorted(edge_index.items(), key=lambda item: item[1])
    ]
    degree_masks, diagnostics = degree_preserving_null_masks(
        graph_masks,
        universe,
        degree_replicates,
        swaps_per_edge,
        seed + 1,
        opportunity_masks,
    )
    degree_summary, _ = _null_support_summary(
        "degree_preserving_eligible_dyad_edge_swap",
        degree_masks,
        observed,
        motifs,
        edge_index,
        paired_evaluable,
    )
    return (
        pd.concat([fixed_summary, degree_summary], ignore_index=True),
        diagnostics,
    )


def fixed_edge_occurrence_probability(
    possible_edges: int,
    graph_edges: int,
    motif_edges: int,
) -> float:
    if motif_edges > possible_edges or graph_edges < motif_edges:
        return 0.0
    if graph_edges > possible_edges:
        raise ValueError("Graph edge count exceeds eligible dyad count")
    return math.comb(
        possible_edges - motif_edges,
        graph_edges - motif_edges,
    ) / math.comb(possible_edges, graph_edges)


def density_residuals(
    graph_masks: np.ndarray,
    opportunity_masks: np.ndarray,
    paired_evaluable: np.ndarray,
    motifs: pd.DataFrame,
    edge_index: dict[tuple[str, str], int],
) -> tuple[np.ndarray, np.ndarray]:
    motif_masks = motif_edge_masks(motifs, edge_index)
    observed = occurrence_matrix(graph_masks, motif_masks).astype(float)
    edge_counts = np.asarray(
        [int(int(mask).bit_count()) for mask in graph_masks],
        dtype=int,
    )
    possible_edge_counts = np.asarray(
        [int(int(mask).bit_count()) for mask in opportunity_masks],
        dtype=int,
    )
    motif_edge_counts = motifs["edge_count"].to_numpy(dtype=int)
    expected = np.empty_like(observed)
    for graph_index, graph_edges in enumerate(edge_counts):
        expected[graph_index] = [
            fixed_edge_occurrence_probability(
                int(possible_edge_counts[graph_index]),
                int(graph_edges),
                int(motif_edges),
            )
            for motif_edges in motif_edge_counts
        ]
    observed[~paired_evaluable] = np.nan
    expected[~paired_evaluable] = np.nan
    return observed - expected, observed


def _paired_t(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    count = finite.sum(axis=0)
    safe = np.nan_to_num(values, nan=0.0)
    means = np.divide(
        safe.sum(axis=0),
        count,
        out=np.zeros(values.shape[1], dtype=float),
        where=count > 0,
    )
    sums_of_squares = np.square(safe).sum(axis=0)
    variances = np.maximum(
        np.divide(
            sums_of_squares - count * np.square(means),
            count - 1,
            out=np.zeros(values.shape[1], dtype=float),
            where=count > 1,
        ),
        0,
    )
    standard_errors = np.sqrt(
        np.divide(
            variances,
            count,
            out=np.zeros(values.shape[1], dtype=float),
            where=count > 0,
        )
    )
    result = np.divide(
        means,
        standard_errors,
        out=np.zeros_like(means),
        where=standard_errors > 0,
    )
    degenerate = standard_errors == 0
    result[degenerate & (means != 0)] = np.sign(
        means[degenerate & (means != 0)]
    ) * np.inf
    return result


def paired_ses_permutation(
    graph_ids: list[str],
    graph_masks: np.ndarray,
    opportunity_masks: np.ndarray,
    motifs: pd.DataFrame,
    registry: pd.DataFrame,
    edge_index: dict[tuple[str, str], int],
    permutations: int,
    seed: int,
    minimum_paired_states: int,
    batch_size: int = 250,
) -> pd.DataFrame:
    motif_masks = motif_edge_masks(motifs, edge_index)
    pair_table = paired_graph_indices(graph_ids, registry)
    paired_evaluable, state_evaluable = paired_motif_evaluability(
        opportunity_masks,
        motif_masks,
        pair_table,
    )
    residuals, observed = density_residuals(
        graph_masks,
        opportunity_masks,
        paired_evaluable,
        motifs,
        edge_index,
    )
    lower_indices = pair_table["lower_index"].to_numpy(dtype=int)
    higher_indices = pair_table["higher_index"].to_numpy(dtype=int)
    differences = residuals[lower_indices] - residuals[higher_indices]
    pair_counts = state_evaluable.sum(axis=0)
    if np.any(pair_counts < minimum_paired_states):
        raise ValueError("Motif lacks the minimum paired-state opportunity")
    observed_t = _paired_t(differences)
    exceedances = np.zeros(len(motifs), dtype=int)
    maximum_exceedances = np.zeros(len(motifs), dtype=int)
    generator = np.random.default_rng(seed)
    completed = 0
    while completed < permutations:
        size = min(batch_size, permutations - completed)
        signs = generator.choice(
            np.asarray([-1.0, 1.0]),
            size=(size, len(pair_table)),
        )
        safe_differences = np.nan_to_num(differences, nan=0.0)
        means = signs @ safe_differences / pair_counts
        sums_of_squares = np.square(safe_differences).sum(axis=0)
        variances = np.maximum(
            (
                sums_of_squares[np.newaxis, :]
                - pair_counts[np.newaxis, :] * np.square(means)
            )
            / (pair_counts[np.newaxis, :] - 1),
            0,
        )
        standard_errors = np.sqrt(
            variances / pair_counts[np.newaxis, :]
        )
        permuted_t = np.divide(
            means,
            standard_errors,
            out=np.zeros_like(means),
            where=standard_errors > 0,
        )
        degenerate = (standard_errors == 0) & (means != 0)
        permuted_t[degenerate] = np.sign(means[degenerate]) * np.inf
        absolute = np.abs(permuted_t)
        exceedances += (
            absolute >= np.abs(observed_t)[np.newaxis, :]
        ).sum(axis=0)
        maximum = absolute.max(axis=1)
        maximum_exceedances += (
            maximum[:, np.newaxis]
            >= np.abs(observed_t)[np.newaxis, :]
        ).sum(axis=0)
        completed += size
    p_value = (exceedances + 1) / (permutations + 1)
    max_t_p_value = (maximum_exceedances + 1) / (permutations + 1)
    lower_support = np.nansum(observed[lower_indices], axis=0)
    higher_support = np.nansum(observed[higher_indices], axis=0)
    result = pd.DataFrame(
        {
            "motif_id": motifs["motif_id"],
            "state_pair_count": pair_counts,
            "lower_support_count": lower_support,
            "higher_support_count": higher_support,
            "raw_support_fraction_difference": (
                lower_support - higher_support
            )
            / pair_counts,
            "density_residual_mean_lower": np.nanmean(
                residuals[lower_indices],
                axis=0,
            ),
            "density_residual_mean_higher": np.nanmean(
                residuals[higher_indices],
                axis=0,
            ),
            "paired_density_residual_difference": np.nanmean(
                differences,
                axis=0,
            ),
            "paired_density_residual_t": observed_t,
            "permutation_p_value": p_value,
            "permutation_max_t_p_value": max_t_p_value,
            "permutation_count": permutations,
            "permutation_scheme": "within_state_lower_higher_label_swap",
            "density_adjustment": (
                "fixed_eligible_dyad_edge_count_expected_occurrence"
            ),
        }
    )
    result["permutation_bh_q_value"] = benjamini_hochberg(
        result["permutation_p_value"]
    )
    result["permutation_p_value_mcse"] = np.sqrt(
        result["permutation_p_value"]
        * (1 - result["permutation_p_value"])
        / (permutations + 1)
    )
    result["permutation_max_t_p_value_mcse"] = np.sqrt(
        result["permutation_max_t_p_value"]
        * (1 - result["permutation_max_t_p_value"])
        / (permutations + 1)
    )
    return result


def motif_shape_summary(motifs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    support_columns = sorted(
        column
        for column in motifs
        if column.startswith("support_") and column.endswith("pct")
        and not column.startswith("support_count_")
    )
    for support_column in support_columns:
        scoped = motifs[motifs[support_column]]
        for node_count, frame in scoped.groupby("node_count", sort=True):
            rows.append(
                {
                    "support_level": support_column,
                    "node_count": int(node_count),
                    "motif_count": len(frame),
                    "closed_motif_count": int(frame["is_closed"].sum()),
                    "maximal_motif_count": int(frame["is_maximal"].sum()),
                    "occurrence_class_count": int(
                        frame["occurrence_class_id"].nunique()
                    ),
                    "tree_fraction": float(frame["is_tree"].mean()),
                    "clique_fraction": float(frame["is_clique"].mean()),
                    "median_edge_count": float(frame["edge_count"].median()),
                    "median_motif_density": float(
                        frame["motif_density"].median()
                    ),
                }
            )
    return pd.DataFrame(rows)
