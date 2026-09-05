from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from mm_motifs.motifs.mining import (
    graphs_from_edge_masks,
    mine_frequent_motifs,
    motif_edge_masks,
    occurrence_matrix,
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
    support_counts: dict[float, int],
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
        support[:, start:stop] = occurrences.sum(axis=1)
    summary = motifs[
        ["motif_id", "node_count", "edge_count", "support_count"]
    ].copy()
    summary = summary.rename(
        columns={"support_count": "baseline_support_count"}
    )
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
    for fraction, count in support_counts.items():
        suffix = int(round(fraction * 100))
        summary[f"p_boot_support_{suffix:02d}pct"] = (
            support >= count
        ).mean(axis=0)
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
        }
    )
    return summary, long, support


def frozen_vocabulary_stability(
    motifs: pd.DataFrame,
    bootstrap_support: np.ndarray,
    support_counts: dict[float, int],
) -> pd.DataFrame:
    rows = []
    for fraction, count in sorted(support_counts.items()):
        baseline = motifs["support_count"].ge(count).to_numpy()
        replicate_selected = bootstrap_support >= count
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
                    "support_count": count,
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
    baseline_motifs: pd.DataFrame,
    support_fractions: list[float],
    primary_support_fraction: float,
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
    possible_edge_count: int,
    replicates: int,
    seed: int,
) -> np.ndarray:
    generator = np.random.default_rng(seed)
    masks = np.zeros((replicates, len(edge_counts)), dtype=np.uint64)
    for replicate in range(replicates):
        for graph_index, edge_count in enumerate(edge_counts):
            selected = generator.choice(
                possible_edge_count,
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
) -> tuple[set[tuple[str, str]], int]:
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
        remainder = current - {first, second}
        if proposed & remainder:
            continue
        current = remainder | proposed
        accepted += 1
    return current, accepted


def degree_preserving_null_masks(
    graph_masks: np.ndarray,
    universe: list[tuple[str, str]],
    replicates: int,
    swaps_per_edge: int,
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    edge_index = {edge: index for index, edge in enumerate(universe)}
    generator = np.random.default_rng(seed)
    result = np.zeros(
        (replicates, len(graph_masks)),
        dtype=np.uint64,
    )
    diagnostics = []
    baseline_edges = [_mask_edges(mask, universe) for mask in graph_masks]
    for replicate in range(replicates):
        for graph_index, edges in enumerate(baseline_edges):
            target = swaps_per_edge * len(edges)
            randomized, accepted = _degree_preserving_swap(
                edges,
                target,
                generator,
            )
            result[replicate, graph_index] = _edges_mask(
                randomized,
                edge_index,
            )
            diagnostics.append(
                {
                    "null_replicate": replicate + 1,
                    "graph_index": graph_index,
                    "target_swaps": target,
                    "accepted_swaps": accepted,
                    "changed": randomized != edges,
                }
            )
    return result, pd.DataFrame(diagnostics)


def _null_support_summary(
    name: str,
    null_graph_masks: np.ndarray,
    observed_occurrences: np.ndarray,
    motifs: pd.DataFrame,
    edge_index: dict[tuple[str, str], int],
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
        support[:, start:stop] = values.sum(axis=1)
    observed = observed_occurrences.sum(axis=0)
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
        len(edge_index),
        fixed_edge_replicates,
        seed,
    )
    fixed_summary, _ = _null_support_summary(
        "fixed_node_edge_count",
        fixed_masks,
        observed,
        motifs,
        edge_index,
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
    )
    degree_summary, _ = _null_support_summary(
        "degree_preserving_edge_swap",
        degree_masks,
        observed,
        motifs,
        edge_index,
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
    if graph_edges < motif_edges:
        return 0.0
    return math.comb(
        possible_edges - motif_edges,
        graph_edges - motif_edges,
    ) / math.comb(possible_edges, graph_edges)


def density_residuals(
    graph_masks: np.ndarray,
    motifs: pd.DataFrame,
    edge_index: dict[tuple[str, str], int],
) -> tuple[np.ndarray, np.ndarray]:
    motif_masks = motif_edge_masks(motifs, edge_index)
    observed = occurrence_matrix(graph_masks, motif_masks).astype(float)
    edge_counts = np.asarray(
        [int(int(mask).bit_count()) for mask in graph_masks],
        dtype=int,
    )
    motif_edge_counts = motifs["edge_count"].to_numpy(dtype=int)
    expected = np.empty_like(observed)
    for graph_index, graph_edges in enumerate(edge_counts):
        expected[graph_index] = [
            fixed_edge_occurrence_probability(
                len(edge_index),
                int(graph_edges),
                int(motif_edges),
            )
            for motif_edges in motif_edge_counts
        ]
    return observed - expected, observed


def _paired_t(values: np.ndarray) -> np.ndarray:
    count = values.shape[0]
    means = values.mean(axis=0)
    sums_of_squares = np.square(values).sum(axis=0)
    variances = np.maximum(
        (sums_of_squares - count * np.square(means)) / (count - 1),
        0,
    )
    standard_errors = np.sqrt(variances / count)
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
    motifs: pd.DataFrame,
    registry: pd.DataFrame,
    edge_index: dict[tuple[str, str], int],
    permutations: int,
    seed: int,
    batch_size: int = 250,
) -> pd.DataFrame:
    residuals, observed = density_residuals(
        graph_masks,
        motifs,
        edge_index,
    )
    graph_lookup = {graph_id: index for index, graph_id in enumerate(graph_ids)}
    pairs = []
    eligible_registry = registry[
        registry["graph_id"].isin(graph_lookup)
    ]
    for state_code, eligible in eligible_registry.groupby(
        "geography_code",
        sort=True,
    ):
        lower = eligible[eligible["ses_category"].eq("lower")]
        higher = eligible[eligible["ses_category"].eq("higher")]
        if len(lower) != 1 or len(higher) != 1:
            raise ValueError(f"State {state_code} lacks one paired SES graph")
        pairs.append(
            (
                int(state_code),
                graph_lookup[str(lower.iloc[0]["graph_id"])],
                graph_lookup[str(higher.iloc[0]["graph_id"])],
            )
        )
    lower_indices = np.asarray([value[1] for value in pairs], dtype=int)
    higher_indices = np.asarray([value[2] for value in pairs], dtype=int)
    differences = residuals[lower_indices] - residuals[higher_indices]
    observed_t = _paired_t(differences)
    exceedances = np.zeros(len(motifs), dtype=int)
    maximum_exceedances = np.zeros(len(motifs), dtype=int)
    generator = np.random.default_rng(seed)
    completed = 0
    while completed < permutations:
        size = min(batch_size, permutations - completed)
        signs = generator.choice(
            np.asarray([-1.0, 1.0]),
            size=(size, len(pairs)),
        )
        means = signs @ differences / len(pairs)
        sums_of_squares = np.square(differences).sum(axis=0)
        variances = np.maximum(
            (
                sums_of_squares[np.newaxis, :]
                - len(pairs) * np.square(means)
            )
            / (len(pairs) - 1),
            0,
        )
        standard_errors = np.sqrt(variances / len(pairs))
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
    lower_support = observed[lower_indices].sum(axis=0)
    higher_support = observed[higher_indices].sum(axis=0)
    result = pd.DataFrame(
        {
            "motif_id": motifs["motif_id"],
            "state_pair_count": len(pairs),
            "lower_support_count": lower_support,
            "higher_support_count": higher_support,
            "raw_support_fraction_difference": (
                lower_support - higher_support
            )
            / len(pairs),
            "density_residual_mean_lower": residuals[
                lower_indices
            ].mean(axis=0),
            "density_residual_mean_higher": residuals[
                higher_indices
            ].mean(axis=0),
            "paired_density_residual_difference": differences.mean(axis=0),
            "paired_density_residual_t": observed_t,
            "permutation_p_value": p_value,
            "permutation_max_t_p_value": max_t_p_value,
            "permutation_count": permutations,
            "permutation_scheme": "within_state_lower_higher_label_swap",
            "density_adjustment": "fixed_node_edge_count_expected_occurrence",
        }
    )
    result["permutation_bh_q_value"] = benjamini_hochberg(
        result["permutation_p_value"]
    )
    return result


def motif_shape_summary(motifs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    support_columns = sorted(
        column
        for column in motifs
        if column.startswith("support_") and column.endswith("pct")
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
