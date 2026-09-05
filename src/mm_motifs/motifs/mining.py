from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd


def edge_universe(conditions: list[str]) -> list[tuple[str, str]]:
    return [tuple(sorted(edge)) for edge in combinations(sorted(conditions), 2)]


def graph_database_from_edges(
    edge_table: pd.DataFrame,
    rule_id: str,
    edge_label: int = 1,
) -> tuple[list[nx.Graph], pd.DataFrame]:
    scoped = edge_table[edge_table["rule_id"].eq(rule_id)].copy()
    if scoped.empty:
        raise ValueError(f"No edges found for frozen rule {rule_id}")
    conditions = sorted(
        set(scoped["source_condition"]) | set(scoped["target_condition"])
    )
    label_rows = [
        {"condition": condition, "gspan_label": index + 1}
        for index, condition in enumerate(conditions)
    ]
    labels = pd.DataFrame(label_rows)
    label_lookup = labels.set_index("condition")["gspan_label"].to_dict()
    graphs = []
    for graph_id, frame in scoped.groupby("graph_id", sort=True):
        graph = nx.Graph(graph_id=str(graph_id))
        graph.add_nodes_from(
            (
                condition,
                {
                    "label": int(label_lookup[condition]),
                    "condition": condition,
                },
            )
            for condition in conditions
        )
        selected = frame[frame["edge_present"].fillna(False)]
        graph.add_edges_from(
            (
                str(row.source_condition),
                str(row.target_condition),
                {"label": int(edge_label)},
            )
            for row in selected.itertuples(index=False)
        )
        graphs.append(graph)
    if len({graph.graph["graph_id"] for graph in graphs}) != len(graphs):
        raise ValueError("Frozen graph identifiers are not unique")
    return graphs, labels


def support_counts(
    graph_count: int,
    support_fractions: list[float],
) -> dict[float, int]:
    return {
        float(fraction): int(math.ceil(float(fraction) * graph_count))
        for fraction in support_fractions
    }


def _canonical_pattern(
    vertices: list[tuple[int, int]],
    edges: list[tuple[int, int, int]],
    inverse_labels: dict[int, str],
) -> tuple[list[str], list[tuple[str, str]], str]:
    vertex_labels = {
        int(vertex_id): inverse_labels[int(label)]
        for vertex_id, label in vertices
    }
    nodes = sorted(vertex_labels.values())
    canonical_edges = sorted(
        tuple(
            sorted(
                (
                    vertex_labels[int(source)],
                    vertex_labels[int(target)],
                )
            )
        )
        for source, target, _ in edges
    )
    payload = {
        "nodes": nodes,
        "edges": [list(edge) for edge in canonical_edges],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    return nodes, canonical_edges, f"motif_{digest}"


def _pattern_sets(row: pd.Series) -> tuple[frozenset[str], frozenset[tuple[str, str]]]:
    return (
        frozenset(json.loads(row["node_labels"])),
        frozenset(tuple(edge) for edge in json.loads(row["edge_list"])),
    )


def _strict_subgraph(
    candidate: tuple[frozenset[str], frozenset[tuple[str, str]]],
    possible_supergraph: tuple[
        frozenset[str],
        frozenset[tuple[str, str]],
    ],
) -> bool:
    candidate_nodes, candidate_edges = candidate
    super_nodes, super_edges = possible_supergraph
    return (
        candidate_nodes.issubset(super_nodes)
        and candidate_edges.issubset(super_edges)
        and (
            candidate_nodes != super_nodes
            or candidate_edges != super_edges
        )
    )


def annotate_redundancy(motifs: pd.DataFrame) -> pd.DataFrame:
    result = motifs.copy()
    result["occurrence_class_id"] = result["graph_ids"].map(
        lambda value: "occ_"
        + hashlib.sha256(value.encode()).hexdigest()[:16]
    )
    result["occurrence_class_size"] = result.groupby(
        "occurrence_class_id"
    )["motif_id"].transform("size")
    pattern_sets = {
        row.motif_id: _pattern_sets(pd.Series(row._asdict()))
        for row in result.itertuples(index=False)
    }
    occurrence_groups = {
        key: frame["motif_id"].tolist()
        for key, frame in result.groupby("occurrence_class_id", sort=False)
    }
    closed = {}
    for motif_id, pattern in pattern_sets.items():
        class_id = result.loc[
            result["motif_id"].eq(motif_id),
            "occurrence_class_id",
        ].iloc[0]
        closed[motif_id] = not any(
            _strict_subgraph(pattern, pattern_sets[other])
            for other in occurrence_groups[class_id]
            if other != motif_id
        )
    motif_ids = list(pattern_sets)
    maximal = {}
    for motif_id, pattern in pattern_sets.items():
        maximal[motif_id] = not any(
            _strict_subgraph(pattern, pattern_sets[other])
            for other in motif_ids
            if other != motif_id
        )
    result["is_closed"] = result["motif_id"].map(closed)
    result["is_maximal"] = result["motif_id"].map(maximal)
    ranked = result.sort_values(
        [
            "occurrence_class_id",
            "node_count",
            "edge_count",
            "motif_id",
        ],
        ascending=[True, False, False, True],
    )
    representatives = ranked.groupby(
        "occurrence_class_id",
        sort=False,
    ).head(1)["motif_id"]
    result["is_occurrence_representative"] = result["motif_id"].isin(
        representatives
    )
    return result


def mine_frequent_motifs(
    graphs: list[nx.Graph],
    condition_labels: pd.DataFrame,
    support_fractions: list[float],
    primary_support_fraction: float,
    minimum_nodes: int,
    maximum_nodes: int,
    threads: int,
    expected_backend_version: str,
    include_redundancy: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    from fast_gspan import FastgSpan

    backend_version = importlib.metadata.version("fast-gspan")
    if backend_version != expected_backend_version:
        raise RuntimeError(
            f"Expected fast-gspan {expected_backend_version}, "
            f"found {backend_version}"
        )
    graph_ids = [str(graph.graph["graph_id"]) for graph in graphs]
    counts = support_counts(len(graphs), support_fractions)
    minimum_support = min(counts.values())
    miner = FastgSpan(
        min_support=minimum_support,
        min_num_vertices=minimum_nodes,
        max_num_vertices=maximum_nodes,
        num_threads=max(1, int(threads)),
    )
    backend_path = Path(miner._gbolt._gbolt_path)
    backend_binary_sha256 = hashlib.sha256(
        backend_path.read_bytes()
    ).hexdigest()
    raw = miner.run_from_graphs(graphs)
    inverse_labels = {
        int(row.gspan_label): str(row.condition)
        for row in condition_labels.itertuples(index=False)
    }
    rows = []
    for pattern in raw.itertuples(index=False):
        nodes, edges, motif_id = _canonical_pattern(
            list(pattern.vertices),
            list(pattern.edges),
            inverse_labels,
        )
        occurrence_indices = sorted(int(value) for value in pattern.graph_ids)
        occurrence_graph_ids = [graph_ids[index] for index in occurrence_indices]
        if int(pattern.support) != len(occurrence_graph_ids):
            raise RuntimeError("gSpan support and occurrence IDs disagree")
        row = {
            "motif_id": motif_id,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "motif_density": len(edges) / math.comb(len(nodes), 2),
            "is_tree": len(edges) == len(nodes) - 1,
            "is_clique": len(edges) == math.comb(len(nodes), 2),
            "node_labels": json.dumps(nodes, separators=(",", ":")),
            "edge_list": json.dumps(edges, separators=(",", ":")),
            "support_count": int(pattern.support),
            "support_fraction": int(pattern.support) / len(graphs),
            "graph_ids": json.dumps(
                occurrence_graph_ids,
                separators=(",", ":"),
            ),
            "gspan_pattern_id": int(pattern.pattern_id),
            "gspan_dfs_code": str(pattern.description),
        }
        for fraction, count in counts.items():
            row[f"support_{int(round(fraction * 100)):02d}pct"] = (
                int(pattern.support) >= count
            )
        row["primary_support"] = int(pattern.support) >= counts[
            float(primary_support_fraction)
        ]
        rows.append(row)
    motifs = pd.DataFrame(rows)
    if motifs.empty:
        raise RuntimeError("gSpan found no motifs at the minimum support")
    if motifs["motif_id"].duplicated().any():
        raise RuntimeError("gSpan returned duplicate canonical motifs")
    if include_redundancy:
        motifs = annotate_redundancy(motifs)
    motifs = motifs.sort_values(
        ["support_count", "node_count", "edge_count", "motif_id"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)
    metadata = {
        "backend": "fast_gspan",
        "backend_version": backend_version,
        "algorithm_backend": "gBolt gSpan",
        "backend_binary_sha256": backend_binary_sha256,
        "graph_count": len(graphs),
        "minimum_nodes": minimum_nodes,
        "maximum_nodes": maximum_nodes,
        "connected": True,
        "induced": False,
        "support_counts": {
            f"{fraction:.2f}": count
            for fraction, count in counts.items()
        },
        "primary_support_fraction": primary_support_fraction,
        "motif_count": len(motifs),
    }
    return motifs, metadata


def graphs_from_edge_masks(
    masks: np.ndarray,
    graph_ids: list[str],
    condition_labels: pd.DataFrame,
    universe: list[tuple[str, str]],
    edge_label: int = 1,
) -> list[nx.Graph]:
    if len(masks) != len(graph_ids):
        raise ValueError("Graph masks and identifiers differ in length")
    label_lookup = condition_labels.set_index("condition")[
        "gspan_label"
    ].to_dict()
    conditions = condition_labels.sort_values("gspan_label")[
        "condition"
    ].tolist()
    graphs = []
    for graph_id, mask in zip(graph_ids, masks, strict=True):
        graph = nx.Graph(graph_id=graph_id)
        graph.add_nodes_from(
            (
                condition,
                {
                    "label": int(label_lookup[condition]),
                    "condition": condition,
                },
            )
            for condition in conditions
        )
        graph.add_edges_from(
            (
                source,
                target,
                {"label": int(edge_label)},
            )
            for index, (source, target) in enumerate(universe)
            if int(mask) & (1 << index)
        )
        graphs.append(graph)
    return graphs


def graph_edge_masks(
    graphs: list[nx.Graph],
    edge_index: dict[tuple[str, str], int],
) -> np.ndarray:
    masks = np.zeros(len(graphs), dtype=np.uint64)
    for graph_index, graph in enumerate(graphs):
        value = np.uint64(0)
        for source, target in graph.edges():
            edge = tuple(sorted((str(source), str(target))))
            value |= np.uint64(1) << np.uint64(edge_index[edge])
        masks[graph_index] = value
    return masks


def motif_edge_masks(
    motifs: pd.DataFrame,
    edge_index: dict[tuple[str, str], int],
) -> np.ndarray:
    masks = np.zeros(len(motifs), dtype=np.uint64)
    for motif_index, edge_json in enumerate(motifs["edge_list"]):
        value = np.uint64(0)
        for source, target in json.loads(edge_json):
            edge = tuple(sorted((str(source), str(target))))
            value |= np.uint64(1) << np.uint64(edge_index[edge])
        masks[motif_index] = value
    return masks


def occurrence_matrix(
    graph_masks: np.ndarray,
    motif_masks: np.ndarray,
) -> np.ndarray:
    return (
        np.bitwise_and(graph_masks[:, np.newaxis], motif_masks[np.newaxis, :])
        == motif_masks[np.newaxis, :]
    )


def write_gspan_database(
    graphs: list[nx.Graph],
    path: Path,
) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    index_rows = []
    for graph_index, graph in enumerate(graphs):
        graph_id = str(graph.graph["graph_id"])
        nodes = sorted(
            graph.nodes(),
            key=lambda node: int(graph.nodes[node]["label"]),
        )
        local_ids = {node: index for index, node in enumerate(nodes)}
        lines.append(f"t # {graph_index}")
        for node in nodes:
            lines.append(
                f"v {local_ids[node]} {int(graph.nodes[node]['label'])}"
            )
        for source, target, attributes in sorted(
            graph.edges(data=True),
            key=lambda value: tuple(sorted((str(value[0]), str(value[1])))),
        ):
            lines.append(
                f"e {local_ids[source]} {local_ids[target]} "
                f"{int(attributes['label'])}"
            )
        index_rows.append(
            {
                "gspan_graph_index": graph_index,
                "graph_id": graph_id,
                "node_count": graph.number_of_nodes(),
                "edge_count": graph.number_of_edges(),
                "component_count": nx.number_connected_components(graph),
                "isolate_count": nx.number_of_isolates(graph),
                "largest_component_nodes": max(
                    len(component)
                    for component in nx.connected_components(graph)
                ),
                "triangle_count": sum(nx.triangles(graph).values()) // 3,
            }
        )
    lines.append("t # -1")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)
    return pd.DataFrame(index_rows)
