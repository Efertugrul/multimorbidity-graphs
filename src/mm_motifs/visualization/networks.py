from __future__ import annotations

import os
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


def _save(figure: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    figure.savefig(temporary, dpi=dpi, bbox_inches="tight", format="png")
    plt.close(figure)
    os.replace(temporary, path)


def draw_population_graph(
    graph: nx.Graph,
    path: Path,
    settings: dict[str, Any],
) -> None:
    figure, axis = plt.subplots(
        figsize=(float(settings["width"]), float(settings["height"]))
    )
    nodes = sorted(graph.nodes())
    if nodes:
        positions_array = nx.circular_layout(nodes)
        positions = {node: positions_array[node] for node in nodes}
        prevalences = [graph.nodes[node]["prevalence"] for node in nodes]
        node_sizes = [900 + 5000 * value for value in prevalences]
        node_color = "#b45309" if graph.graph["ses_category"] == "lower" else "#2563eb"
        edge_widths = [
            1.0 + 12.0 * graph.edges[edge]["association"] for edge in graph.edges()
        ]
        edge_colors = [graph.edges[edge]["association"] for edge in graph.edges()]
        nx.draw_networkx_nodes(
            graph,
            positions,
            node_size=node_sizes,
            node_color=node_color,
            alpha=0.84,
            linewidths=1,
            edgecolors="white",
            ax=axis,
        )
        if graph.number_of_edges():
            nx.draw_networkx_edges(
                graph,
                positions,
                width=edge_widths,
                edge_color=edge_colors,
                edge_cmap=plt.cm.viridis,
                edge_vmin=0,
                edge_vmax=max(0.25, max(edge_colors)),
                alpha=0.8,
                ax=axis,
            )
        labels = {
            node: "\n".join(
                textwrap.wrap(graph.nodes[node]["label"], width=16)
            )
            for node in nodes
        }
        nx.draw_networkx_labels(graph, positions, labels=labels, font_size=7, ax=axis)
    axis.set_title(
        f"{graph.graph['geography']} — {graph.graph['ses_category']} SES\n"
        f"{graph.number_of_nodes()} conditions, {graph.number_of_edges()} associations"
    )
    axis.axis("off")
    _save(figure, path, int(settings["dpi"]))


def draw_similarity_heatmap(
    similarities: pd.DataFrame,
    graph_order: list[str],
    path: Path,
    dpi: int,
) -> None:
    matrix = similarities.pivot(
        index="graph_id_a",
        columns="graph_id_b",
        values="edge_jaccard",
    ).reindex(index=graph_order, columns=graph_order)
    figure, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(matrix.to_numpy(), vmin=0, vmax=1, cmap="magma")
    axis.set_xticks(np.arange(len(graph_order)), graph_order, rotation=60, ha="right")
    axis.set_yticks(np.arange(len(graph_order)), graph_order)
    axis.set_title("Population graph edge-set Jaccard similarity")
    figure.colorbar(image, ax=axis, label="Jaccard similarity")
    figure.tight_layout()
    _save(figure, path, dpi)
