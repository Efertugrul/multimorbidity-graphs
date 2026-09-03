from __future__ import annotations

import os
from pathlib import Path

import networkx as nx

from mm_motifs.runtime import write_json


def write_graph(graph: nx.Graph, directory: Path) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    graph_id = str(graph.graph["graph_id"])
    graphml_path = directory / f"{graph_id}.graphml"
    temporary = graphml_path.with_suffix(".graphml.tmp")
    nx.write_graphml(graph, temporary)
    os.replace(temporary, graphml_path)

    json_path = directory / f"{graph_id}.json"
    write_json(nx.node_link_data(graph, edges="edges"), json_path)
    return graphml_path, json_path
