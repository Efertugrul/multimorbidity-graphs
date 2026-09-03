from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from mm_motifs.config import ProjectConfig
from mm_motifs.data.load import read_selected
from mm_motifs.features.conditions import (
    condition_names,
    condition_sources,
    derive_conditions,
)
from mm_motifs.features.populations import (
    make_graph_id,
    population_registry,
    state_name,
)
from mm_motifs.features.socioeconomic import derive_ses, ses_source
from mm_motifs.graphs.construct import construct_graph, node_prevalence_table
from mm_motifs.graphs.qc import (
    graph_statistics,
    pairwise_similarity,
    viability_report,
)
from mm_motifs.graphs.serialize import write_graph
from mm_motifs.runtime import (
    configure_logging,
    interim_directory,
    manifest,
    output_directory,
    raw_xpt_path,
    write_csv,
    write_json,
)
from mm_motifs.statistics.disease_association import estimate_edge_table
from mm_motifs.visualization.networks import (
    draw_population_graph,
    draw_similarity_heatmap,
)


def _selected_columns(config: ProjectConfig) -> list[str]:
    design = config.analysis["survey_design"]
    columns = list(design.values())
    columns.extend(condition_sources(config.conditions, config.year))
    for definition_name in config.analysis["socioeconomic"]["definitions"]:
        columns.append(ses_source(config.analysis, definition_name, config.year))
    return list(dict.fromkeys(columns))


def _harmonize(raw: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    design = config.analysis["survey_design"]
    states = [int(value) for value in config.analysis["prototype"]["states"]]
    state_values = pd.to_numeric(raw[design["state"]], errors="coerce")
    raw = raw.loc[state_values.isin(states)].copy()
    state_values = pd.to_numeric(raw[design["state"]], errors="coerce").astype("Int64")
    derived = derive_conditions(raw, config.conditions, config.year)
    ses = derive_ses(raw, config.analysis, config.year)
    frame = pd.DataFrame(index=raw.index)
    frame["state_code"] = state_values
    frame["geography"] = state_values.map(
        lambda value: state_name(int(value)) if pd.notna(value) else pd.NA
    )
    frame["ses_category"] = ses
    frame["survey_weight"] = pd.to_numeric(raw[design["weight"]], errors="coerce")
    frame["survey_strata"] = pd.to_numeric(raw[design["strata"]], errors="coerce")
    frame["survey_psu"] = pd.to_numeric(raw[design["psu"]], errors="coerce")
    frame["age_group"] = pd.to_numeric(raw[design["age_group"]], errors="coerce")
    frame["sex"] = pd.to_numeric(raw[design["sex"]], errors="coerce")
    for name in derived:
        frame[name] = derived[name]
    frame["graph_id"] = [
        make_graph_id(config.year, int(state), str(category))
        if pd.notna(state) and pd.notna(category)
        else pd.NA
        for state, category in zip(
            frame["state_code"], frame["ses_category"], strict=True
        )
    ]
    return frame.reset_index(drop=True)


def _persist_harmonized(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def run_prototype(
    config: ProjectConfig,
    verbose: bool = False,
    persist_harmonized: bool | None = None,
) -> Path:
    source = raw_xpt_path(config)
    if not source.exists():
        raise FileNotFoundError(f"BRFSS XPT file not found: {source}")
    output = output_directory(config, "prototype")
    logger = configure_logging(output / "run.log", verbose)
    logger.info("Loading selected BRFSS variables for Phase 2")
    raw = read_selected(source, _selected_columns(config))
    harmonized = _harmonize(raw, config)
    del raw
    logger.info("Harmonized %s respondents in prototype states", len(harmonized))

    should_persist = (
        bool(config.analysis["prototype"]["persist_harmonized"])
        if persist_harmonized is None
        else persist_harmonized
    )
    harmonized_path: Path | None = None
    if should_persist:
        harmonized_path = interim_directory(config) / "prototype_harmonized.parquet"
        _persist_harmonized(harmonized, harmonized_path)

    registry = population_registry(harmonized, config.analysis, config.year)
    names = condition_names(config.conditions)
    edge_criteria = config.graph["edge_criteria"]
    graphs = {}
    edge_tables: dict[str, pd.DataFrame] = {}
    node_tables = []
    graph_index: list[dict[str, Any]] = []

    for population_row in registry.itertuples(index=False):
        if population_row.eligibility_status != "eligible":
            continue
        population = harmonized[harmonized["graph_id"] == population_row.graph_id]
        edges = estimate_edge_table(
            population,
            names,
            population_row.graph_id,
            edge_criteria,
        )
        nodes = node_prevalence_table(population, names, population_row.graph_id)
        graph = construct_graph(
            graph_id=population_row.graph_id,
            year=config.year,
            state_code=population_row.geography_code,
            geography=population_row.geography,
            ses_category=population_row.ses_category,
            population=population,
            node_table=nodes,
            edge_table=edges,
            graph_config=config.graph,
        )
        graphml_path, json_path = write_graph(graph, output / "graphs")
        draw_population_graph(
            graph,
            output / "figures" / f"{population_row.graph_id}.png",
            config.graph["visualization"],
        )
        graphs[population_row.graph_id] = graph
        edge_tables[population_row.graph_id] = edges
        node_tables.append(nodes)
        graph_index.append(
            {
                "graph_id": population_row.graph_id,
                "graphml_path": str(graphml_path.relative_to(config.project_root)),
                "json_path": str(json_path.relative_to(config.project_root)),
            }
        )
        logger.info(
            "Built %s with %s nodes and %s edges",
            population_row.graph_id,
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )

    if not graphs:
        raise ValueError("No prototype populations met the configured minimum sample size")

    all_edges = pd.concat(edge_tables.values(), ignore_index=True)
    all_nodes = pd.concat(node_tables, ignore_index=True)
    statistics = pd.DataFrame(
        [graph_statistics(graph) for graph in graphs.values()]
    ).sort_values("graph_id")
    similarities = pairwise_similarity(graphs, edge_tables, names)
    report = viability_report(
        graphs,
        statistics,
        similarities,
        config.graph["viability"],
    )

    graph_paths = pd.DataFrame(graph_index)
    registry = registry.merge(graph_paths, on="graph_id", how="left")
    write_csv(registry, output / "population_registry.csv")
    write_csv(all_edges, output / "edge_table.csv")
    write_csv(all_nodes, output / "disease_prevalence_by_graph.csv")
    write_csv(statistics, output / "graph_statistics.csv")
    write_csv(similarities, output / "structural_similarity.csv")
    write_json(report, output / "viability_report.json")
    draw_similarity_heatmap(
        similarities,
        list(graphs),
        output / "figures" / "edge_jaccard_heatmap.png",
        int(config.graph["visualization"]["dpi"]),
    )

    write_json(
        manifest(
            config,
            "prototype",
            {
                "source_file": str(source.relative_to(config.project_root)),
                "prototype_states": [
                    {
                        "code": int(code),
                        "name": state_name(int(code)),
                    }
                    for code in config.analysis["prototype"]["states"]
                ],
                "eligible_graphs": len(graphs),
                "condition_count": len(names),
                "ses_definition": config.active_ses,
                "harmonized_dataset": (
                    str(harmonized_path.relative_to(config.project_root))
                    if harmonized_path
                    else None
                ),
                "unclassified_ses_respondents": int(
                    harmonized["ses_category"].isna().sum()
                ),
                "inference_scope": (
                    "descriptive survey-weighted phi; strata and PSU retained but "
                    "not used for Phase 2 variance inference"
                ),
                "phase_boundary": "Stopped before frequent subgraph mining",
            },
        ),
        output / "manifest.json",
    )
    logger.info("Prototype complete: %s", output)
    return output
