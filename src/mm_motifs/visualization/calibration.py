from __future__ import annotations

import inspect
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def _save(figure: plt.Figure, path: Path, dpi: int = 160) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    figure.savefig(temporary, dpi=dpi, bbox_inches="tight", format="png")
    plt.close(figure)
    os.replace(temporary, path)


def _boxplot_labels(axis: plt.Axes, labels: list[str]) -> dict[str, list[str]]:
    parameter = (
        "tick_labels"
        if "tick_labels" in inspect.signature(axis.boxplot).parameters
        else "labels"
    )
    return {parameter: labels}


def draw_rule_density_distributions(statistics: pd.DataFrame, path: Path) -> None:
    groups = []
    labels = []
    for (ses_definition, rule_id), frame in statistics.groupby(
        ["ses_definition", "rule_id"],
        sort=False,
    ):
        groups.append(frame["density"].to_numpy())
        labels.append(f"{ses_definition}\n{rule_id}")
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.boxplot(groups, showfliers=True, **_boxplot_labels(axis, labels))
    axis.axhline(20 / 45, color="black", linestyle="--", linewidth=1)
    axis.set_ylabel("Graph density")
    axis.set_title("Phase 2.5 density by SES definition and edge rule")
    axis.tick_params(axis="x", labelrotation=25)
    _save(figure, path)


def draw_similarity_distributions(similarities: pd.DataFrame, path: Path) -> None:
    groups = []
    labels = []
    for comparison_type, frame in similarities.groupby(
        "comparison_type",
        sort=False,
    ):
        groups.append(frame["edge_jaccard"].to_numpy())
        labels.append(comparison_type.replace("_", "\n"))
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.boxplot(groups, showfliers=True, **_boxplot_labels(axis, labels))
    axis.set_ylim(0, 1)
    axis.set_ylabel("Edge-set Jaccard similarity")
    axis.set_title("Graph similarity comparison")
    _save(figure, path)
