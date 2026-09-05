from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def draw_point_phi_vs_selection_stability(
    bootstrap: pd.DataFrame,
    path: Path,
    point_threshold: float,
    stability_threshold: float,
    dpi: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = bootstrap[
        bootstrap["pair_eligible"]
        & bootstrap["point_phi"].ge(point_threshold)
    ].copy()
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.scatter(
        selected["point_phi"],
        selected["p_phi_ge_012"],
        s=13,
        alpha=0.28,
        color="#2563eb",
        linewidths=0,
        label="Raw phi ≥ 0.12 edges",
    )
    bin_edges = np.arange(
        point_threshold,
        selected["point_phi"].max() + 0.011,
        0.01,
    )
    selected["point_phi_bin"] = pd.cut(
        selected["point_phi"],
        bins=bin_edges,
        include_lowest=True,
    )
    medians = selected.groupby(
        "point_phi_bin",
        observed=True,
    ).agg(
        point_phi=("point_phi", "median"),
        selection_stability=("p_phi_ge_012", "median"),
    )
    axis.plot(
        medians["point_phi"],
        medians["selection_stability"],
        color="#111827",
        marker="o",
        markersize=3,
        linewidth=1.5,
        label="0.01-band median",
    )
    axis.axhline(
        stability_threshold,
        color="#b91c1c",
        linestyle="--",
        linewidth=1.8,
        label=f"Stability gate = {stability_threshold:.2f}",
    )
    axis.axvline(
        0.15,
        color="#6b7280",
        linestyle=":",
        linewidth=1.5,
        label="Descriptive phi = 0.15",
    )
    axis.set_xlabel("Point survey-weighted phi")
    axis.set_ylabel("Bootstrap P(phi ≥ 0.12)")
    axis.set_ylim(-0.02, 1.02)
    axis.set_title("Phase 2.6 threshold-location diagnostic")
    axis.legend(frameon=False, loc="lower right")
    axis.text(
        0.01,
        -0.16,
        (
            f"Each point is one of the {len(selected):,} raw edges. The 0.15 "
            "line is descriptive; it does not estimate P(phi ≥ 0.15)."
        ),
        transform=axis.transAxes,
        fontsize=9,
        color="#4b5563",
    )
    figure.tight_layout()
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def draw_threshold_trajectories(
    statistics: pd.DataFrame,
    path: Path,
    dpi: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pivot = statistics.pivot(
        index="graph_id",
        columns="threshold",
        values="edge_count",
    ).sort_index(axis=1)
    figure, axis = plt.subplots(figsize=(8, 6))
    for _, values in pivot.iterrows():
        axis.plot(
            pivot.columns,
            values,
            color="#6b7280",
            alpha=0.18,
            linewidth=0.8,
        )
    axis.plot(
        pivot.columns,
        pivot.median(axis=0),
        color="#b91c1c",
        marker="o",
        linewidth=2.5,
        label="Median",
    )
    axis.set_xticks(list(pivot.columns))
    axis.set_xlabel("Weighted phi threshold")
    axis.set_ylabel("Edges per graph")
    axis.set_title("Threshold-neighborhood edge-count continuity")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def draw_selection_stability(
    bootstrap: pd.DataFrame,
    path: Path,
    point_threshold: float,
    stability_threshold: float,
    dpi: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = bootstrap[
        bootstrap["pair_eligible"]
        & bootstrap["point_phi"].ge(point_threshold)
    ]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(
        selected["p_phi_ge_012"],
        bins=20,
        color="#2563eb",
        edgecolor="white",
    )
    axis.axvline(
        stability_threshold,
        color="#b91c1c",
        linestyle="--",
        linewidth=2,
        label=f"Stable ≥ {stability_threshold:.2f}",
    )
    axis.set_xlabel("Bootstrap P(phi ≥ 0.12)")
    axis.set_ylabel("Point-selected edges")
    axis.set_title("Phi 0.12 edge-selection stability")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=dpi)
    plt.close(figure)
