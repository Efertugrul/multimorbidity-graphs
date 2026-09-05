from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def draw_motif_support_spectrum(
    shape_summary: pd.DataFrame,
    path: Path,
    dpi: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pivot = shape_summary.pivot(
        index="node_count",
        columns="support_level",
        values="motif_count",
    ).fillna(0)
    figure, axis = plt.subplots(figsize=(9, 6))
    x_values = np.arange(len(pivot.index))
    width = 0.24
    colors = ["#2563eb", "#6b7280", "#b91c1c"]
    for index, column in enumerate(pivot.columns):
        axis.bar(
            x_values + (index - 1) * width,
            pivot[column],
            width=width,
            label=column.replace("support_", "").replace("pct", "%"),
            color=colors[index % len(colors)],
        )
    axis.set_xticks(x_values)
    axis.set_xticklabels(pivot.index)
    axis.set_xlabel("Motif nodes")
    axis.set_ylabel("Frequent connected motifs")
    axis.set_title("Pooled gSpan motif-support spectrum")
    axis.legend(title="Minimum pooled support", frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def draw_motif_bootstrap_stability(
    stability: pd.DataFrame,
    path: Path,
    primary_support_percent: int,
    dpi: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    probability = f"p_boot_support_{primary_support_percent:02d}pct"
    primary = stability[
        stability["baseline_support_count"].ge(
            stability.attrs["primary_support_count"]
        )
    ]
    figure, axis = plt.subplots(figsize=(9, 6))
    for node_count, frame in primary.groupby("node_count", sort=True):
        axis.scatter(
            frame["baseline_support_count"],
            frame[probability],
            s=18,
            alpha=0.45,
            label=f"{node_count} nodes",
        )
    axis.axhline(0.80, color="#6b7280", linestyle="--", linewidth=1.5)
    axis.axhline(0.90, color="#b91c1c", linestyle=":", linewidth=1.5)
    axis.set_xlabel("Baseline pooled support count")
    axis.set_ylabel(
        f"Bootstrap P(support ≥ {primary_support_percent}%)"
    )
    axis.set_ylim(-0.02, 1.02)
    axis.set_title("Fixed-vocabulary motif support stability")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def draw_discovery_set_stability(
    discovery: pd.DataFrame,
    path: Path,
    dpi: int,
    title: str,
    metric: str,
    y_label: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fractions = sorted(discovery["support_fraction"].unique())
    values = [
        discovery.loc[
            discovery["support_fraction"].eq(fraction),
            metric,
        ]
        for fraction in fractions
    ]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.boxplot(
        values,
        tick_labels=[f"{fraction:.0%}" for fraction in fractions],
        showfliers=False,
    )
    axis.set_xlabel("Minimum pooled support")
    axis.set_ylabel(y_label)
    axis.set_ylim(0, 1.02)
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def draw_density_null(
    density_null: pd.DataFrame,
    path: Path,
    dpi: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11, 5))
    for axis, (null_model, frame) in zip(
        axes,
        density_null.groupby("null_model", sort=True),
        strict=True,
    ):
        axis.scatter(
            frame["null_support_mean"],
            frame["observed_support_count"],
            s=13,
            alpha=0.35,
            color="#2563eb",
        )
        maximum = max(
            frame["null_support_mean"].max(),
            frame["observed_support_count"].max(),
        )
        axis.plot([0, maximum], [0, maximum], color="#b91c1c", linestyle="--")
        axis.set_xlabel("Null mean support count")
        axis.set_ylabel("Observed support count")
        axis.set_title(null_model.replace("_", " "))
    figure.suptitle("Motif support relative to density-conditioned nulls")
    figure.tight_layout()
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def draw_ses_permutation(
    ses_results: pd.DataFrame,
    path: Path,
    dpi: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    p_values = ses_results["permutation_max_t_p_value"].clip(lower=1e-12)
    figure, axis = plt.subplots(figsize=(9, 6))
    scatter = axis.scatter(
        ses_results["paired_density_residual_difference"],
        -np.log10(p_values),
        c=ses_results["node_count"],
        cmap="viridis",
        s=20,
        alpha=0.55,
    )
    axis.axvline(0, color="#6b7280", linewidth=1)
    axis.axhline(
        -np.log10(0.05),
        color="#b91c1c",
        linestyle="--",
        linewidth=1,
    )
    axis.set_xlabel("Lower − higher density-adjusted motif occurrence")
    axis.set_ylabel("−log10 paired maxT permutation p-value")
    axis.set_title("Exploratory paired SES motif comparison")
    colorbar = figure.colorbar(scatter, ax=axis)
    colorbar.set_label("Motif nodes")
    figure.tight_layout()
    figure.savefig(path, dpi=dpi)
    plt.close(figure)
