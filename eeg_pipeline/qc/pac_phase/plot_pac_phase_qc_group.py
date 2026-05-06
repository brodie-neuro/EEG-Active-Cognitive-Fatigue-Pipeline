"""Group-level PAC phase QC plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    1: "#1E88E5",
    5: "#E53935",
    "clear": "#2E7D32",
    "weak": "#F9A825",
    "indeterminate": "#B71C1C",
}


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.2)


def _paired_plot(ax, df: pd.DataFrame, band: str, metric: str, ylabel: str):
    sub = df[df["band"] == band]
    wide = sub.pivot_table(index="subject", columns="block", values=metric, aggfunc="first")
    if 1 not in wide.columns or 5 not in wide.columns:
        ax.text(0.5, 0.5, "Missing paired blocks", transform=ax.transAxes, ha="center", va="center")
        return
    for subject, row in wide.iterrows():
        if not np.isfinite(row.get(1, np.nan)) or not np.isfinite(row.get(5, np.nan)):
            continue
        ax.plot([1, 5], [row[1], row[5]], color="#90A4AE", lw=1.0, alpha=0.75)
        ax.scatter([1, 5], [row[1], row[5]], color=[COLORS[1], COLORS[5]], s=28, zorder=3)
    ax.set_xticks([1, 5])
    ax.set_xticklabels(["Block 1", "Block 5"])
    ax.set_ylabel(ylabel)
    ax.set_title(f"{band}: {ylabel}")
    _style(ax)


def plot_group_pac_phase_qc(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Create concise group-level PAC phase QC plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if df.empty:
        return paths

    bands = list(df["band"].dropna().unique())

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0, 0]
    status_counts = (
        df.groupby(["band", "block", "qc_status"]).size().reset_index(name="n")
    )
    labels = []
    bottoms = {}
    x = np.arange(len(status_counts[["band", "block"]].drop_duplicates()))
    groups = status_counts[["band", "block"]].drop_duplicates().reset_index(drop=True)
    for idx, row in groups.iterrows():
        labels.append(f"{row['band']}\nB{int(row['block'])}")
        bottoms[idx] = 0
    for status in ["clear", "weak", "indeterminate"]:
        heights = []
        for _, group_row in groups.iterrows():
            hit = status_counts[
                (status_counts["band"] == group_row["band"])
                & (status_counts["block"] == group_row["block"])
                & (status_counts["qc_status"] == status)
            ]
            heights.append(int(hit["n"].iloc[0]) if not hit.empty else 0)
        ax.bar(x, heights, bottom=[bottoms[i] for i in range(len(x))], color=COLORS[status], label=status)
        for i, h in enumerate(heights):
            bottoms[i] += h
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("participant-block count")
    ax.set_title("QC status counts")
    ax.legend(fontsize=8)
    _style(ax)

    ax = axes[0, 1]
    valid_slopes = df.drop_duplicates(["subject", "block"])["aperiodic_slope"].dropna()
    ax.hist(valid_slopes, bins=min(12, max(4, len(valid_slopes))), color="#607D8B", edgecolor="white")
    ax.set_xlabel("aperiodic slope")
    ax.set_ylabel("count")
    ax.set_title("Aperiodic slope distribution")
    _style(ax)

    if bands:
        _paired_plot(axes[1, 0], df, bands[0], "centre_of_mass_hz", "CoM (Hz)")
        _paired_plot(axes[1, 1], df, bands[0], "positive_residual_area", "positive residual area")
    fig.suptitle("PAC phase QC group summary", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = output_dir / "pac_phase_qc_group_summary.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths.append(out)

    for band in bands[1:]:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        _paired_plot(axes[0], df, band, "centre_of_mass_hz", "CoM (Hz)")
        _paired_plot(axes[1], df, band, "positive_residual_area", "positive residual area")
        fig.suptitle(f"PAC phase QC paired metrics: {band}", fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        out = output_dir / f"pac_phase_qc_group_{band}.png"
        fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        paths.append(out)

    return paths
