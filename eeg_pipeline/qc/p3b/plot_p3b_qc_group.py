"""Group-level P3b QC plots."""

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
    "review": "#B71C1C",
}


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.2)


def _paired(ax, df: pd.DataFrame, metric: str, ylabel: str):
    wide = df.pivot_table(index="subject", columns="block", values=metric, aggfunc="first")
    if 1 not in wide.columns or 5 not in wide.columns:
        ax.text(0.5, 0.5, "Missing paired blocks", transform=ax.transAxes, ha="center", va="center")
        return
    for _, row in wide.iterrows():
        if not np.isfinite(row.get(1, np.nan)) or not np.isfinite(row.get(5, np.nan)):
            continue
        ax.plot([1, 5], [row[1], row[5]], color="#90A4AE", lw=1.0, alpha=0.75)
        ax.scatter([1, 5], [row[1], row[5]], color=[COLORS[1], COLORS[5]], s=28, zorder=3)
    ax.set_xticks([1, 5])
    ax.set_xticklabels(["Block 1", "Block 5"])
    ax.set_ylabel(ylabel)
    _style(ax)


def plot_group_p3b_qc(df: pd.DataFrame, output_dir: Path) -> Path | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    counts = df.groupby(["block", "qc_status"]).size().reset_index(name="n")
    x_groups = sorted(df["block"].dropna().unique())
    x = np.arange(len(x_groups))
    bottoms = np.zeros(len(x_groups))
    for status in ["clear", "weak", "review"]:
        vals = []
        for block in x_groups:
            hit = counts[(counts["block"] == block) & (counts["qc_status"] == status)]
            vals.append(int(hit["n"].iloc[0]) if not hit.empty else 0)
        axes[0, 0].bar(x, vals, bottom=bottoms, color=COLORS[status], label=status)
        bottoms += np.asarray(vals)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([f"B{int(b)}" for b in x_groups])
    axes[0, 0].set_ylabel("participant-block count")
    axes[0, 0].set_title("P3b QC status counts")
    axes[0, 0].legend(fontsize=8)
    _style(axes[0, 0])

    _paired(axes[0, 1], df, "p3b_mean_uV", "P3b mean (uV)")
    axes[0, 1].set_title("P3b mean amplitude")

    _paired(axes[1, 0], df, "p3b_fractional_area_latency_ms", "FAL (ms)")
    axes[1, 0].set_title("Fractional area latency")

    _paired(axes[1, 1], df, "n_epochs", "N epochs")
    axes[1, 1].set_title("Retained target epochs")

    fig.suptitle("P3b QC group summary", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = output_dir / "p3b_qc_group_summary.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out
