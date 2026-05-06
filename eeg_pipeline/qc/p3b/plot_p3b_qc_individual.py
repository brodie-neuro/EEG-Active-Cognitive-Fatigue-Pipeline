"""Individual P3b QC plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd


COLORS = {
    1: "#1E88E5",
    5: "#E53935",
    "window": "#4CAF50",
    "zero": "#777777",
}


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.18)


def _plot_heatmap(ax, payload, block: int, p3b_window: tuple[float, float], plot_window: tuple[float, float]):
    times_ms = payload["times"] * 1000.0
    mask = (payload["times"] >= plot_window[0]) & (payload["times"] <= plot_window[1])
    data = payload["roi_trials"][:, mask]
    if data.size == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        return
    vmax = max(abs(np.nanpercentile(data, 5)), abs(np.nanpercentile(data, 95)), 1e-6)
    im = ax.imshow(
        data,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        extent=[times_ms[mask][0], times_ms[mask][-1], data.shape[0], 0],
    )
    ax.axvline(0, color=COLORS["zero"], lw=0.8, ls=":")
    ax.axvline(p3b_window[0] * 1000.0, color="k", lw=0.9, ls="--")
    ax.axvline(p3b_window[1] * 1000.0, color="k", lw=0.9, ls="--")
    ax.set_title(f"Block {block}: ROI single trials")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Trial")
    return im


def _topomap_data(epochs, p3b_window: tuple[float, float]):
    evoked = epochs.average().copy().pick("eeg")
    mask = (evoked.times >= p3b_window[0]) & (evoked.times <= p3b_window[1])
    data = evoked.data[:, mask].mean(axis=1) * 1e6
    return evoked, data


def plot_subject_p3b_qc(
    subject: str,
    subject_index: int,
    epochs_by_block: dict[int, object],
    payload_by_block: dict[int, dict],
    metrics_df: pd.DataFrame,
    roi_channels: list[str],
    p3b_window: tuple[float, float],
    plot_window: tuple[float, float],
    output_dir: Path,
) -> Path:
    """Save one P3b QC dashboard for one participant."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(17, 12))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.05, 1.0])

    ax_erp = fig.add_subplot(gs[0, :2])
    ax_text = fig.add_subplot(gs[0, 2])
    ax_heat1 = fig.add_subplot(gs[1, 0])
    ax_heat5 = fig.add_subplot(gs[1, 1])
    ax_channels = fig.add_subplot(gs[1, 2])
    ax_topo1 = fig.add_subplot(gs[2, 0])
    ax_topo5 = fig.add_subplot(gs[2, 1])
    ax_topod = fig.add_subplot(gs[2, 2])

    for block, payload in sorted(payload_by_block.items()):
        times_ms = payload["times"] * 1000.0
        mask = (payload["times"] >= plot_window[0]) & (payload["times"] <= plot_window[1])
        row = metrics_df[(metrics_df["subject"] == subject) & (metrics_df["block"] == block)].iloc[0]
        label = (
            f"B{block} n={int(row['n_epochs'])}, "
            f"mean={row['p3b_mean_uV']:.2f} uV, "
            f"FAL={row['p3b_fractional_area_latency_ms']:.0f} ms"
        )
        ax_erp.plot(times_ms[mask], payload["roi_erp"][mask], color=COLORS.get(block, "#607D8B"), lw=2.0, label=label)

    ax_erp.axvspan(p3b_window[0] * 1000.0, p3b_window[1] * 1000.0, color=COLORS["window"], alpha=0.13)
    ax_erp.axhline(0, color=COLORS["zero"], lw=0.8)
    ax_erp.axvline(0, color=COLORS["zero"], lw=0.8, ls="--")
    ax_erp.set_xlim(plot_window[0] * 1000.0, plot_window[1] * 1000.0)
    ax_erp.set_xlabel("Time (ms)")
    ax_erp.set_ylabel("Amplitude (uV)")
    ax_erp.set_title("P3b ROI ERP: Block 1 vs Block 5")
    ax_erp.legend(fontsize=8, loc="best")
    _style(ax_erp)

    lines = []
    for _, row in metrics_df[metrics_df["subject"] == subject].sort_values("block").iterrows():
        lines.append(
            f"B{int(row['block'])}: {row['qc_status']}\n"
            f"  mean={row['p3b_mean_uV']:.2f} uV\n"
            f"  FAL={row['p3b_fractional_area_latency_ms']:.0f} ms\n"
            f"  peak={row['p3b_peak_uV']:.2f} uV @ {row['p3b_peak_latency_ms']:.0f} ms\n"
            f"  ROI={row['roi_channels']}"
        )
        if isinstance(row.get("missing_roi_channels"), str) and row.get("missing_roi_channels"):
            lines.append(f"  missing={row['missing_roi_channels']}")
    ax_text.axis("off")
    ax_text.text(
        0.02,
        0.98,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#B0BEC5"),
    )

    if 1 in payload_by_block:
        im = _plot_heatmap(ax_heat1, payload_by_block[1], 1, p3b_window, plot_window)
        if im is not None:
            fig.colorbar(im, ax=ax_heat1, fraction=0.046, pad=0.04, label="uV")
    else:
        ax_heat1.text(0.5, 0.5, "No Block 1", transform=ax_heat1.transAxes, ha="center", va="center")
    if 5 in payload_by_block:
        im = _plot_heatmap(ax_heat5, payload_by_block[5], 5, p3b_window, plot_window)
        if im is not None:
            fig.colorbar(im, ax=ax_heat5, fraction=0.046, pad=0.04, label="uV")
    else:
        ax_heat5.text(0.5, 0.5, "No Block 5", transform=ax_heat5.transAxes, ha="center", va="center")

    for block, payload in sorted(payload_by_block.items()):
        times_ms = payload["times"] * 1000.0
        mask = (payload["times"] >= plot_window[0]) & (payload["times"] <= plot_window[1])
        channel_data = payload["channel_data_uV"]
        for idx, ch in enumerate(payload["available_channels"]):
            erp = channel_data[:, idx, :].mean(axis=0)
            ax_channels.plot(times_ms[mask], erp[mask], color=COLORS.get(block, "#607D8B"), lw=1.1, alpha=0.65)
    ax_channels.axvspan(p3b_window[0] * 1000.0, p3b_window[1] * 1000.0, color=COLORS["window"], alpha=0.10)
    ax_channels.axhline(0, color=COLORS["zero"], lw=0.8)
    ax_channels.axvline(0, color=COLORS["zero"], lw=0.8, ls="--")
    ax_channels.set_xlim(plot_window[0] * 1000.0, plot_window[1] * 1000.0)
    ax_channels.set_title("Planned ROI channel ERPs")
    ax_channels.set_xlabel("Time (ms)")
    ax_channels.set_ylabel("Amplitude (uV)")
    _style(ax_channels)

    topo_payload = {}
    for block, ax in [(1, ax_topo1), (5, ax_topo5)]:
        epochs = epochs_by_block.get(block)
        if epochs is None:
            ax.text(0.5, 0.5, f"No Block {block}", transform=ax.transAxes, ha="center", va="center")
            ax.axis("off")
            continue
        evoked, data = _topomap_data(epochs, p3b_window)
        topo_payload[block] = (evoked, data)
        mne.viz.plot_topomap(data, evoked.info, axes=ax, show=False, cmap="RdBu_r", contours=4, sensors=True)
        ax.set_title(f"Block {block} topography\n{int(p3b_window[0] * 1000)}-{int(p3b_window[1] * 1000)} ms")

    if 1 in topo_payload and 5 in topo_payload:
        evoked = topo_payload[5][0]
        data = topo_payload[5][1] - topo_payload[1][1]
        mne.viz.plot_topomap(data, evoked.info, axes=ax_topod, show=False, cmap="RdBu_r", contours=4, sensors=True)
        ax_topod.set_title("Delta topography\nB5 - B1")
    else:
        ax_topod.text(0.5, 0.5, "Delta unavailable", transform=ax_topod.transAxes, ha="center", va="center")
        ax_topod.axis("off")

    fig.suptitle(f"{subject}: P3b QC dashboard", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    out = output_dir / f"{subject_index:02d}_{subject}_p3b_qc.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out
