"""Individual PAC phase QC plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "block1": "#1E88E5",
    "block5": "#E53935",
    "theta": "#2E7D32",
    "alpha": "#6A1B9A",
    "fit": "#263238",
}


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.22)


def _shade_bands(ax, bands):
    for name, (lo, hi) in bands.items():
        ax.axvspan(lo, hi, color=COLORS.get(name, "#607D8B"), alpha=0.10)
        ax.text(
            (lo + hi) / 2,
            0.98,
            name,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
            color=COLORS.get(name, "#607D8B"),
            fontweight="bold",
        )


def _metrics_text(rows: pd.DataFrame) -> str:
    parts = []
    for _, row in rows.sort_values("band").iterrows():
        band = row["band"]
        status = row["qc_status"]
        com = row["centre_of_mass_hz"]
        peak = row["peak_frequency_hz"]
        area = row["positive_residual_area"]
        if np.isfinite(com):
            parts.append(f"{band}: {status}, CoM={com:.2f} Hz, peak={peak:.2f} Hz, area={area:.3f}")
        else:
            parts.append(f"{band}: {status}, CoM=NA, peak=NA")
    return "\n".join(parts)


def _metrics_panel(ax, block: int, spectrum, rows: pd.DataFrame):
    ax.axis("off")
    lines = [
        f"Block {block}",
        "solid line = CoM",
        "dotted line = peak",
        "",
        f"slope: {spectrum.aperiodic_slope:.3f}",
        f"intercept: {spectrum.aperiodic_intercept:.3f}",
        f"fit: {spectrum.aperiodic_method}",
        "",
    ]
    for _, row in rows.sort_values("band").iterrows():
        band = row["band"]
        status = row["qc_status"]
        com = row["centre_of_mass_hz"]
        peak = row["peak_frequency_hz"]
        area = row["positive_residual_area"]
        lines.append(f"{band.upper()} [{status}]")
        if np.isfinite(com):
            lines.append(f"  CoM:  {com:.2f} Hz")
            lines.append(f"  peak: {peak:.2f} Hz")
            lines.append(f"  area: {area:.3f}")
        else:
            lines.append("  CoM:  NA")
            lines.append("  peak: NA")
            lines.append("  area: 0.000")
        lines.append("")
    ax.text(
        0.02,
        0.98,
        "\n".join(lines).rstrip(),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#B0BEC5"),
    )


def plot_subject_pac_phase_qc(
    subject: str,
    subject_index: int,
    spectra_by_block: dict[int, object],
    metrics_df: pd.DataFrame,
    bands: dict[str, tuple[float, float]],
    output_dir: Path,
    plot_cfg: dict,
) -> Path:
    """Save a 2x2 PAC phase QC dashboard for one participant."""
    output_dir.mkdir(parents=True, exist_ok=True)
    blocks = sorted(spectra_by_block)
    if not blocks:
        raise ValueError(f"No spectra available for {subject}.")

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(18, 9),
        sharex="col",
        gridspec_kw={"width_ratios": [1.05, 1.05, 0.62]},
    )
    xlim = tuple(float(v) for v in plot_cfg.get("xlim", [1.0, 30.0]))

    for row_idx, block in enumerate(blocks[:2]):
        spectrum = spectra_by_block[block]
        rows = metrics_df[(metrics_df["subject"] == subject) & (metrics_df["block"] == block)]
        color = COLORS["block1"] if block == min(blocks) else COLORS["block5"]

        ax_psd = axes[row_idx, 0]
        _shade_bands(ax_psd, bands)
        ax_psd.plot(spectrum.freqs, spectrum.log_psd, color=color, lw=1.8, label="PSD")
        ax_psd.plot(spectrum.freqs, spectrum.aperiodic_log, color=COLORS["fit"], lw=1.6, ls="--", label="Aperiodic fit")
        ax_psd.set_ylabel("log10 power")
        ax_psd.set_title(f"Block {block}: frontal PSD + aperiodic fit")
        ax_psd.set_xlim(*xlim)
        ax_psd.legend(loc="best", fontsize=8)
        _style(ax_psd)

        ax_res = axes[row_idx, 1]
        _shade_bands(ax_res, bands)
        ax_res.axhline(0, color="#777777", lw=0.8)
        ax_res.plot(spectrum.freqs, spectrum.residual_log, color=color, lw=1.8)
        for _, row in rows.iterrows():
            band = row["band"]
            band_color = COLORS.get(band, "#607D8B")
            com = row["centre_of_mass_hz"]
            peak = row["peak_frequency_hz"]
            if np.isfinite(com):
                ax_res.axvline(com, color=band_color, lw=1.5, label=f"{band} CoM")
            if np.isfinite(peak):
                ax_res.axvline(peak, color=band_color, lw=1.1, ls=":")
        ax_res.set_ylabel("residual log10 power")
        ax_res.set_title(f"Block {block}: residual spectrum")
        ax_res.set_xlim(*xlim)
        _style(ax_res)

        _metrics_panel(axes[row_idx, 2], block, spectrum, rows)

    for ax in axes[-1, :2]:
        ax.set_xlabel("Frequency (Hz)")

    fig.suptitle(
        f"{subject}: PAC phase spectral QC (C_broad_F, PAC window samples)",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = output_dir / f"{subject_index:02d}_{subject}_pac_phase_qc.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out
