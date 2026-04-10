"""
Publication-quality P3b ERP visualization (Nature Human Behaviour style).

Single-panel ERP waveform: Block 1 vs Block 5 with P3b window and SEM.

Usage:
    python vis_p3b.py                    # All subjects
    python vis_p3b.py --subject sub-p006 # Single subject
    python vis_p3b.py --subject sub-p006 sub-p003  # Multiple
    python vis_p3b.py --group            # Group-level summary

Reads clean P3b epochs from:
    outputs/derivatives/epochs_clean/  (per-subject layout)

Output figures saved to:
    outputs/publication_figures/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# --- Path setup ---
PIPELINE_DIR = Path(__file__).resolve().parents[2]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from src.utils_features import load_block_epochs
from src.utils_io import load_config, discover_subjects

EPOCHS_DIR = PIPELINE_DIR / "outputs" / "derivatives" / "epochs_clean"
FIG_DIR = PIPELINE_DIR / "outputs" / "publication_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
FEATURES_DIR = PIPELINE_DIR / "outputs" / "features"

# --- Design constants (Nature Human Behaviour palette) ---
COLOR_B1 = "#2166AC"        # Refined blue
COLOR_B5 = "#B2182B"        # Refined red
COLOR_B1_FILL = "#D1E5F0"   # Light blue ribbon
COLOR_B5_FILL = "#FDDBC7"   # Light red ribbon
COLOR_P3B_WIN = "#C8E6C9"   # Soft green P3b highlight
COLOR_GRID = "#E0E0E0"
COLOR_TEXT = "#212121"
COLOR_SUBTEXT = "#616161"

P3B_WINDOW = (0.300, 0.500)  # seconds
CROP_TMIN = -0.100
CROP_TMAX = 0.700
LP_FREQ = 12.0
P3B_CLUSTER = ["Pz", "P1", "P2", "POz"]  # centroparietal cluster from parameters.json


def _setup_style():
    """Set Nature Human Behaviour-quality matplotlib defaults."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.transparent": False,
        "lines.linewidth": 1.5,
    })


def _load_erp(subj: str, block: int, ch_picks: list[str] | None = None):
    """Load clean P3b epochs, compute ERP and SEM averaged across cluster."""
    if ch_picks is None:
        ch_picks = P3B_CLUSTER

    ep = load_block_epochs(subj, block, "p3b_erp", EPOCHS_DIR)
    if ep is None:
        return None

    # Use available channels from the cluster — no fallback
    avail = [ch for ch in ch_picks if ch in ep.ch_names]
    if not avail:
        raise ValueError(
            f"{subj} block {block}: none of {ch_picks} found in {ep.ch_names}"
        )

    # Mild low-pass for readability
    try:
        ep = ep.copy().filter(
            l_freq=None, h_freq=LP_FREQ, method="iir",
            iir_params=dict(order=4, ftype="butter"),
            n_jobs=1, verbose=False,
        )
    except Exception:
        pass

    ep = ep.copy().crop(tmin=CROP_TMIN, tmax=CROP_TMAX)
    data = ep.copy().pick(avail).get_data()  # (n_epochs, n_chs, n_times)
    data_uv = data * 1e6
    # Average across channels first, then compute ERP
    data_uv_avg = data_uv.mean(axis=1)  # (n_epochs, n_times)

    erp = data_uv_avg.mean(axis=0)
    sem = data_uv_avg.std(axis=0, ddof=1) / np.sqrt(data_uv_avg.shape[0])
    times_ms = ep.times * 1000.0
    n_trials = data_uv_avg.shape[0]

    # P3b metrics
    p3b_mask = (ep.times >= P3B_WINDOW[0]) & (ep.times <= P3B_WINDOW[1])
    p3b_mean = float(erp[p3b_mask].mean())
    # Fractional area latency (50%, Luck 2014)
    win_erp = erp[p3b_mask]
    win_times = times_ms[p3b_mask]
    pos_erp = np.clip(win_erp, 0, None)
    total = pos_erp.sum()
    if total > 0:
        cum = np.cumsum(pos_erp)
        fal_idx = min(int(np.searchsorted(cum, 0.5 * total)), len(win_times) - 1)
    else:
        fal_idx = len(win_times) // 2
    p3b_fal_lat = float(win_times[fal_idx])
    p3b_fal_amp = float(win_erp[fal_idx])

    cluster_label = ", ".join(avail)

    return {
        "erp": erp, "sem": sem, "times_ms": times_ms,
        "n_trials": n_trials, "cluster_label": cluster_label,
        "n_channels": len(avail),
        "p3b_mean": p3b_mean,
        "p3b_fal_lat": p3b_fal_lat, "p3b_fal_amp": p3b_fal_amp,
    }



def plot_subject(subj: str) -> Path | None:
    """Create single-panel P3b ERP figure for one subject."""
    _setup_style()

    d1 = _load_erp(subj, 1)
    d5 = _load_erp(subj, 5)
    if d1 is None or d5 is None:
        print(f"  {subj}: missing P3b epochs for one or both blocks, skipping.")
        return None

    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor="white")

    # P3b window shading
    ax.axvspan(
        P3B_WINDOW[0] * 1000, P3B_WINDOW[1] * 1000,
        color=COLOR_P3B_WIN, alpha=0.45, zorder=0,
    )
    # Subtle P3b label inside the window
    y_range = max(d1["erp"].max(), d5["erp"].max()) - min(d1["erp"].min(), d5["erp"].min())
    y_bottom = min(d1["erp"].min(), d5["erp"].min()) - y_range * 0.08
    ax.text(
        400, y_bottom, "P3b window",
        ha="center", va="bottom", fontsize=7, color="#2E7D32",
        fontstyle="italic", alpha=0.7,
    )

    # Block 1: SEM ribbon + line
    ax.fill_between(
        d1["times_ms"], d1["erp"] - d1["sem"], d1["erp"] + d1["sem"],
        color=COLOR_B1_FILL, alpha=0.5, zorder=1, linewidth=0,
    )
    ax.plot(
        d1["times_ms"], d1["erp"], color=COLOR_B1, lw=2.0, zorder=3,
        label=f'Block 1  (n = {d1["n_trials"]})',
    )

    # Block 5: SEM ribbon + line
    ax.fill_between(
        d5["times_ms"], d5["erp"] - d5["sem"], d5["erp"] + d5["sem"],
        color=COLOR_B5_FILL, alpha=0.5, zorder=1, linewidth=0,
    )
    ax.plot(
        d5["times_ms"], d5["erp"], color=COLOR_B5, lw=2.0, zorder=3,
        label=f'Block 5  (n = {d5["n_trials"]})',
    )

    # Reference lines
    ax.axhline(0, color=COLOR_GRID, lw=0.5, zorder=2)
    ax.axvline(0, color=COLOR_GRID, lw=0.5, zorder=2)

    # Fractional area latency markers
    for d, c in [(d1, COLOR_B1), (d5, COLOR_B5)]:
        ax.plot(
            d["p3b_fal_lat"], d["p3b_fal_amp"], "v",
            color=c, ms=7, zorder=5,
            markeredgecolor="white", markeredgewidth=0.9,
        )

    # Inline delta annotation
    delta_mean = d5["p3b_mean"] - d1["p3b_mean"]
    sign = "+" if delta_mean >= 0 else ""
    ax.annotate(
        f"Δmean = {sign}{delta_mean:.2f} µV",
        xy=(P3B_WINDOW[1] * 1000 + 15,
            (d1["p3b_fal_amp"] + d5["p3b_fal_amp"]) / 2),
        fontsize=8, color=COLOR_SUBTEXT, fontstyle="italic",
        va="center",
    )

    ax.set_xlabel("Time (ms)", color=COLOR_TEXT)
    ax.set_ylabel("Amplitude (µV)", color=COLOR_TEXT)
    ax.set_xlim(CROP_TMIN * 1000, CROP_TMAX * 1000)
    ax.legend(loc="upper left", fontsize=8.5, handlelength=1.8)
    ax.tick_params(colors=COLOR_SUBTEXT)
    for spine in ax.spines.values():
        spine.set_color(COLOR_SUBTEXT)

    # Title — left-aligned, NHB style
    ax.set_title(
        f"P3b Event-Related Potential — {subj}",
        loc="left", fontsize=11, fontweight="bold", color=COLOR_TEXT, pad=10,
    )

    out = FIG_DIR / f"p3b_{subj}.png"
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


def plot_group(exclude=None):
    """Create group-level P3b summary from features CSV."""
    _setup_style()

    csv_path = FEATURES_DIR / "p3b_features.csv"
    if not csv_path.exists():
        print(f"No P3b features CSV at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if exclude:
        df = df[~df["subject"].isin(exclude)]
        if df.empty:
            print("No subjects remaining after exclusions.")
            return
    pivot = df.pivot(index="subject", columns="block", values="p3b_amp_uV")
    pivot["delta"] = pivot[5] - pivot[1]
    pivot = pivot.sort_values("delta")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), facecolor="white")

    # --- Panel a: Individual deltas ---
    ax = axes[0]
    subjects = pivot.index.tolist()
    deltas = pivot["delta"].values
    n = len(subjects)
    colors = [COLOR_B5 if d < 0 else COLOR_B1 for d in deltas]

    ax.barh(
        range(n), deltas, color=colors,
        edgecolor="white", linewidth=0.8, alpha=0.88, height=0.55,
    )
    ax.set_yticks(range(n))
    ax.set_yticklabels(
        [s.replace("sub-", "") for s in subjects],
        fontsize=8.5, fontweight="medium",
    )
    ax.axvline(0, color=COLOR_SUBTEXT, lw=0.6)
    ax.set_xlabel("ΔP3b amplitude (µV)", fontsize=9)
    ax.set_title("a   Individual fatigue effect (B5 – B1)",
                 loc="left", fontsize=10.5, pad=10)

    mean_d = deltas.mean()
    ax.axvline(mean_d, color="#FF6F00", lw=1.8, ls="--", zorder=5,
               label=f"Group mean Δ = {mean_d:+.2f} µV")
    ax.legend(fontsize=7.5, loc="lower left")
    ax.tick_params(colors=COLOR_SUBTEXT)
    for spine in ax.spines.values():
        spine.set_color(COLOR_SUBTEXT)

    # --- Panel b: Paired spaghetti ---
    ax = axes[1]
    for subj in subjects:
        b1 = pivot.loc[subj, 1]
        b5 = pivot.loc[subj, 5]
        c = COLOR_B5 if b5 < b1 else COLOR_B1
        ax.plot([0, 1], [b1, b5], color=c, alpha=0.55, lw=1.2, zorder=2)
        ax.scatter([0, 1], [b1, b5], color=c, s=25, zorder=3,
                   edgecolors="white", linewidths=0.6)

    m1, m5 = pivot[1].mean(), pivot[5].mean()
    ax.plot([0, 1], [m1, m5], color=COLOR_TEXT, lw=2.5, zorder=4)
    ax.scatter([0, 1], [m1, m5], color=COLOR_TEXT, s=55, zorder=5,
              edgecolors="white", linewidths=1.2)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Block 1\n(pre-fatigue)", "Block 5\n(post-fatigue)"],
                       fontsize=9)
    ax.set_ylabel("P3b amplitude (µV)", fontsize=9)
    ax.set_title("b   Paired comparison", loc="left", fontsize=10.5, pad=10)
    ax.tick_params(colors=COLOR_SUBTEXT)
    for spine in ax.spines.values():
        spine.set_color(COLOR_SUBTEXT)

    if exclude:
        excl_str = ", ".join(s.replace("sub-", "") for s in exclude)
        fig.text(0.99, 0.01, f"Excluded: {excl_str}", fontsize=6.5,
                 ha="right", color=COLOR_SUBTEXT, fontstyle="italic")

    fig.suptitle(
        f"Group P3b Summary (N = {n})",
        fontsize=12, fontweight="bold", color=COLOR_TEXT,
        x=0.06, ha="left", y=0.99,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    out = FIG_DIR / "p3b_group_summary.png"
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out}")


def main():
    parser = argparse.ArgumentParser(
        description="Publication P3b ERP visualization (NHB style).",
    )
    parser.add_argument("--subject", type=str, nargs="+", default=None,
                        help="One or more subjects (e.g. sub-p006 sub-p003)")
    parser.add_argument("--group", action="store_true",
                        help="Generate group-level summary only")
    parser.add_argument("--exclude", nargs="*", default=["sub-p005"],
                        help="Exclude from group plot (default: sub-p005)")
    args = parser.parse_args()

    if args.group:
        plot_group(exclude=args.exclude)
        return

    cfg = load_config()
    blocks = cfg.get("blocks", [1, 5])

    if args.subject:
        for subj in args.subject:
            plot_subject(subj)
    else:
        subjects = discover_subjects(
            epochs_dir=EPOCHS_DIR, blocks=blocks,
            epoch_type="p3b", require_all_blocks=False,
        )
        for subj in subjects:
            plot_subject(subj)

    # Always generate group if features CSV exists
    if (FEATURES_DIR / "p3b_features.csv").exists():
        plot_group(exclude=args.exclude)


if __name__ == "__main__":
    main()
