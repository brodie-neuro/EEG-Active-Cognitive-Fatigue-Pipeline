"""Generate group-average P3b ERP waveform figure.

Uses 0.1 Hz HP epochs (as stored), 12 Hz LP display filter.
Fractional area latency (50%) for peak markers (Luck, 2014).
Reports mean amplitude and 50% area latency only.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PIPELINE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPELINE_DIR))

from src.utils_features import load_block_epochs
from src.utils_io import load_config, discover_subjects

EPOCHS_DIR = PIPELINE_DIR / "outputs" / "derivatives" / "epochs_clean"
FEATURES_DIR = PIPELINE_DIR / "outputs" / "features"
FIG_DIR = PIPELINE_DIR / "outputs" / "publication_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CHANNELS = ["Pz", "P1", "P2", "POz"]
P3B_WINDOW = (0.300, 0.500)
CROP = (-0.100, 0.650)
LP_FREQ = 12.0
EXCLUDE = ["sub-p005"]

COLOR_B1 = "#2166AC"
COLOR_B5 = "#B2182B"
COLOR_B1_FILL = "#D1E5F0"
COLOR_B5_FILL = "#FDDBC7"
COLOR_P3B_WIN = "#C8E6C9"
COLOR_TEXT = "#212121"
COLOR_SUBTEXT = "#616161"


def _frac_area_latency(erp, times, t_mask, fraction=0.5):
    """
    Fractional area latency (Luck, 2014).

    Find the timepoint at which `fraction` (default 50%) of the total
    positive area under the ERP in the window has been reached.

    Returns (latency_ms, amplitude_at_latency).
    """
    win_erp = np.clip(erp[t_mask], 0, None)  # only positive area
    win_times = times[t_mask]

    if win_erp.sum() == 0:
        mid = len(win_times) // 2
        return win_times[mid], 0.0

    cumulative = np.cumsum(win_erp)
    total_area = cumulative[-1]
    target = fraction * total_area

    # Find first index where cumulative area >= target
    idx = np.searchsorted(cumulative, target)
    idx = min(idx, len(win_times) - 1)

    lat = win_times[idx]
    amp = erp[t_mask][idx]
    return lat, amp


def main():
    cfg = load_config()
    blocks = cfg.get("blocks", [1, 5])
    subjects = discover_subjects(
        epochs_dir=EPOCHS_DIR, blocks=blocks,
        epoch_type="p3b_erp", require_all_blocks=True,
    )
    subjects = [s for s in subjects if s not in EXCLUDE]
    print(f"Subjects (excl p005): {subjects}")

    all_erps = {1: [], 5: []}

    for subj in subjects:
        for block in [1, 5]:
            ep = load_block_epochs(subj, block, "p3b_erp", EPOCHS_DIR)
            if ep is None:
                print(f"  SKIP {subj} block {block}")
                continue
            avail = [c for c in CHANNELS if c in ep.ch_names]
            if not avail:
                continue

            # Display LP filter only
            try:
                ep = ep.copy().filter(
                    l_freq=None, h_freq=LP_FREQ, method="iir",
                    iir_params=dict(order=4, ftype="butter"),
                    n_jobs=1, verbose=False,
                )
            except Exception:
                pass

            ep = ep.copy().crop(tmin=CROP[0], tmax=CROP[1])
            evoked = ep.pick(avail).average()
            data_uv = evoked.data.mean(axis=0) * 1e6
            all_erps[block].append(data_uv)
            print(f"  {subj} B{block}: {len(ep)} epochs, {len(avail)} channels")

    if not all_erps[1] or not all_erps[5]:
        print("Not enough data")
        return

    times = evoked.times * 1000

    min_len = min(min(len(e) for e in all_erps[1]), min(len(e) for e in all_erps[5]))
    erps_b1 = np.array([e[:min_len] for e in all_erps[1]])
    erps_b5 = np.array([e[:min_len] for e in all_erps[5]])
    times = times[:min_len]

    mean_b1 = erps_b1.mean(axis=0)
    mean_b5 = erps_b5.mean(axis=0)
    sem_b1 = erps_b1.std(axis=0, ddof=1) / np.sqrt(len(erps_b1))
    sem_b5 = erps_b5.std(axis=0, ddof=1) / np.sqrt(len(erps_b5))
    n = len(subjects)

    t_mask = (times >= P3B_WINDOW[0]*1000) & (times <= P3B_WINDOW[1]*1000)
    delta_mean = mean_b5[t_mask].mean() - mean_b1[t_mask].mean()

    # Fractional area latency (50%)
    fal_b1_t, fal_b1_amp = _frac_area_latency(mean_b1, times, t_mask)
    fal_b5_t, fal_b5_amp = _frac_area_latency(mean_b5, times, t_mask)
    print(f"  B1 FAL(50%): {fal_b1_t:.0f} ms, {fal_b1_amp:.2f} uV")
    print(f"  B5 FAL(50%): {fal_b5_t:.0f} ms, {fal_b5_amp:.2f} uV")
    print(f"  Delta mean amp: {delta_mean:+.2f} uV")

    # --- Plot ---
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6,
    })

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="white")

    ax.axvspan(P3B_WINDOW[0]*1000, P3B_WINDOW[1]*1000,
               color=COLOR_P3B_WIN, alpha=0.4, zorder=0)
    ax.axvline(0, color="#9E9E9E", lw=0.8, ls="--", zorder=1)

    ax.fill_between(times, mean_b1 - sem_b1, mean_b1 + sem_b1,
                    color=COLOR_B1_FILL, alpha=0.5, zorder=2)
    ax.fill_between(times, mean_b5 - sem_b5, mean_b5 + sem_b5,
                    color=COLOR_B5_FILL, alpha=0.5, zorder=2)

    ax.plot(times, mean_b1, color=COLOR_B1, lw=2.2, zorder=3,
            label=f"Block 1  (N = {n})")
    ax.plot(times, mean_b5, color=COLOR_B5, lw=2.2, zorder=3,
            label=f"Block 5  (N = {n})")

    # Fractional area latency markers
    ax.plot(fal_b1_t, fal_b1_amp, "v", color=COLOR_B1, ms=8, zorder=4,
            markeredgecolor="white", mew=0.8)
    ax.plot(fal_b5_t, fal_b5_amp, "v", color=COLOR_B5, ms=8, zorder=4,
            markeredgecolor="white", mew=0.8)

    ax.text(400, -1.8, "P3b window", fontsize=7.5, color="#388E3C",
            ha="center", va="top", style="italic")

    ax.text(0.78, 0.88, f"$\\Delta$mean = {delta_mean:+.2f} $\\mu$V",
            transform=ax.transAxes, fontsize=10, color=COLOR_SUBTEXT,
            fontstyle="italic")

    ax.set_xlabel("Time (ms)", fontsize=10)
    ax.set_ylabel("Amplitude ($\\mu$V)", fontsize=10)
    ax.set_title(f"P3b Event-Related Potential — Group Mean (N = {n})",
                 fontsize=13, fontweight="bold", color=COLOR_TEXT, pad=10)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)

    ax.set_ylim(-2, None)
    ax.set_xlim(CROP[0]*1000, CROP[1]*1000)

    if EXCLUDE:
        excl_str = ", ".join(s.replace("sub-", "") for s in EXCLUDE)
        fig.text(0.99, 0.01, f"Excluded: {excl_str}", fontsize=6.5,
                 ha="right", color=COLOR_SUBTEXT, fontstyle="italic")

    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(0.8)

    fig.tight_layout()
    out = FIG_DIR / "p3b_group_erp.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
