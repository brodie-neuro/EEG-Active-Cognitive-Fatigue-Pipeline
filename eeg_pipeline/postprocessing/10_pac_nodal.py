# eeg_pipeline/analysis/13_pac_nodal.py
"""
Step 4 -- Phase-Amplitude Coupling Analysis

Between-region PAC — direct frontal→parietal coupling
    Primary (locked): C_broad_F → C_broad_P
    Descriptive 9-node delta-PAC heatmap data (all-pairs)

Modulation Index (Tort et al., 2010) with surrogate z-scoring.
Uses FIR filtering, trial concatenation, and 500 surrogates.
Analysis window: 0-0.6 s stimulus-locked.
Outputs long format: one row per subject x block.

Reference: post_processing_EEG_plan_v2.docx, Step 4
"""
import sys
import os
import copy
import argparse
from pathlib import Path
# Deterministic BLAS/LAPACK thread limits: set before scientific imports.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MNE_DONTWRITE_HOME"] = "true"
os.environ.setdefault("_MNE_FAKE_HOME_DIR", os.path.dirname(os.path.dirname(__file__)))
import mne
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

pipeline_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pipeline_dir))

from src.utils_io import load_config, discover_subjects
from src.utils_determinism import file_sha256, save_step_qc
from src.utils_features import (
    load_block_epochs, get_subjects_with_blocks,
    available_channels, get_node_channels,
    filter_excluded_channels,
)

# Tensorpac REMOVED — trial-concatenation is the sole PAC method.
# This ensures fully deterministic Z-scores with RandomState(42).

OUTPUT_DIR = pipeline_dir / "outputs" / "features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = pipeline_dir / "outputs" / "figures" / "pac"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_TAG = os.environ.get("EEG_OUTPUT_TAG", "").strip()

# -- Plotting constants --
_CLR_B1 = '#1E88E5'
_CLR_B5 = '#E53935'
_STAT_BOX = dict(boxstyle='round,pad=0.4', facecolor='#ECEFF1', edgecolor='#B0BEC5')


def _style_ax(ax):
    """Apply publication style: remove top/right spines, light grid."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)


def _draw_head(ax, lw=1.5):
    """Draw a standard head outline with nose and ears."""
    theta = np.linspace(0, 2 * np.pi, 100)
    r = 0.5
    ax.plot(r * np.cos(theta), r * np.sin(theta), 'k-', lw=lw)
    ax.plot([-.08, 0, .08], [r, r + 0.08, r], 'k-', lw=lw)  # nose
    for sign in [-1, 1]:
        ear_x = sign * np.array([r, r + .03, r + .04, r + .03, r])
        ear_y = np.array([.08, .06, 0, -.06, -.08])
        ax.plot(ear_x, ear_y, 'k-', lw=lw)
    ax.set_xlim(-0.65, 0.65)
    ax.set_ylim(-0.60, 0.65)
    ax.set_aspect('equal')
    ax.axis('off')


# Approximate 2D positions for frontal and parietal clusters
_NODE_XY = {
    'C_broad_F': (0.0, 0.30),   # frontal midline cluster
    'C_broad_P': (0.0, -0.30),  # parietal midline cluster
}


def _draw_pac_arrow(ax, z_score, src_xy, dst_xy, block_clr, label):
    """Draw a directed arrow colored/sized by PAC Z-score."""
    from matplotlib.patches import FancyArrowPatch

    width = min(6.0, max(1.0, abs(z_score) * 40))
    alpha = min(0.9, max(0.25, abs(z_score) * 0.3))
    arrow = FancyArrowPatch(
        src_xy, dst_xy,
        arrowstyle='Simple,tail_width=2,head_width=10,head_length=10',
        color=block_clr, linewidth=width, alpha=alpha, zorder=8,
        connectionstyle='arc3,rad=0.0',
    )
    ax.add_patch(arrow)
    mid_x = (src_xy[0] + dst_xy[0]) / 2 + 0.15
    mid_y = (src_xy[1] + dst_xy[1]) / 2
    ax.text(mid_x, mid_y, f'Z={z_score:+.2f}', fontsize=10, fontweight='bold',
            color=block_clr, ha='left', va='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))


def _get_montage_positions():
    import mne
    montage = mne.channels.make_standard_montage('easycap-M1')
    pos3d = montage.get_positions()['ch_pos']
    pos2d = {ch: (coords[0], coords[1]) for ch, coords in pos3d.items()}
    return pos2d

def _draw_convex_hull(ax, points, color, alpha=0.3, label=None):
    from scipy.spatial import ConvexHull
    from matplotlib.patches import Polygon
    if len(points) > 2:
        hull = ConvexHull(points)
        poly = Polygon(points[hull.vertices], closed=True,
                       facecolor=color, edgecolor=color, alpha=alpha, linewidth=2, zorder=2)
        ax.add_patch(poly)
        cx = np.mean([p[0] for p in points])
        cy = np.mean([p[1] for p in points])
        if label:
            y_offset = 0.02 if cy > 0 else -0.02
            ax.text(cx, cy + y_offset, label, ha='center', va='center', fontsize=8, fontweight='bold', color='#37474F', zorder=10)
        return (cx, cy)
    return (0, 0)

def _plot_pac_layout_A(subj, block_details):
    """Layout A: Publication-quality Scalp Montage — B1 | B5 | Delta."""
    sorted_blocks = sorted(block_details.keys())
    if len(sorted_blocks) < 2:
        return

    b1, b5 = sorted_blocks[0], sorted_blocks[-1]
    z_b1, z_b5 = block_details[b1]['z'], block_details[b5]['z']
    z_delta = z_b5 - z_b1

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    titles = [f'Block {b1} (Baseline)', f'Block {b5} (Fatigued)', f'Delta (B{b5} - B{b1})']
    z_vals = [z_b1, z_b5, z_delta]
    clrs = ['#1E88E5', '#E53935', '#7B1FA2']
    
    pos2d = _get_montage_positions()
    C_BROAD_F = ['Fz', 'FCz', 'F3', 'F4', 'FC1', 'FC2', 'FC3', 'FC4']
    C_BROAD_P = ['Pz', 'POz', 'P3', 'P4', 'P1', 'P2', 'PO3', 'PO4']
    
    # Set up colormap for arrows
    max_z = max(abs(v) for v in z_vals)
    import matplotlib.colors as mcolors
    norm = mcolors.TwoSlopeNorm(vmin=-max_z, vcenter=0.0, vmax=max_z)
    cmap = plt.cm.RdBu_r

    from matplotlib.patches import FancyArrowPatch

    for ax, title, z_val, clr in zip(axes, titles, z_vals, clrs):
        _style_ax(ax)
        ax.set_xticks([])
        ax.set_yticks([])
        
        all_x = [p[0] for p in pos2d.values()]
        all_y = [p[1] for p in pos2d.values()]
        ax.scatter(all_x, all_y, c='lightgray', s=15, zorder=1)
        
        theta = np.linspace(0, 2*np.pi, 100)
        r = 0.1  
        ax.plot(r*np.cos(theta), r*np.sin(theta), 'k-', lw=1.5, zorder=0)
        ax.plot([-0.015, 0, 0.015], [r, r + 0.015, r], 'k-', lw=1.5, zorder=0)
        for sign in [-1, 1]:
            ax.plot([sign*r, sign*(r+0.005), sign*(r+0.005), sign*r], [0.015, 0.01, -0.01, -0.015], 'k-', lw=1.5, zorder=0)
        
        f_pts = np.array([pos2d[ch] for ch in C_BROAD_F if ch in pos2d])
        p_pts = np.array([pos2d[ch] for ch in C_BROAD_P if ch in pos2d])
        
        f_center = _draw_convex_hull(ax, f_pts, color='#1E88E5', alpha=0.3, label='Frontal\nCluster')
        p_center = _draw_convex_hull(ax, p_pts, color='#E53935', alpha=0.3, label='Parietal\nCluster')
        
        width = min(8.0, max(1.5, abs(z_val) * 2.5))
        arr_clr = cmap(norm(z_val))
        
        arrow = FancyArrowPatch(
            f_center, p_center,
            arrowstyle='Simple,tail_width=2.5,head_width=12,head_length=12',
            color=arr_clr, linewidth=width, alpha=0.85, zorder=8,
            connectionstyle='arc3,rad=-0.1',
        )
        ax.add_patch(arrow)
        
        stats_txt = f"Z = {z_val:+.2f}"
        bbox_props = dict(boxstyle='round,pad=0.4', facecolor='#ECEFF1', edgecolor='#B0BEC5', alpha=0.9)
        ax.text(0.95, 0.95, stats_txt, transform=ax.transAxes,
                fontsize=11, fontweight='bold', color=clr, ha='right', va='top', bbox=bbox_props, zorder=10)
        
        ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
        ax.set_xlim(-0.12, 0.12)
        ax.set_ylim(-0.12, 0.12)
        ax.set_aspect('equal')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(sm, cax=cbar_ax, label='PAC Z-score')

    fig.suptitle(f'PAC Scalp Montage — {subj}\nFrontal theta phase $\\rightarrow$ Parietal gamma amplitude',
                 fontsize=14, fontweight='bold', y=1.05)
    
    out = FIG_DIR / f"pac_layoutA_montage_{subj}.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved Layout A (montage): {out}")


def _plot_pac_layout_B(subj, block_details):
    """Layout B: Bar/paired comparison with surrogate null band."""
    sorted_blocks = sorted(block_details.keys())
    if len(sorted_blocks) < 2:
        return

    b1, b5 = sorted_blocks[0], sorted_blocks[-1]
    info_b1, info_b5 = block_details[b1], block_details[b5]

    fig, (ax_z, ax_mi) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left: Z-score comparison ---
    bars = ax_z.bar([f'Block {b1}\n(Baseline)', f'Block {b5}\n(Fatigued)'],
                    [info_b1['z'], info_b5['z']],
                    color=[_CLR_B1, _CLR_B5], width=0.5, edgecolor='black', lw=1.2)
    # Significance thresholds
    ax_z.axhline(1.96, color='gray', ls='--', lw=1, alpha=0.6, label='Z=1.96 (p<.05)')
    ax_z.axhline(-1.96, color='gray', ls='--', lw=1, alpha=0.6)
    ax_z.axhline(2.58, color='gray', ls=':', lw=1, alpha=0.4, label='Z=2.58 (p<.01)')
    ax_z.axhline(-2.58, color='gray', ls=':', lw=1, alpha=0.4)
    ax_z.axhline(0, color='black', lw=0.5)

    # Delta annotation
    delta_z = info_b5['z'] - info_b1['z']
    max_z = max(abs(info_b1['z']), abs(info_b5['z']))
    ax_z.annotate(f'Delta Z = {delta_z:+.2f}',
                  xy=(0.5, max(info_b1['z'], info_b5['z'])),
                  xytext=(0.5, max_z + 0.5),
                  fontsize=11, fontweight='bold', ha='center',
                  arrowprops=dict(arrowstyle='-', color='gray', lw=0.8))

    # Stats box
    stats = (f"B{b1}: Z={info_b1['z']:.2f}, MI={info_b1['mi_real']:.4f}\n"
             f"B{b5}: Z={info_b5['z']:.2f}, MI={info_b5['mi_real']:.4f}\n"
             f"Delta Z: {delta_z:+.2f}")
    ax_z.text(0.02, 0.98, stats, transform=ax_z.transAxes,
              fontsize=9, va='top', bbox=_STAT_BOX)
    ax_z.set_ylabel('PAC Z-score', fontsize=12)
    ax_z.set_title('Frontal->Parietal PAC Strength', fontsize=13, fontweight='bold')
    ax_z.legend(fontsize=8, loc='lower right')
    _style_ax(ax_z)

    # --- Right: MI with surrogate null bands ---
    # Plot surrogate distributions as shaded ranges
    for i, (block, info, clr, xpos) in enumerate([
        (b1, info_b1, _CLR_B1, 0), (b5, info_b5, _CLR_B5, 1)
    ]):
        surr = info['surr_mis']
        mean_s, std_s = np.mean(surr), np.std(surr)
        # Null band (±2 SD)
        ax_mi.fill_between([xpos - 0.2, xpos + 0.2],
                           mean_s - 2 * std_s, mean_s + 2 * std_s,
                           color=clr, alpha=0.15, label=f'B{block} null (+-2 SD)' if i == 0 else None)
        # Null mean line
        ax_mi.hlines(mean_s, xpos - 0.2, xpos + 0.2, color=clr, ls='--', lw=1, alpha=0.5)
        # Real MI as large diamond
        ax_mi.plot(xpos, info['mi_real'], 'D', color=clr, ms=14, zorder=5,
                   markeredgecolor='black', markeredgewidth=1.5)
        ax_mi.text(xpos + 0.25, info['mi_real'], f"MI={info['mi_real']:.4f}",
                   fontsize=9, va='center', color=clr, fontweight='bold')

    # Connect with line
    ax_mi.plot([0, 1], [info_b1['mi_real'], info_b5['mi_real']],
               'k--', lw=1.5, alpha=0.5)

    ax_mi.set_xticks([0, 1])
    ax_mi.set_xticklabels([f'Block {b1}\n(Baseline)', f'Block {b5}\n(Fatigued)'], fontsize=11)
    ax_mi.set_ylabel('Modulation Index', fontsize=12)
    ax_mi.set_title('Raw MI vs Surrogate Null', fontsize=13, fontweight='bold')
    _style_ax(ax_mi)

    fig.suptitle(f'PAC Comparison — {subj} (C_broad_F -> C_broad_P)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    out = FIG_DIR / f"pac_layoutB_comparison_{subj}.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved Layout B (comparison): {out}")


def _plot_pac_layout_C(subj, block_details):
    """Layout C: Hybrid — scalp arrows (top) + surrogate distributions (bottom)."""
    sorted_blocks = sorted(block_details.keys())
    if len(sorted_blocks) < 2:
        return

    b1, b5 = sorted_blocks[0], sorted_blocks[-1]
    info_b1, info_b5 = block_details[b1], block_details[b5]
    z_delta = info_b5['z'] - info_b1['z']

    fig = plt.figure(figsize=(18, 12))
    # Top row: 3 scalp montages
    ax_h1 = fig.add_subplot(2, 3, 1)
    ax_h2 = fig.add_subplot(2, 3, 2)
    ax_h3 = fig.add_subplot(2, 3, 3)
    # Bottom row: 2 surrogate histograms + 1 summary
    ax_s1 = fig.add_subplot(2, 3, 4)
    ax_s2 = fig.add_subplot(2, 3, 5)
    ax_sum = fig.add_subplot(2, 3, 6)

    # --- Top row: scalp montages ---
    for ax, title, z_val, clr in [
        (ax_h1, f'Block {b1} (Baseline)', info_b1['z'], _CLR_B1),
        (ax_h2, f'Block {b5} (Fatigued)', info_b5['z'], _CLR_B5),
        (ax_h3, f'Delta (B{b5} - B{b1})', z_delta, '#7B1FA2'),
    ]:
        _draw_head(ax)
        for node, (x, y) in _NODE_XY.items():
            label = 'Frontal' if 'F' in node else 'Parietal'
            ax.scatter(x, y, s=500, c='#E8EAF6', edgecolors='#37474F',
                       linewidth=2, zorder=10)
            ax.text(x, y, label, ha='center', va='center', fontsize=7,
                    fontweight='bold', zorder=11)
        _draw_pac_arrow(ax, z_val, _NODE_XY['C_broad_F'],
                        _NODE_XY['C_broad_P'], clr, title)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=12)

    # --- Bottom left: B1 surrogate distribution ---
    for ax, block, info, clr in [
        (ax_s1, b1, info_b1, _CLR_B1),
        (ax_s2, b5, info_b5, _CLR_B5),
    ]:
        surr = info['surr_mis']
        mean_s, std_s = np.mean(surr), np.std(surr)
        ax.hist(surr, bins=40, color='lightgray', edgecolor='gray', lw=0.5)
        ax.axvline(info['mi_real'], color='red', lw=2.5, label=f'Real MI')
        ax.axvspan(mean_s - 2 * std_s, mean_s + 2 * std_s,
                   alpha=0.1, color='blue', label='+-2 SD')
        stats_txt = f"Z = {info['z']:.2f}\nMI = {info['mi_real']:.4f}"
        ax.text(0.97, 0.97, stats_txt, transform=ax.transAxes,
                fontsize=9, va='top', ha='right', bbox=_STAT_BOX)
        ax.set_xlabel('Modulation Index', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(f'Block {block} Surrogates', fontsize=11, fontweight='bold')
        ax.legend(fontsize=7, loc='upper left')
        _style_ax(ax)

    # --- Bottom right: Summary panel ---
    ax_sum.axis('off')
    summary = (
        f"Frontal theta phase -> Parietal gamma amplitude\n"
        f"{'='*45}\n\n"
        f"Block {b1} (Baseline):\n"
        f"  Z-score:  {info_b1['z']:+.2f}\n"
        f"  MI:       {info_b1['mi_real']:.4f}\n\n"
        f"Block {b5} (Fatigued):\n"
        f"  Z-score:  {info_b5['z']:+.2f}\n"
        f"  MI:       {info_b5['mi_real']:.4f}\n\n"
        f"Delta (B{b5} - B{b1}):\n"
        f"  Delta Z:  {z_delta:+.2f}\n"
        f"  Delta MI: {info_b5['mi_real'] - info_b1['mi_real']:+.4f}\n\n"
        f"Surrogates: 500 circular-shift\n"
        f"Seed: 42 (deterministic)"
    )
    ax_sum.text(0.1, 0.95, summary, transform=ax_sum.transAxes,
                fontsize=11, va='top', family='monospace',
                bbox=dict(boxstyle='round,pad=0.6', facecolor='#ECEFF1',
                          edgecolor='#78909C', lw=1.5))

    fig.suptitle(f'PAC Hybrid Diagnostic — {subj}',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    out = FIG_DIR / f"pac_layoutC_hybrid_{subj}.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved Layout C (hybrid): {out}")


def _plot_pac_diagnostic(subj, block_details):
    """Generate all three PAC diagnostic layout options."""
    _plot_pac_layout_A(subj, block_details)
    _plot_pac_layout_B(subj, block_details)
    _plot_pac_layout_C(subj, block_details)

# Revised node pairs (2026-03-04): C_broad bilateral
H1_PAIRS = [("C_broad_F", "C_broad_P")]      # primary: bilateral frontal->parietal

# All pairs to compute in one pass
BETWEEN_PAC_PAIRS = H1_PAIRS
PAC_ANALYSIS_WINDOW = (0.0, 0.6)


def _tag_path(name: str, root: Path) -> Path:
    if not OUTPUT_TAG:
        return root / name
    p = Path(name)
    return root / f"{p.stem}_{OUTPUT_TAG}{p.suffix}"


def get_node_signal(epochs, node_channels):
    """Extract mean signal across node channels. Returns (n_epochs, n_times) or None."""
    # Apply pre-specified analysis exclusion
    node_channels, _ = filter_excluded_channels(node_channels)
    avail = available_channels(node_channels, epochs.ch_names)
    if avail:
        # Drop non-EEG channels (e.g., EOG repurposed as EEG sites).
        types = epochs.get_channel_types(picks=avail)
        avail = [ch for ch, ch_type in zip(avail, types) if ch_type == 'eeg']
    if not avail:
        return None
    data = epochs.copy().pick(avail).get_data()
    return data.mean(axis=1)


def _theta_phase(signal_1d, sfreq, lo, hi):
    """Extract theta phase via IIR Butterworth + Hilbert."""
    from scipy.signal import hilbert as sp_hilbert
    filt = mne.filter.filter_data(
        signal_1d, sfreq, lo, hi, verbose=False,
        method='iir', iir_params=dict(order=4, ftype='butter'))
    return np.angle(sp_hilbert(filt))


def _gamma_amp(signal_1d, sfreq, lo, hi):
    """Extract gamma amplitude via FIR (zero-padded) + Hilbert."""
    from scipy.signal import hilbert as sp_hilbert
    filt = mne.filter.filter_data(
        signal_1d, sfreq, lo, hi, verbose=False,
        method='fir')   # MNE default: auto length, zero-padded
    return np.abs(sp_hilbert(filt))


# compute_pac_tensorpac REMOVED — see _pac_from_precomputed() for sole PAC path.


def compute_pac_trial_concat(phase_signal, amp_signal, sfreq, cfg):
    """Trial-concatenated MI + surrogates PAC implementation.

    Concatenates all trials before computing MI, giving more data points
    per phase bin and a more stable MI estimate (especially for short
    0.6 s epochs with only ~2-5 theta cycles each).
    """
    from scipy.signal import hilbert as sp_hilbert

    pac_cfg = cfg.get('pac', {})
    f_pha = pac_cfg.get('phase_band', [4, 8])
    f_amp = pac_cfg.get('amp_band', [55, 85])
    n_surr = pac_cfg.get('surrogates', 500)

    # Use ALL available trials (no cap)
    n_epochs = phase_signal.shape[0]

    # --- Filter each epoch then concatenate ---
    all_phase = []
    all_amp = []

    for ep in range(n_epochs):
        phase = _theta_phase(phase_signal[ep], sfreq, f_pha[0], f_pha[1])
        amp = _gamma_amp(amp_signal[ep], sfreq, f_amp[0], f_amp[1])
        all_phase.append(phase)
        all_amp.append(amp)

    if not all_phase:
        return np.nan

    # Concatenate across trials
    concat_phase = np.concatenate(all_phase)
    concat_amp = np.concatenate(all_amp)

    # Compute MI on concatenated data
    mi_real = _modulation_index(concat_phase, concat_amp)

    # Surrogate distribution: circular-shift the concatenated amplitude
    n_samples = len(concat_phase)
    shift_range = n_samples // 4
    rng = np.random.RandomState(42)  # Fixed seed
    surr_mis = []
    for _ in range(n_surr):
        shift = rng.randint(shift_range, n_samples - shift_range)
        shifted_amp = np.roll(concat_amp, shift)
        surr_mis.append(_modulation_index(concat_phase, shifted_amp))

    mean_surr = np.mean(surr_mis)
    std_surr = np.std(surr_mis)
    z = (mi_real - mean_surr) / std_surr if std_surr > 0 else 0
    return z


def _precompute_phase_amp(node_signal, sfreq, f_pha, f_amp, times=None, crop_window=None):
    """Precompute theta phase and gamma amplitude for ALL trials.

    Theta: IIR Butterworth (standard, aperiodic handled elsewhere).
    Gamma: FIR with zero-padding (avoids phase distortion in HF).
    """
    n_epochs = node_signal.shape[0]
    if n_epochs == 0:
        return None, None

    theta_ph = np.zeros((n_epochs, node_signal.shape[1]), dtype=float)
    gamma_am = np.zeros((n_epochs, node_signal.shape[1]), dtype=float)
    for ep in range(n_epochs):
        theta_ph[ep] = _theta_phase(node_signal[ep], sfreq, f_pha[0], f_pha[1])
        gamma_am[ep] = _gamma_amp(node_signal[ep], sfreq, f_amp[0], f_amp[1])

    # Optional post-filter crop (used to keep 0-0.6s analysis window while
    # preserving pre/post buffer for stable filtering).
    if crop_window is not None:
        crop_tmin, crop_tmax = crop_window
        if times is not None:
            mask = (times >= crop_tmin) & (times <= crop_tmax)
            if int(mask.sum()) < 10:
                return None, None
            theta_ph = theta_ph[:, mask]
            gamma_am = gamma_am[:, mask]
        else:
            start = max(0, int(round(crop_tmin * sfreq)))
            stop = max(start + 1, int(round(crop_tmax * sfreq)) + 1)
            stop = min(stop, theta_ph.shape[1])
            theta_ph = theta_ph[:, start:stop]
            gamma_am = gamma_am[:, start:stop]
            if theta_ph.shape[1] < 10:
                return None, None

    return theta_ph, gamma_am


def _pac_from_precomputed(theta_phase, gamma_amp, cfg, return_details=False):
    """Compute PAC from precomputed phase and amplitude arrays.

    Uses trial-concatenation: all epochs are concatenated into a single
    long time-series before computing MI. This gives many more data points
    per phase bin, producing a more stable MI estimate -- especially
    important for short 0.6 s epochs with only ~2-5 theta cycles each.

    The MI is then z-scored against 500 circular-shift surrogates.

    Parameters
    ----------
    return_details : bool
        If True, return (z, details_dict) with intermediate data for diagnostics.
    """
    pac_cfg = cfg.get('pac', {})
    n_surr = pac_cfg.get('surrogates', 500)
    n_epochs = min(theta_phase.shape[0], gamma_amp.shape[0])

    if n_epochs == 0:
        if return_details:
            return np.nan, None
        return np.nan

    # Concatenate all trials into a single long vector
    concat_phase = theta_phase[:n_epochs].ravel()
    concat_amp = gamma_amp[:n_epochs].ravel()

    # Compute MI on concatenated data (with details for plotting)
    mi_real, phase_bins, mean_amp_raw = _modulation_index(
        concat_phase, concat_amp, return_details=True)

    # Surrogate distribution: circular-shift the concatenated amplitude
    n_samples = len(concat_phase)
    shift_range = n_samples // 4
    rng = np.random.RandomState(42)  # Fixed seed
    surr_mis = []
    for _ in range(n_surr):
        shift = rng.randint(shift_range, n_samples - shift_range)
        shifted_amp = np.roll(concat_amp, shift)
        surr_mis.append(_modulation_index(concat_phase, shifted_amp))

    surr_mis = np.array(surr_mis)
    mean_surr = np.mean(surr_mis)
    std_surr = np.std(surr_mis)
    z = (mi_real - mean_surr) / std_surr if std_surr > 0 else 0

    if return_details:
        details = {
            'phase_bins': phase_bins,
            'mean_amp': mean_amp_raw,
            'mi_real': mi_real,
            'surr_mis': surr_mis,
            'z': z,
        }
        return z, details
    return z


def _modulation_index(theta_phase, gamma_amp, n_bins=12, return_details=False):
    """Tort et al. (2010) Modulation Index.

    Uses 12 phase bins (30 deg each) for better bin coverage with
    trial-concatenated data from short 0.6 s epochs.

    Parameters
    ----------
    return_details : bool
        If True, return (mi, phase_bins, mean_amp_raw) for diagnostic plots.
    """
    phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    mean_amp = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (theta_phase >= phase_bins[i]) & (theta_phase < phase_bins[i + 1])
        if mask.sum() > 0:
            mean_amp[i] = gamma_amp[mask].mean()
    mean_amp_raw = mean_amp.copy()  # keep unnormalized for plotting
    if mean_amp.sum() == 0:
        if return_details:
            return 0, phase_bins, mean_amp_raw
        return 0
    mean_amp = mean_amp / mean_amp.sum()
    mean_amp = np.clip(mean_amp, 1e-10, None)
    uniform = np.ones(n_bins) / n_bins
    kl_div = np.sum(mean_amp * np.log(mean_amp / uniform))
    mi = kl_div / np.log(n_bins)
    if return_details:
        return mi, phase_bins, mean_amp_raw
    return mi


# compute_pac dispatcher REMOVED — _pac_from_precomputed() is the sole PAC path.
# No tensorpac dependency and no alternate PAC path. Fully deterministic.




def load_individual_peaks(output_dir):
    """Load individual theta (ITF) peak frequencies.
    
    Returns
    -------
    itf_map : dict  {(subject, block): f_theta}
    """
    itf_map = {}
    feat_file = output_dir / "theta_freq_features.csv"
    if feat_file.exists():
        df = pd.read_csv(feat_file)
        itf_map = {(row['subject'], row['block']): row['f_theta']
                   for _, row in df.iterrows() if not np.isnan(row['f_theta'])}

    return itf_map


def main():
    parser = argparse.ArgumentParser(description='PAC analysis')
    parser.add_argument('--no-plots', action='store_true', help='Skip diagnostic figures')
    args, _ = parser.parse_known_args()
    do_plots = not args.no_plots

    cfg = load_config()
    blocks = cfg.get('blocks', [1, 5])
    epochs_dir = pipeline_dir / "outputs" / "derivatives" / "epochs_clean"
    features_dir = pipeline_dir / "outputs" / "features"

    # Load Individual Peak Frequencies
    itf_map = load_individual_peaks(features_dir)
    if itf_map:
        print(f"Loaded {len(itf_map)} individual theta peaks (descriptive only).")
    else:
        print("No individual theta peaks found.")

    subjects = get_subjects_with_blocks(epochs_dir, 'pac', blocks)
    if not subjects:
        subjects = discover_subjects(
            epochs_dir=epochs_dir,
            blocks=blocks,
            epoch_type='pac',
            require_all_blocks=False,
        )
    if not subjects:
        print("No PAC epoch files found.")
        return

    # Load node channels from config
    h1_nodes = cfg.get('h1_nodes', {})
    if not h1_nodes:
        raise KeyError("Missing required h1_nodes config in study.yml.")

    between_node_channels = h1_nodes
    print("Node channels:")
    for node_name, channels in between_node_channels.items():
        print(f"  {node_name}: {channels}")

    between_rows = []
    pac_plot_data = {}  # subj -> {block: details_dict}
    qc_records = []

    for subj in subjects:
        print(f"\n{'='*50}")
        print(f"  PAC Analysis: {subj}")
        print(f"{'='*50}")

        for block in blocks:
            print(f"\n  --- Block {block} ---")
            epochs = load_block_epochs(subj, block, 'pac', epochs_dir)

            if epochs is None or 'eeg' not in epochs.get_channel_types():
                print(f"  No data for block {block}")
                continue

            sfreq = epochs.info['sfreq']
            times = epochs.times
            
            # Canonical theta band (fixed 4-8 Hz)
            pac_cfg = copy.deepcopy(cfg)
            fixed_band = cfg.get('pac', {}).get('phase_band', [4, 8])
            # Optional runtime override for surrogate count
            env_surr = os.environ.get("EEG_PAC_SURROGATES")
            if env_surr:
                pac_cfg.setdefault('pac', {})['surrogates'] = int(env_surr)
            
            print(f'  Theta Band: {fixed_band[0]}-{fixed_band[1]} Hz (canonical)')
            print(f"  PAC analysis window (post-filter crop): {PAC_ANALYSIS_WINDOW[0]:.1f}-{PAC_ANALYSIS_WINDOW[1]:.1f} s")

            # === 4a: Between-region PAC ===
            between_row = {'subject': subj, 'block': block}
            node_signals = {
                node: get_node_signal(epochs, chs)
                for node, chs in between_node_channels.items()
            }

            # Precompute phase/amplitude per node (speeds up all-pair analysis)
            f_pha = pac_cfg.get('pac', {}).get('phase_band', [4, 8])
            f_amp = pac_cfg.get('pac', {}).get('amp_band', [55, 85])
            node_phase = {}
            node_amp = {}
            for node, sig in node_signals.items():
                if sig is None:
                    continue
                theta_phase, gamma_amp = _precompute_phase_amp(
                    sig,
                    sfreq,
                    f_pha,
                    f_amp,
                    times=times,
                    crop_window=PAC_ANALYSIS_WINDOW,
                )
                if theta_phase is not None and gamma_amp is not None:
                    node_phase[node] = theta_phase
                    node_amp[node] = gamma_amp

            for phase_node, amp_node in BETWEEN_PAC_PAIRS:
                col = f'pac_between_{phase_node}_{amp_node}'
                phase_arr = node_phase.get(phase_node)
                amp_arr = node_amp.get(amp_node)

                is_primary = (phase_node, amp_node) in H1_PAIRS
                if is_primary:
                    role = "PRIMARY"
                else:
                    role = "other"

                if phase_arr is not None and amp_arr is not None:
                    # Get details for primary pair (for diagnostic figure)
                    if is_primary and do_plots:
                        pac_between, details = _pac_from_precomputed(
                            phase_arr, amp_arr, pac_cfg, return_details=True)
                        if details is not None:
                            pac_plot_data.setdefault(subj, {})[block] = details
                    else:
                        pac_between = _pac_from_precomputed(phase_arr, amp_arr, pac_cfg)
                    between_row[col] = pac_between
                    print(f"  Between PAC ({phase_node}->{amp_node}): "
                          f"{pac_between:.3f} [{role}]")
                    if is_primary:
                        qc_records.append({
                            "subject": subj,
                            "block": block,
                            "input_file": str(getattr(epochs, "filenames", [""])[0]),
                            "phase_node": phase_node,
                            "amp_node": amp_node,
                            "n_trials": int(phase_arr.shape[0]),
                            "phase_band": list(f_pha),
                            "amp_band": list(f_amp),
                            "method": "trial_concatenation",
                            "pac_z": float(pac_between),
                        })
                else:
                    if is_primary:
                        raise RuntimeError(
                            f"Missing primary PAC inputs for {subj} block {block}: "
                            f"{phase_node}->{amp_node}"
                        )
                    between_row[col] = np.nan

            between_rows.append(between_row)

    # Save between-region PAC (H1 theta-gamma)
    if between_rows:
        df_between = pd.DataFrame(between_rows)
        out_between = _tag_path("pac_between_features.csv", OUTPUT_DIR)
        df_between.to_csv(out_between, index=False)
        print(f"\nSaved theta-gamma PAC to {out_between}")
        print(df_between.to_string(index=False))
        for record in qc_records:
            save_step_qc(
                "10_pac",
                record["subject"],
                record["block"],
                {
                    "status": "PASS",
                    "input_file": record["input_file"],
                    "input_hash": file_sha256(record["input_file"]) if record["input_file"] else "UNKNOWN",
                    "output_file": str(out_between),
                    "output_hash": file_sha256(out_between),
                    "parameters_used": {
                        "phase_band": record["phase_band"],
                        "amp_band": record["amp_band"],
                    },
                    "step_specific": {
                        "n_trials": record["n_trials"],
                        "freq_bands": {
                            "phase": record["phase_band"],
                            "amplitude": record["amp_band"],
                        },
                        "method": record["method"],
                        "primary_pair": f"{record['phase_node']}->{record['amp_node']}",
                        "pac_z": record["pac_z"],
                    },
                },
            )

    # Generate per-subject diagnostic figures
    if do_plots:
        for subj, block_details in pac_plot_data.items():
            if block_details:
                _plot_pac_diagnostic(subj, block_details)


if __name__ == "__main__":
    main()
