"""
Visualise analysis outputs from the v2 post-processing pipeline.
Publication-quality figures on scalp montages where appropriate.
"""
import sys
from pathlib import Path
import mne
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.collections import PathCollection

pipeline_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pipeline_dir))

OUTPUT_DIR = pipeline_dir / "outputs" / "analysis_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_DIR = pipeline_dir / "outputs" / "features"
EPOCHS_DIR = pipeline_dir / "outputs" / "derivatives" / "epochs_clean"


# ─── Scalp node positions (2D projection) ───────────────────────────
# 9 canonical regions on a standard head circle
NODE_POSITIONS = {
    'LF': (-0.30,  0.35),   # Left Frontal
    'CF': ( 0.00,  0.40),   # Centre Frontal
    'RF': ( 0.30,  0.35),   # Right Frontal
    'LC': (-0.35,  0.00),   # Left Central
    'CC': ( 0.00,  0.00),   # Centre Central
    'RC': ( 0.35,  0.00),   # Right Central
    'LP': (-0.30, -0.35),   # Left Parietal
    'CP': ( 0.00, -0.40),   # Centre Parietal
    'RP': ( 0.30, -0.35),   # Right Parietal
}

NODE_LABELS = {
    'LF': 'L-Front', 'CF': 'Mid-Front', 'RF': 'R-Front',
    'LC': 'L-Cent',  'CC': 'Mid-Cent',  'RC': 'R-Cent',
    'LP': 'L-Par',   'CP': 'Mid-Par',   'RP': 'R-Par',
}


def draw_head(ax, linewidth=1.5):
    """Draw a standard head outline with nose and ears."""
    # Head circle
    theta = np.linspace(0, 2 * np.pi, 100)
    r = 0.5
    ax.plot(r * np.cos(theta), r * np.sin(theta), 'k-', linewidth=linewidth)

    # Nose
    nose_x = [-.08, 0, .08]
    nose_y = [r, r + 0.08, r]
    ax.plot(nose_x, nose_y, 'k-', linewidth=linewidth)

    # Ears
    for sign in [-1, 1]:
        ear_x = sign * np.array([r, r + 0.03, r + 0.04, r + 0.03, r])
        ear_y = np.array([0.08, 0.06, 0.0, -0.06, -0.08])
        ax.plot(ear_x, ear_y, 'k-', linewidth=linewidth)

    ax.set_xlim(-0.65, 0.65)
    ax.set_ylim(-0.60, 0.65)
    ax.set_aspect('equal')
    ax.axis('off')


def plot_pac_montage():
    """PAC values on a scalp montage — 3 panels: B1, B5, ΔPAC."""
    df = pd.read_csv(FEATURES_DIR / "pac_local_features.csv")

    nodes = ['LF', 'CF', 'RF', 'LC', 'CC', 'RC', 'LP', 'CP', 'RP']
    node_cols = [f'pac_{n}' for n in nodes]
    available = [c for c in node_cols if c in df.columns]
    avail_nodes = [n for n, c in zip(nodes, node_cols) if c in df.columns]

    if not available:
        print("No PAC node data found")
        return

    b1 = df[df['block'] == 1][available].values
    b5 = df[df['block'] == 5][available].values
    if len(b1) == 0 or len(b5) == 0:
        print("Missing block data for PAC montage")
        return

    b1_vals = b1.mean(axis=0)
    b5_vals = b5.mean(axis=0)
    delta = b5_vals - b1_vals

    vmax = max(abs(b1_vals).max(), abs(b5_vals).max(), abs(delta).max())
    if vmax == 0:
        vmax = 0.01

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, vals, title in zip(axes,
                                [b1_vals, b5_vals, delta],
                                ['Block 1 (Baseline)', 'Block 5 (Fatigued)', 'ΔPAC (B5 − B1)']):
        draw_head(ax)

        xs = [NODE_POSITIONS[n][0] for n in avail_nodes]
        ys = [NODE_POSITIONS[n][1] for n in avail_nodes]

        sc = ax.scatter(xs, ys, c=vals, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                       s=800, edgecolors='black', linewidth=1.2, zorder=5)

        for i, node in enumerate(avail_nodes):
            ax.text(xs[i], ys[i] - 0.08, NODE_LABELS.get(node, node),
                   ha='center', va='top', fontsize=8, fontweight='bold')
            ax.text(xs[i], ys[i], f'{vals[i]:.4f}',
                   ha='center', va='center', fontsize=7, zorder=6)

        ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02, label='PAC (MI)')

    fig.suptitle('Local Theta–Gamma PAC — Scalp Montage', fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "pac_montage.png", dpi=150, bbox_inches='tight')
    print("Saved PAC scalp montage")
    plt.close(fig)


def plot_between_pac_montage():
    """Between-region PAC shown as directional curved arc on a scalp montage."""
    df = pd.read_csv(FEATURES_DIR / "pac_between_features.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    titles = ['Block 1 (Baseline)', 'Block 5 (Fatigued)']
    colors_block = ['#2196F3', '#F44336']

    for i, (ax, block, title, col) in enumerate(
            zip(axes, [1, 5], titles, colors_block)):

        draw_head(ax)

        row = df[df['block'] == block]
        if len(row) == 0:
            ax.set_title(f'{title}\nNo data')
            continue

        pac_val = row['pac_between_RF_RP'].values[0]

        # Plot all 9 nodes
        for node, (x, y) in NODE_POSITIONS.items():
            node_col = '#E0E0E0'
            edge_col = '#606060'
            z = 5
            
            # Highlight Source and Target
            if node == 'RF':
                node_col = '#4CAF50'
                edge_col = 'black'
                z = 10
            elif node == 'RP':
                node_col = '#FF9800'
                edge_col = 'black'
                z = 10
            
            ax.scatter(x, y, c=node_col, s=350, edgecolors=edge_col,
                      linewidth=1.5, zorder=z)
            
            # Use cleaner labels (e.g. Fz instead of Mid-Front, or just region codes)
            # Map standard 9-node codes to descriptive but short labels
            short_labels = {
                'LF': 'L-Front', 'CF': 'Fz',   'RF': 'R-Front',
                'LC': 'L-Cent',  'CC': 'Cz',   'RC': 'R-Cent',
                'LP': 'L-Par',   'CP': 'Pz',   'RP': 'R-Par'
            }
            label = short_labels.get(node, node)
            
            ax.text(x, y - 0.09, label,
                   ha='center', va='top', fontsize=9, fontweight='bold', zorder=z)

        # Draw curved directional arrow from RF to RP
        rf = NODE_POSITIONS['RF']
        rp = NODE_POSITIONS['RP']

        # Width scales with magnitude (min width 1.5)
        width = max(2.0, abs(pac_val) * 4000)
        
        # Connection patch with arc (subtler curve)
        arrow = FancyArrowPatch(rf, rp, 
                                connectionstyle="arc3,rad=-0.15", # Subtler curve
                                arrowstyle='Simple,tail_width=2.0,head_width=10,head_length=10',
                                color=col, linewidth=width,
                                alpha=0.9, zorder=8)
        ax.add_patch(arrow)

        # Display value in text (smaller, cleaner)
        ax.text(0, -0.65, f'PAC = {pac_val:.5f}',
               ha='center', va='center', fontsize=11, fontweight='bold',
               color='black', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#4CAF50', 
                   markersize=10, label='Source: RF (θ phase)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF9800', 
                   markersize=10, label='Target: RP (γ amp)'),
    ]
    axes[0].legend(handles=legend_elements, loc='upper left', fontsize=9, frameon=True)

    # Move suptitle higher to avoid overlap
    fig.suptitle('Between-Region PAC: Frontal $\\theta$ $\\rightarrow$ Parietal $\\gamma$',
                fontsize=16, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95]) # Reserve space at top
    fig.savefig(OUTPUT_DIR / "pac_between_montage.png", dpi=150, bbox_inches='tight')
    print("Saved between-region PAC montage")
    plt.close(fig)


def plot_psd_specparam():
    """PSD spectrum with specparam fit: Block 1 (blue) vs Block 5 (red)."""
    from specparam import SpectralModel

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {1: '#2196F3', 5: '#F44336'}
    labels = {1: 'Block 1 (Baseline)', 5: 'Block 5 (Fatigued)'}

    # --- Theta range (left panel) ---
    ax = axes[0]
    for block in [1, 5]:
        fname = EPOCHS_DIR / f"sub-TEST01_block{block}_pac_clean-epo.fif"
        if not fname.exists():
            continue

        epochs = mne.read_epochs(str(fname), verbose=False)
        # Frontal midline channels
        picks = [ch for ch in ['Fz', 'FCz', 'Cz', 'F1', 'F2'] if ch in epochs.ch_names]
        if not picks:
            picks = epochs.ch_names[:3]

        psd = epochs.copy().pick(picks).compute_psd(
            method='welch', fmin=1, fmax=30, n_fft=256, verbose=False)
        freqs = psd.freqs
        psds = psd.get_data().mean(axis=(0, 1)) * 1e12  # µV²/Hz

        ax.semilogy(freqs, psds, color=colors[block], linewidth=2,
                   label=labels[block], alpha=0.8)

        # Fit specparam and overlay
        sm = SpectralModel(peak_width_limits=[1, 8], max_n_peaks=6,
                          min_peak_height=0.1, aperiodic_mode='fixed', verbose=False)
        try:
            sm.fit(freqs, np.log10(psds), [1, 30])
            model_fit = 10 ** sm.modeled_spectrum_
            ax.semilogy(freqs, model_fit, color=colors[block], linewidth=1.5,
                       linestyle='--', alpha=0.6)
        except Exception:
            pass

    ax.axvspan(4, 8, alpha=0.1, color='green', label='θ band (4–8 Hz)')
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Power (µV²/Hz)', fontsize=12)
    ax.set_title('PSD + Specparam Fit: Frontal Midline', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Alpha range (right panel) ---
    ax = axes[1]
    for block in [1, 5]:
        fname = EPOCHS_DIR / f"sub-TEST01_block{block}_pac_clean-epo.fif"
        if not fname.exists():
            continue

        epochs = mne.read_epochs(str(fname), verbose=False)
        picks = [ch for ch in ['Oz', 'O1', 'O2', 'POz', 'Pz'] if ch in epochs.ch_names]
        if not picks:
            picks = epochs.ch_names[:3]

        psd = epochs.copy().pick(picks).compute_psd(
            method='welch', fmin=1, fmax=30, n_fft=256, verbose=False)
        freqs = psd.freqs
        psds = psd.get_data().mean(axis=(0, 1)) * 1e12

        ax.semilogy(freqs, psds, color=colors[block], linewidth=2,
                   label=labels[block], alpha=0.8)

        sm = SpectralModel(peak_width_limits=[1, 8], max_n_peaks=6,
                          min_peak_height=0.1, aperiodic_mode='fixed', verbose=False)
        try:
            sm.fit(freqs, np.log10(psds), [1, 30])
            model_fit = 10 ** sm.modeled_spectrum_
            ax.semilogy(freqs, model_fit, color=colors[block], linewidth=1.5,
                       linestyle='--', alpha=0.6)
        except Exception:
            pass

    ax.axvspan(8, 13, alpha=0.1, color='purple', label='α band (8–13 Hz)')
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Power (µV²/Hz)', fontsize=12)
    ax.set_title('PSD + Specparam Fit: Posterior (IAF)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "psd_specparam_overlay.png", dpi=150)
    print("Saved PSD + specparam overlay")
    plt.close(fig)


def plot_p3b_erp_filtered():
    """Plot P3b ERP: Block 1 vs Block 5 (Low-pass filtered at 20Hz)."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    colors = {'1': '#2196F3', '5': '#F44336'}
    labels = {'1': 'Block 1 (Baseline)', '5': 'Block 5 (Fatigued)'}

    for block in [1, 5]:
        fname = EPOCHS_DIR / f"sub-TEST01_block{block}_p3b_clean-epo.fif"
        if not fname.exists():
            continue

        epochs = mne.read_epochs(str(fname), verbose=False)
        picks = [ch for ch in ['Pz', 'CPz', 'POz', 'P1', 'P2'] if ch in epochs.ch_names]
        if not picks:
            picks = epochs.ch_names[:5]

        # Apply 20Hz low-pass filter for visualization smoothing
        epochs_plot = epochs.copy().filter(l_freq=None, h_freq=20.0, n_jobs=-1, verbose=False)
        data = epochs_plot.pick(picks).get_data()
        erp = data.mean(axis=0).mean(axis=0) * 1e6  # µV
        times = epochs.times * 1000  # ms

        ax.plot(times, erp, color=colors[str(block)], linewidth=2,
                label=labels[str(block)])

    # P3b window
    ax.axvspan(300, 500, alpha=0.15, color='green', label='P3b Window (300-500 ms)')
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')

    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Amplitude (µV)', fontsize=12)
    ax.set_title('P3b ERP (20Hz LP)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_xlim(-200, 800)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "p3b_erp_filtered.png", dpi=150)
    print(f"Saved filtered P3b ERP plot")
    plt.close(fig)


if __name__ == "__main__":
    print("=== Generating Analysis Visualisations (v2) ===\n")
    plot_psd_specparam()
    plot_pac_montage()
    plot_between_pac_montage()
    plot_p3b_erp_filtered()
    print(f"\nAll figures saved to {OUTPUT_DIR}")
