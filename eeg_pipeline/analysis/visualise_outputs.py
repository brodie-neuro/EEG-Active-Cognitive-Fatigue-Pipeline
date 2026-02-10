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


def _discover_subject():
    """Find first available subject from clean epoch files."""
    for pat in ['*_pac_clean-epo.fif', '*_p3b_clean-epo.fif']:
        files = sorted(EPOCHS_DIR.glob(pat))
        if files:
            # Extract subject ID: sub-XXX from sub-XXX_blockN_type_clean-epo.fif
            stem = files[0].stem
            parts = stem.split('_block')
            if parts:
                return parts[0]
    return None

# --- Scalp node positions (2D projection) ---------------------------
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
    """PAC values on a scalp montage -- 3 panels: B1, B5, delta PAC."""
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
                                ['Block 1 (Baseline)', 'Block 5 (Fatigued)', 'dPAC (B5 - B1)']):
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

    fig.suptitle('Local Theta-Gamma PAC -- Scalp Montage', fontsize=15, fontweight='bold', y=1.02)
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

        rows = df[df['block'] == block]
        if len(rows) == 0:
            ax.set_title(f'{title}\nNo data')
            continue

        pac_val = rows['pac_between_RF_RP'].mean()

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
                   markersize=10, label='Source: RF (theta phase)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF9800', 
                   markersize=10, label='Target: RP (gamma amp)'),
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

    subj = _discover_subject()
    if not subj:
        print("No epoch files found for PSD plot")
        return

    # --- Theta range (left panel) ---
    ax = axes[0]
    for block in [1, 5]:
        fname = EPOCHS_DIR / f"{subj}_block{block}_pac_clean-epo.fif"
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
        psds = psd.get_data().mean(axis=(0, 1)) * 1e12  # uV^2/Hz

        ax.semilogy(freqs, psds, color=colors[block], linewidth=2,
                   label=labels[block], alpha=0.8)

        # Fit specparam and overlay
        sm = SpectralModel(peak_width_limits=[1, 8], max_n_peaks=6,
                          min_peak_height=0.1, aperiodic_mode='fixed', verbose=False)
        try:
            sm.fit(freqs, psds, [1, 30])
            model_fit = sm.modeled_spectrum_
            ax.semilogy(freqs, model_fit, color=colors[block], linewidth=1.5,
                       linestyle='--', alpha=0.6)
        except Exception:
            pass

    ax.axvspan(4, 8, alpha=0.1, color='green', label='theta band (4-8 Hz)')
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Power (uV^2/Hz)', fontsize=12)
    ax.set_title('PSD + Specparam Fit: Frontal Midline', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Alpha range (right panel) ---
    ax = axes[1]
    for block in [1, 5]:
        fname = EPOCHS_DIR / f"{subj}_block{block}_pac_clean-epo.fif"
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
            sm.fit(freqs, psds, [1, 30])
            model_fit = sm.modeled_spectrum_
            ax.semilogy(freqs, model_fit, color=colors[block], linewidth=1.5,
                       linestyle='--', alpha=0.6)
        except Exception:
            pass

    # Dynamic band markers from features (or fallback to fixed)
    iaf_file = FEATURES_DIR / "iaf_features.csv"
    theta_file = FEATURES_DIR / "theta_freq_features.csv"
    
    alpha_lo, alpha_hi = 8, 13  # defaults
    theta_lo, theta_hi = 4, 8
    
    if iaf_file.exists():
        df_iaf = pd.read_csv(iaf_file)
        valid = df_iaf.dropna(subset=['iaf'])
        if len(valid) > 0:
            mean_iaf = valid['iaf'].mean()
            alpha_lo = max(6, mean_iaf - 2.0)
            alpha_hi = min(16, mean_iaf + 2.0)
    
    if theta_file.exists():
        df_theta = pd.read_csv(theta_file)
        valid = df_theta.dropna(subset=['f_theta'])
        if len(valid) > 0:
            mean_itf = valid['f_theta'].mean()
            theta_lo = max(2, mean_itf - 2.0)
            theta_hi = min(alpha_lo, mean_itf + 2.0)  # cap at alpha start
    
    ax.axvspan(theta_lo, theta_hi, alpha=0.08, color='blue',
               label=f'theta band ({theta_lo:.0f}-{theta_hi:.0f} Hz)')
    ax.axvspan(alpha_lo, alpha_hi, alpha=0.08, color='purple',
               label=f'alpha band ({alpha_lo:.0f}-{alpha_hi:.0f} Hz)')
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Power (uV^2/Hz)', fontsize=12)
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

    subj = _discover_subject()
    if not subj:
        print("No epoch files found for P3b plot")
        return

    for block in [1, 5]:
        fname = EPOCHS_DIR / f"{subj}_block{block}_p3b_clean-epo.fif"
        if not fname.exists():
            continue

        epochs = mne.read_epochs(str(fname), verbose=False)
        picks = [ch for ch in ['Pz'] if ch in epochs.ch_names]
        if not picks:
            picks = ['CPz'] if 'CPz' in epochs.ch_names else epochs.ch_names[:1]

        # Apply 20Hz low-pass filter for visualization smoothing
        epochs_plot = epochs.copy().filter(l_freq=None, h_freq=20.0, n_jobs=-1, verbose=False)
        data = epochs_plot.pick(picks).get_data()
        erp = data.mean(axis=0).mean(axis=0) * 1e6  # uV
        times = epochs.times * 1000  # ms

        ax.plot(times, erp, color=colors[str(block)], linewidth=2,
                label=labels[str(block)])

    # P3b window
    ax.axvspan(300, 500, alpha=0.15, color='green', label='P3b Window (300-500 ms)')
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')

    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Amplitude (uV)', fontsize=12)
    ax.set_title('P3b ERP (20Hz LP)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_xlim(-200, 800)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "p3b_erp_filtered.png", dpi=150)
    print(f"Saved filtered P3b ERP plot")
    plt.close(fig)

def plot_iaf_comparison():
    """Plot IAF comparison: Pre vs Post fatigue resting-state.
    
    Shows specparam-extracted alpha peaks with Gaussian bandwidth,
    and paired t-test results. No mixed models needed -- just two
    resting-state timepoints.
    """
    from scipy.stats import ttest_rel
    
    iaf_file = FEATURES_DIR / "iaf_features.csv"
    if not iaf_file.exists():
        print("No IAF features found -- skipping IAF comparison plot.")
        return

    df = pd.read_csv(iaf_file)
    
    pre = df[df['timepoint'] == 'pre'].dropna(subset=['iaf'])
    post = df[df['timepoint'] == 'post'].dropna(subset=['iaf'])
    
    if len(pre) == 0 or len(post) == 0:
        print("Insufficient IAF data for comparison -- skipping.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # --- Panel 1: Gaussian peaks overlay (Pre vs Post) ---
    ax1 = axes[0]
    freqs = np.linspace(4, 18, 500)
    
    # Plot individual subject peaks as Gaussians
    for _, row in pre.iterrows():
        peak = row['iaf']
        width = 1.5  # Approximate Gaussian bandwidth (Hz)
        gaussian = np.exp(-0.5 * ((freqs - peak) / width) ** 2)
        ax1.plot(freqs, gaussian, color='#2196F3', alpha=0.3, linewidth=1)
    
    for _, row in post.iterrows():
        peak = row['iaf']
        width = 1.5
        gaussian = np.exp(-0.5 * ((freqs - peak) / width) ** 2)
        ax1.plot(freqs, gaussian, color='#F44336', alpha=0.3, linewidth=1)
    
    # Group mean Gaussians (bold)
    pre_mean = pre['iaf'].mean()
    post_mean = post['iaf'].mean()
    
    ax1.plot(freqs, np.exp(-0.5 * ((freqs - pre_mean) / 1.5) ** 2),
             color='#1565C0', linewidth=3, label=f'Pre (M={pre_mean:.2f} Hz)')
    ax1.plot(freqs, np.exp(-0.5 * ((freqs - post_mean) / 1.5) ** 2),
             color='#C62828', linewidth=3, label=f'Post (M={post_mean:.2f} Hz)')
    
    ax1.set_xlabel('Frequency (Hz)', fontsize=11)
    ax1.set_ylabel('Normalised Power', fontsize=11)
    ax1.set_title('Individual Alpha Peaks', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # --- Panel 2: Paired comparison (lines connecting pre-post) ---
    ax2 = axes[1]
    
    # Merge on subject for paired comparison
    merged = pre.merge(post, on='subject', suffixes=('_pre', '_post'))
    
    if len(merged) > 0:
        for _, row in merged.iterrows():
            ax2.plot([0, 1], [row['iaf_pre'], row['iaf_post']],
                     'o-', color='gray', alpha=0.5, markersize=6)
        
        # Group means
        ax2.plot([0, 1], [merged['iaf_pre'].mean(), merged['iaf_post'].mean()],
                 's-', color='black', linewidth=3, markersize=10, zorder=5,
                 label='Group Mean')
        
        # Paired t-test
        t_stat, p_val = ttest_rel(merged['iaf_pre'], merged['iaf_post'])
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
        
        ax2.set_title(f'IAF Slowing: t={t_stat:.2f}, p={p_val:.3f} {sig}',
                      fontsize=12, fontweight='bold')
    else:
        ax2.set_title('IAF Pre vs Post', fontsize=12, fontweight='bold')
    
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Pre-Fatigue', 'Post-Fatigue'], fontsize=11)
    ax2.set_ylabel('IAF (Hz)', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # --- Panel 3: Distribution (violin/box) ---
    ax3 = axes[2]
    
    pre_vals = pre['iaf'].values
    post_vals = post['iaf'].values
    
    bp = ax3.boxplot([pre_vals, post_vals], positions=[0, 1], widths=0.4,
                     patch_artist=True, showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor='black', markersize=8))
    bp['boxes'][0].set_facecolor('#BBDEFB')
    bp['boxes'][1].set_facecolor('#FFCDD2')
    
    # Individual points
    ax3.scatter(np.zeros(len(pre_vals)) + np.random.normal(0, 0.04, len(pre_vals)),
                pre_vals, color='#1565C0', alpha=0.6, s=30, zorder=5)
    ax3.scatter(np.ones(len(post_vals)) + np.random.normal(0, 0.04, len(post_vals)),
                post_vals, color='#C62828', alpha=0.6, s=30, zorder=5)
    
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(['Pre', 'Post'], fontsize=11)
    ax3.set_ylabel('IAF (Hz)', fontsize=11)
    ax3.set_title('IAF Distribution', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('Individual Alpha Frequency: Pre vs Post Fatigue (Resting State)',
                 fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "iaf_pre_post_comparison.png", dpi=150, bbox_inches='tight')
    print("Saved IAF pre vs post comparison plot")
    plt.close(fig)


if __name__ == "__main__":
    print("=== Generating Analysis Visualisations (v2) ===\n")
    plot_psd_specparam()
    plot_pac_montage()
    plot_between_pac_montage()
    plot_p3b_erp_filtered()
    plot_iaf_comparison()
    print(f"\nAll figures saved to {OUTPUT_DIR}")
