# eeg_pipeline/postprocessing/18_alpha_gamma_pac.py
"""
Step 18 — Between-region alpha-gamma phase-amplitude coupling (exploratory/descriptive).

Computes frontal alpha (8-13 Hz) phase × parietal gamma (55-85 Hz) amplitude
PAC using the same trial-concatenated MI + circular-shift surrogates method
as step 10 (theta-gamma PAC).

  Phase: C_broad_F (Fz, FCz, F3, F4, FC1, FC2, FC3, FC4)
  Amplitude: C_broad_P (Pz, POz, P3, P4, P1, P2, PO3, PO4)

This mirrors the frontoparietal topology of the confirmatory theta-gamma
analysis (H1) but indexes a distinct cross-frequency interaction:
alpha-mediated executive gating rather than theta-mediated temporal
organisation of WM content.

This is a strictly descriptive state-characterisation measure, not a
confirmatory hypothesis test. It provides an a priori predictor for
a powered follow-up study.

Motivation:
  - Mangan & Kourtis (2026, JoCN) proposed that cognitive fatigue
    disrupts top-down alpha-mediated gating of WM representations.
  - Miller et al. (2018, Neuron) — alpha/beta as top-down gating of
    gamma content in WM 2.0 framework.

Method: identical to step 10
  - Trial-concatenated MI (Tort et al., 2010)
  - 500 circular-shift surrogates, seed=42
  - z-scored MI
  - Analysis window: 0.0-0.6 s (stimulus processing)

Outputs:
  - alpha_gamma_pac_features.csv
  - Per-subject diagnostic figures (optional)

Does NOT modify step 10 outputs or any confirmatory analyses.
"""
import os
import sys
import argparse
import re
from pathlib import Path

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
from scipy.signal import hilbert as sp_hilbert

pipeline_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pipeline_dir))

from src.utils_io import load_config, discover_subjects
from src.utils_features import (
    load_block_epochs, get_subjects_with_blocks,
    available_channels, filter_excluded_channels,
)

OUTPUT_DIR = pipeline_dir / "outputs" / "features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = pipeline_dir / "outputs" / "figures" / "pac"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──
ALPHA_BAND = (8.0, 13.0)
GAMMA_BAND = (55.0, 85.0)
N_SURROGATES = 500
RANDOM_SEED = 42
PAC_WINDOW = (0.0, 0.6)
N_BINS = 12  # match step 10 (30 deg bins)



OUTPUT_TAG = os.environ.get("EEG_OUTPUT_TAG", "").strip()


def _tag_path(base, directory):
    s, ext = Path(base).stem, Path(base).suffix
    if OUTPUT_TAG:
        s = f"{s}_{OUTPUT_TAG}"
    return directory / f"{s}{ext}"


# ── Signal extraction ──

def get_node_signal(epochs, node_channels):
    """Extract mean signal across node channels."""
    node_channels, _ = filter_excluded_channels(node_channels)
    avail = available_channels(node_channels, epochs.ch_names)
    if avail:
        types = epochs.get_channel_types(picks=avail)
        avail = [ch for ch, ch_type in zip(avail, types) if ch_type == 'eeg']
    if not avail:
        return None
    data = epochs.copy().pick(avail).get_data()
    return data.mean(axis=1)  # (n_epochs, n_times)


# ── Filtering ──

def _alpha_phase(signal_1d, sfreq):
    """Extract instantaneous alpha phase (IIR Butterworth, 4th order)."""
    filt = mne.filter.filter_data(
        signal_1d, sfreq, ALPHA_BAND[0], ALPHA_BAND[1], verbose=False,
        method='iir', iir_params=dict(order=4, ftype='butter'))
    return np.angle(sp_hilbert(filt))


def _gamma_amp(signal_1d, sfreq):
    """Extract instantaneous gamma amplitude envelope (FIR)."""
    filt = mne.filter.filter_data(
        signal_1d, sfreq, GAMMA_BAND[0], GAMMA_BAND[1], verbose=False,
        method='fir')
    return np.abs(sp_hilbert(filt))


# ── MI computation (identical to step 10) ──

def _modulation_index(phase, amplitude, n_bins=N_BINS):
    """Tort et al. (2010) modulation index."""
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    mean_amp = np.zeros(n_bins)
    for b in range(n_bins):
        mask = (phase >= bin_edges[b]) & (phase < bin_edges[b + 1])
        if mask.sum() > 0:
            mean_amp[b] = amplitude[mask].mean()
    total = mean_amp.sum()
    if total == 0:
        return 0.0
    p = mean_amp / total
    # KL divergence from uniform
    uniform = np.ones(n_bins) / n_bins
    # Avoid log(0)
    p_safe = np.clip(p, 1e-30, None)
    kl = np.sum(p_safe * np.log(p_safe / uniform))
    mi = kl / np.log(n_bins)
    return mi


def _precompute_phase_amp(node_signal, sfreq, times=None, crop_window=None):
    """Precompute alpha phase and gamma amplitude for ALL trials."""
    n_epochs = node_signal.shape[0]
    if n_epochs == 0:
        return None, None

    alpha_ph = np.zeros((n_epochs, node_signal.shape[1]), dtype=float)
    gamma_am = np.zeros((n_epochs, node_signal.shape[1]), dtype=float)
    for ep in range(n_epochs):
        alpha_ph[ep] = _alpha_phase(node_signal[ep], sfreq)
        gamma_am[ep] = _gamma_amp(node_signal[ep], sfreq)

    # Optional post-filter crop
    if crop_window is not None and times is not None:
        crop_tmin, crop_tmax = crop_window
        mask = (times >= crop_tmin) & (times <= crop_tmax)
        if int(mask.sum()) < 10:
            return None, None
        alpha_ph = alpha_ph[:, mask]
        gamma_am = gamma_am[:, mask]

    return alpha_ph, gamma_am


def compute_alpha_gamma_pac(phase_signal, amp_signal, sfreq, times, crop_window=None):
    """Trial-concatenated alpha-gamma MI + surrogates.

    Matches step 10 method: filter full buffered epochs, THEN crop to
    analysis window. This preserves filter edge stability.
    Returns z-scored MI.
    """
    n_epochs = phase_signal.shape[0]

    # Filter each full-length epoch first, then crop
    all_phase = []
    all_amp = []
    for ep in range(n_epochs):
        phase = _alpha_phase(phase_signal[ep], sfreq)
        amp = _gamma_amp(amp_signal[ep], sfreq)

        # Post-filter crop (preserves filter edge stability)
        if crop_window is not None and times is not None:
            mask = (times >= crop_window[0]) & (times <= crop_window[1])
            phase = phase[mask]
            amp = amp[mask]

        all_phase.append(phase)
        all_amp.append(amp)

    if not all_phase:
        return np.nan, {}

    concat_phase = np.concatenate(all_phase)
    concat_amp = np.concatenate(all_amp)

    # Observed MI
    mi_real = _modulation_index(concat_phase, concat_amp)

    # Surrogate distribution
    n_samples = len(concat_phase)
    shift_range = n_samples // 4
    rng = np.random.RandomState(RANDOM_SEED)
    surr_mis = []
    for _ in range(N_SURROGATES):
        shift = rng.randint(shift_range, n_samples - shift_range)
        shifted_amp = np.roll(concat_amp, shift)
        surr_mis.append(_modulation_index(concat_phase, shifted_amp))

    mean_surr = np.mean(surr_mis)
    std_surr = np.std(surr_mis)
    z = (mi_real - mean_surr) / std_surr if std_surr > 0 else 0.0

    details = {
        'mi_real': mi_real,
        'mean_surr': mean_surr,
        'std_surr': std_surr,
        'z': z,
        'n_surrogates': N_SURROGATES,
        'concat_phase': concat_phase,
        'concat_amp': concat_amp,
    }
    return z, details


# ── Diagnostic figure ──

def _plot_diagnostic(subj, variant_label, block_details):
    """Simple phase-amplitude histogram for each block."""
    n_blocks = len(block_details)
    if n_blocks == 0:
        return

    fig, axes = plt.subplots(1, n_blocks, figsize=(6 * n_blocks, 4.5),
                              sharey=True)
    if n_blocks == 1:
        axes = [axes]

    colors = {1: '#1976D2', 5: '#D32F2F'}

    for ax, (block, det) in zip(axes, sorted(block_details.items())):
        phase = det['concat_phase']
        amp = det['concat_amp']
        bin_edges = np.linspace(-np.pi, np.pi, N_BINS + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        mean_amp = np.zeros(N_BINS)
        for b in range(N_BINS):
            mask = (phase >= bin_edges[b]) & (phase < bin_edges[b + 1])
            if mask.sum() > 0:
                mean_amp[b] = amp[mask].mean() * 1e9  # nV

        c = colors.get(block, '#455A64')
        ax.bar(np.degrees(bin_centers), mean_amp,
               width=360 / N_BINS * 0.8, color=c, alpha=0.65,
               edgecolor=c, linewidth=0.5)
        ax.set_xlabel('α phase (°)', fontsize=10)
        ax.set_title(f'Block {block} (z={det["z"]:.2f})',
                     fontweight='bold', fontsize=11)
        ax.set_xlim(-200, 200)


    axes[0].set_ylabel('γ amplitude (nV)', fontsize=10)
    fig.suptitle(f'{subj} — {variant_label}\n'
                 f'α ({ALPHA_BAND[0]}-{ALPHA_BAND[1]} Hz) × '
                 f'γ ({GAMMA_BAND[0]}-{GAMMA_BAND[1]} Hz)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    safe_label = variant_label.replace(' ', '_').replace('->', '_to_').replace('→', '_to_')
    safe_label = re.sub(r'[<>:"/\\|?*]+', '_', safe_label).strip('_')
    out = FIG_DIR / f"alpha_gamma_pac_{safe_label}_{subj}.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


import logging
import time

LOG_DIR = pipeline_dir / "outputs" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def _setup_logging():
    """Configure logging to both console and file."""
    log_file = LOG_DIR / "18_alpha_gamma_pac.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler(),
        ]
    )
    return logging.getLogger(__name__)


def _save_incremental(rows, out_csv, logger):
    """Save current results to CSV after each participant."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    logger.info(f"Incremental save: {len(rows)} rows -> {out_csv}")


# ── Main ──

def main():
    parser = argparse.ArgumentParser(
        description='Alpha-gamma PAC (exploratory/descriptive)')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip diagnostic figures')
    parser.add_argument('--subject', type=str, default=None,
                        help='Process a single subject')
    args, _ = parser.parse_known_args()
    do_plots = not args.no_plots

    logger = _setup_logging()
    logger.info("=" * 60)
    logger.info("Step 18: Alpha-Gamma PAC (Exploratory/Descriptive)")
    logger.info("=" * 60)

    cfg = load_config()
    blocks = cfg.get('blocks', [1, 5])
    epochs_dir = pipeline_dir / "outputs" / "derivatives" / "epochs_clean"

    subjects = get_subjects_with_blocks(epochs_dir, 'pac', blocks)
    if not subjects:
        subjects = discover_subjects(
            epochs_dir=epochs_dir, blocks=blocks,
            epoch_type='pac', require_all_blocks=False)
    if not subjects:
        logger.warning("No PAC epoch files found.")
        return

    if args.subject:
        if args.subject not in subjects:
            logger.warning(f"{args.subject} not found in {subjects}")
            return
        subjects = [args.subject]

    logger.info(f"Subjects: {subjects}")
    logger.info(f"Blocks: {blocks}")

    # Load node channels
    h1_nodes = cfg.get('h1_nodes', {})
    frontal_chs = h1_nodes.get('C_broad_F', [])
    parietal_chs = h1_nodes.get('C_broad_P', [])

    # Single between-region PAC variant: frontal alpha → parietal gamma
    variants = [
        {
            'label': 'frontal_alpha_to_parietal_gamma',
            'role': 'EXPLORATORY',
            'phase_chs': frontal_chs,
            'phase_desc': 'C_broad_F',
            'amp_chs': parietal_chs,
            'amp_desc': 'C_broad_P',
        },
    ]

    out_csv = _tag_path("alpha_gamma_pac_features.csv", OUTPUT_DIR)
    rows = []
    plot_data = {}

    for subj_idx, subj in enumerate(subjects, 1):
        t_start = time.time()
        logger.info(f"\n{'='*55}")
        logger.info(f"  [{subj_idx}/{len(subjects)}] Alpha-Gamma PAC: {subj}")
        logger.info(f"{'='*55}")

        for block in blocks:
            logger.info(f"  --- Block {block} ---")
            epochs = load_block_epochs(subj, block, 'pac', epochs_dir)

            if epochs is None or 'eeg' not in epochs.get_channel_types():
                logger.warning(f"  No data for {subj} block {block}")
                continue

            sfreq = epochs.info['sfreq']
            times = epochs.times

            for var in variants:
                logger.info(f"    Variant: {var['label']}")

                # Extract phase and amplitude node signals
                phase_sig = get_node_signal(epochs, var['phase_chs'])
                amp_sig = get_node_signal(epochs, var['amp_chs'])

                if phase_sig is None:
                    logger.warning(f"    No phase channels for"
                                   f" {var['phase_desc']}")
                    rows.append({
                        'subject': subj, 'block': block,
                        'variant': var['label'], 'role': var['role'],
                        'z_score': np.nan, 'mi_real': np.nan,
                        'n_trials': 0,
                        'phase_band': list(ALPHA_BAND),
                        'amp_band': list(GAMMA_BAND),
                    })
                    continue

                if amp_sig is None:
                    logger.warning(f"    No amplitude channels for"
                                   f" {var['amp_desc']}")
                    rows.append({
                        'subject': subj, 'block': block,
                        'variant': var['label'], 'role': var['role'],
                        'z_score': np.nan, 'mi_real': np.nan,
                        'n_trials': 0,
                        'phase_band': list(ALPHA_BAND),
                        'amp_band': list(GAMMA_BAND),
                    })
                    continue

                # Pass full epochs to compute_alpha_gamma_pac;
                # it filters first, THEN crops to analysis window
                n_trials = phase_sig.shape[0]
                z, details = compute_alpha_gamma_pac(
                    phase_sig, amp_sig, sfreq, times=times,
                    crop_window=PAC_WINDOW)

                logger.info(
                    f"    z = {z:.3f}  (MI = {details.get('mi_real', 0):.6f},"
                    f" {n_trials} trials) [{var['role']}]")

                rows.append({
                    'subject': subj, 'block': block,
                    'variant': var['label'], 'role': var['role'],
                    'z_score': z,
                    'mi_real': details.get('mi_real', np.nan),
                    'mean_surr': details.get('mean_surr', np.nan),
                    'std_surr': details.get('std_surr', np.nan),
                    'n_trials': n_trials,
                    'phase_band': list(ALPHA_BAND),
                    'amp_band': list(GAMMA_BAND),
                    'window': f"{PAC_WINDOW[0]}-{PAC_WINDOW[1]}",
                    'n_surrogates': N_SURROGATES,
                })

                # Store details for plotting
                if do_plots and details:
                    key = (subj, var['label'])
                    plot_data.setdefault(key, {})[block] = details

        # ── Incremental save after each participant ──
        elapsed = time.time() - t_start
        logger.info(f"  {subj} completed in {elapsed:.1f}s")
        _save_incremental(rows, out_csv, logger)

    # ── Final summary ──
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False)
        logger.info(f"\nFinal save: alpha-gamma PAC features -> {out_csv}")
        summary_cols = ['subject', 'block', 'variant', 'role', 'z_score',
                        'mi_real', 'n_trials']
        avail_cols = [c for c in summary_cols if c in df.columns]
        logger.info("\n" + df[avail_cols].to_string(index=False))

    # Generate plots
    if do_plots and plot_data:
        for (subj, variant_label), block_details in plot_data.items():
            _plot_diagnostic(subj, variant_label, block_details)

    logger.info("\nStep 18 complete.")


if __name__ == "__main__":
    main()
