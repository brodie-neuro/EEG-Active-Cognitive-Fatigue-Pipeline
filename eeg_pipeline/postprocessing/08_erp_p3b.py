# postprocessing/08_erp_p3b.py
"""
P3b ERP Analysis

Extracts P3b amplitude (300-500 ms) and peak latency from target trials
at centroparietal electrodes. Outputs long format:
one row per subject x block.

Uses the dedicated low-cut ERP branch (`p3b_erp`) so conservative P3b
estimation does not alter the main 1 Hz oscillatory pipeline.

Reference: post_processing_EEG_plan_v2.docx, Step 1
"""
import argparse
import os
import sys
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
from src.utils_features import (
    load_block_epochs, get_subjects_with_blocks, available_channels
)
from src.utils_config import get_param

OUTPUT_DIR = pipeline_dir / "outputs" / "features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = pipeline_dir / "outputs" / "figures" / "erp_p3b"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# -- Plotting constants --
_CLR_B1 = '#1E88E5'
_CLR_B5 = '#E53935'
_CLR_P3B = '#4CAF50'
_STAT_BOX = dict(boxstyle='round,pad=0.4', facecolor='#ECEFF1', edgecolor='#B0BEC5')


def _style_ax(ax):
    """Apply publication style: remove top/right spines, light grid."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)


def _plot_erp_diagnostic(subj, block_erps, times_ms):
    """Generate a 1x3 ERP diagnostic figure for one subject across blocks.

    Parameters
    ----------
    subj : str
    block_erps : dict  {block: {'erp': array, 'data': (n_trials, n_times), 'features': dict}}
    times_ms : array  time vector in milliseconds
    """
    fig, (ax_erp, ax_bf, ax_cmp) = plt.subplots(1, 3, figsize=(18, 5))
    block_colors = {}
    for b in sorted(block_erps.keys()):
        block_colors[b] = _CLR_B1 if b == min(block_erps.keys()) else _CLR_B5

    # Use the first available block for single-block panels
    first_block = min(block_erps.keys())
    erp0 = block_erps[first_block]['erp']
    feat0 = block_erps[first_block]['features']

    # --- Left: ERP waveform with P3b window ---
    ax_erp.plot(times_ms, erp0, 'k-', lw=1.5, label='Grand avg')
    ax_erp.axvspan(P3B_TMIN * 1000, P3B_TMAX * 1000, alpha=0.2, color=_CLR_P3B, label='P3b window')
    # Diamond at P3b peak
    if not np.isnan(feat0['p3b_latency_ms']):
        ax_erp.plot(feat0['p3b_latency_ms'], feat0['p3b_mean_uV'], 'D',
                    color=_CLR_P3B, ms=10, zorder=5, label=f'FAL @ {feat0["p3b_latency_ms"]:.0f} ms')
    ax_erp.axhline(0, color='gray', lw=0.5, ls='--')
    ax_erp.axvline(0, color='gray', lw=0.5, ls='--')
    ax_erp.set_xlabel('Time (ms)')
    ax_erp.set_ylabel('Amplitude (µV)')
    ax_erp.set_title(f'ERP — Block {first_block}')
    ax_erp.legend(fontsize=7, loc='upper right')
    _style_ax(ax_erp)

    # --- Middle: Butterfly (all trials) ---
    trial_data = block_erps[first_block]['data']
    if trial_data is not None and trial_data.shape[0] > 0:
        for t in range(trial_data.shape[0]):
            ax_bf.plot(times_ms, trial_data[t] * 1e6, color='lightgray', lw=0.3, alpha=0.5)
        ax_bf.plot(times_ms, erp0, 'k-', lw=2, label='Grand avg')
    ax_bf.axhline(0, color='gray', lw=0.5, ls='--')
    ax_bf.axvline(0, color='gray', lw=0.5, ls='--')
    ax_bf.set_xlabel('Time (ms)')
    ax_bf.set_ylabel('Amplitude (µV)')
    ax_bf.set_title(f'Butterfly — Block {first_block} ({trial_data.shape[0]} trials)')
    _style_ax(ax_bf)

    # --- Right: B1 vs B5 overlay + difference wave ---
    sorted_blocks = sorted(block_erps.keys())
    if len(sorted_blocks) >= 2:
        b1, b5 = sorted_blocks[0], sorted_blocks[-1]
        erp_b1 = block_erps[b1]['erp']
        erp_b5 = block_erps[b5]['erp']
        ax_cmp.plot(times_ms, erp_b1, color=_CLR_B1, lw=1.5, label=f'Block {b1}')
        ax_cmp.plot(times_ms, erp_b5, color=_CLR_B5, lw=1.5, label=f'Block {b5}')
        diff = erp_b1 - erp_b5
        ax_cmp.plot(times_ms, diff, color='gray', lw=1.2, ls='--', label='Difference')
        # Stats box
        feat_b1 = block_erps[b1]['features']
        feat_b5 = block_erps[b5]['features']
        delta_amp = feat_b1['p3b_mean_uV'] - feat_b5['p3b_mean_uV']
        delta_lat = feat_b1['p3b_latency_ms'] - feat_b5['p3b_latency_ms']
        stats_txt = (f"B{b1} mean: {feat_b1['p3b_mean_uV']:.1f} µV @ {feat_b1['p3b_latency_ms']:.0f} ms\n"
                     f"B{b5} mean: {feat_b5['p3b_mean_uV']:.1f} µV @ {feat_b5['p3b_latency_ms']:.0f} ms\n"
                     f"Δamp: {delta_amp:+.1f} µV  Δlat: {delta_lat:+.0f} ms")
        ax_cmp.text(0.02, 0.98, stats_txt, transform=ax_cmp.transAxes,
                    fontsize=7, va='top', bbox=_STAT_BOX)
    else:
        b = sorted_blocks[0]
        ax_cmp.plot(times_ms, block_erps[b]['erp'], color=block_colors[b], lw=1.5,
                    label=f'Block {b}')
        ax_cmp.text(0.5, 0.5, 'Single block only', transform=ax_cmp.transAxes,
                    ha='center', va='center', fontsize=12, color='gray')

    ax_cmp.axhline(0, color='gray', lw=0.5, ls='--')
    ax_cmp.axvline(0, color='gray', lw=0.5, ls='--')
    ax_cmp.set_xlabel('Time (ms)')
    ax_cmp.set_ylabel('Amplitude (µV)')
    ax_cmp.set_title('Block Comparison (P3b)')
    ax_cmp.legend(fontsize=7, loc='upper right')
    _style_ax(ax_cmp)

    fig.suptitle(f'ERP P3b Diagnostic — {subj}', fontsize=14, fontweight='bold')
    fig.tight_layout()
    out = FIG_DIR / f"erp_p3b_{subj}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved figure: {out}")


# P3b parameters from config
_erp_branch_cfg = get_param('erp_branch', default={}) or {}
ERP_EPOCH_TYPE = _erp_branch_cfg.get('epoch_type', 'p3b_erp')
_p3b_cfg = get_param('p3b', default={})
P3B_TMIN = _p3b_cfg.get('tmin_peak', 0.300)
P3B_TMAX = _p3b_cfg.get('tmax_peak', 0.500)
P3B_CHANNELS = _p3b_cfg.get('channels', ['Pz'])


def _fractional_area_latency(erp_win, times_win, fraction=0.5):
    """
    Fractional area latency (Luck, 2014).

    Find the timepoint at which `fraction` (default 50%) of the total
    positive area under the ERP in the window has been reached.
    More robust than argmax, which can be dragged by slow positive drift.

    Parameters
    ----------
    erp_win : ndarray
        ERP values within the P3b window.
    times_win : ndarray
        Time values (seconds) for the window.
    fraction : float
        Area fraction (default 0.5 = 50%).

    Returns
    -------
    float : latency in seconds at which the fraction of area is reached.
    """
    pos_erp = np.clip(erp_win, 0, None)
    total = pos_erp.sum()
    if total == 0:
        return times_win[len(times_win) // 2]
    cumulative = np.cumsum(pos_erp)
    idx = int(np.searchsorted(cumulative, fraction * total))
    idx = min(idx, len(times_win) - 1)
    return times_win[idx]


def extract_p3b(epochs, ch_picks=None):
    """
    Extract P3b mean amplitude and fractional area latency from target epochs.

    Parameters
    ----------
    epochs : mne.Epochs
    ch_picks : list of str, optional
        Channels to average over. Defaults to CP node.

    Returns
    -------
    dict with P3b mean-window amplitude (gold standard) and
    fractional area latency (50%, Luck 2014).
    """
    if ch_picks is None:
        ch_picks = P3B_CHANNELS

    avail = available_channels(ch_picks, epochs.ch_names)
    if not avail:
        raise ValueError(
            f"No P3b cluster channels {ch_picks} found in data channels {epochs.ch_names}. "
            f"Check p3b.channels in parameters.json."
        )

    # Get data: (n_epochs, n_channels, n_times)
    data = epochs.copy().pick(avail).get_data()
    times = epochs.times

    # Grand average ERP (mean across epochs, then across channels)
    erp = data.mean(axis=0).mean(axis=0)  # (n_times,)

    t_mask = (times >= P3B_TMIN) & (times <= P3B_TMAX)
    if not np.any(t_mask):
        p3b_mean = np.nan
        p3b_lat = np.nan
    else:
        win_erp = erp[t_mask]
        win_times = times[t_mask]
        p3b_mean = float(win_erp.mean() * 1e6)
        fal_time = _fractional_area_latency(win_erp, win_times, fraction=0.5)
        p3b_lat = float(fal_time * 1000.0)

    return {
        'p3b_amplitude_uV': p3b_mean,  # legacy key (mean window amplitude)
        'p3b_mean_uV': p3b_mean,
        'p3b_latency_ms': p3b_lat,     # fractional area latency (50%)
        '_erp_uV': erp * 1e6,   # full ERP waveform in µV (for diagnostics)
        '_times_ms': times * 1000,  # time vector in ms
        '_trial_data': data.mean(axis=1),  # (n_trials, n_times) channel-averaged
    }


def main():
    parser = argparse.ArgumentParser(description='P3b ERP analysis')
    parser.add_argument('--no-plots', action='store_true', help='Skip diagnostic figures')
    args, _ = parser.parse_known_args()
    do_plots = not args.no_plots

    cfg = load_config()
    blocks = cfg.get('blocks', [1, 5])
    epochs_dir = pipeline_dir / "outputs" / "derivatives" / "epochs_clean"

    subjects = get_subjects_with_blocks(epochs_dir, ERP_EPOCH_TYPE, blocks)
    if not subjects:
        subjects = discover_subjects(
            epochs_dir=epochs_dir,
            blocks=blocks,
            epoch_type=ERP_EPOCH_TYPE,
            require_all_blocks=False,
        )

    if not subjects:
        print(f"No {ERP_EPOCH_TYPE} epoch files found.")
        return

    rows = []

    for subj in subjects:
        print(f"--- P3b Analysis: {subj} ---")
        block_erps = {}  # accumulate for diagnostic figure

        for block in blocks:
            epochs = load_block_epochs(subj, block, ERP_EPOCH_TYPE, epochs_dir)
            if epochs is None:
                print(f"  Block {block}: no data")
                continue

            if 'eeg' not in epochs.get_channel_types():
                print(f"  Block {block}: no EEG channels")
                continue

            # Filter to target trials if available (supports BrainVision "Comment/" prefix)
            target_keys = [k for k in epochs.event_id.keys() if 'stim/target' in k.lower()]
            if target_keys:
                target_epochs = epochs[target_keys]
            else:
                target_epochs = epochs

            p3b = extract_p3b(target_epochs)
            rows.append({
                'subject': subj,
                'block': block,
                'p3b_amp_uV': p3b['p3b_amplitude_uV'],
                'p3b_mean_uV': p3b['p3b_mean_uV'],
                'p3b_lat_ms': p3b['p3b_latency_ms'],
            })
            print(
                f"  Block {block}: "
                f"P3b mean={p3b['p3b_mean_uV']:.2f} uV, FAL={p3b['p3b_latency_ms']:.0f} ms"
            )

            # Accumulate for diagnostic plot
            if do_plots:
                block_erps[block] = {
                    'erp': p3b['_erp_uV'],
                    'data': p3b['_trial_data'],
                    'features': p3b,
                }

        # Generate per-subject diagnostic figure after all blocks
        if do_plots and block_erps:
            times_ms = p3b['_times_ms']
            _plot_erp_diagnostic(subj, block_erps, times_ms)

    if rows:
        df = pd.DataFrame(rows)
        output_file = OUTPUT_DIR / "p3b_features.csv"
        df.to_csv(output_file, index=False)
        print(f"\nSaved P3b features (long format) to {output_file}")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
