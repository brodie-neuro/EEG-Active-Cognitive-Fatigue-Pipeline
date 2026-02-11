# eeg_pipeline/analysis/10_erp_p3b.py
"""
Step 1 -- P3b ERP Analysis (H5)

Extracts P3b amplitude (300-500 ms) and peak latency from target trials
at centroparietal electrodes (CP node). Outputs long format:
one row per subject x block.

Reference: post_processing_EEG_plan_v2.docx, Step 1
"""
import sys
from pathlib import Path
import mne
import numpy as np
import pandas as pd

pipeline_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pipeline_dir))

from src.utils_io import load_config
from src.utils_features import (
    load_block_epochs, get_subjects_with_blocks, available_channels
)
from src.utils_config import get_param

OUTPUT_DIR = pipeline_dir / "outputs" / "features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# P3b parameters from config
_p3b_cfg = get_param('p3b', default={})
P3B_TMIN = _p3b_cfg.get('tmin_peak', 0.300)
P3B_TMAX = _p3b_cfg.get('tmax_peak', 0.500)
P3B_CHANNELS = _p3b_cfg.get('channels', ['Pz'])


def extract_p3b(epochs, ch_picks=None):
    """
    Extract P3b mean amplitude and peak latency from target epochs.

    Parameters
    ----------
    epochs : mne.Epochs
    ch_picks : list of str, optional
        Channels to average over. Defaults to CP node.

    Returns
    -------
    dict with 'p3b_amplitude_uV' and 'p3b_latency_ms'
    """
    if ch_picks is None:
        ch_picks = P3B_CHANNELS

    avail = available_channels(ch_picks, epochs.ch_names)
    if not avail:
        avail = [epochs.ch_names[0]]

    # Get data: (n_epochs, n_channels, n_times)
    data = epochs.copy().pick(avail).get_data()
    times = epochs.times

    # P3b window mask
    t_mask = (times >= P3B_TMIN) & (times <= P3B_TMAX)

    # Grand average ERP (mean across epochs, then across channels)
    erp = data.mean(axis=0).mean(axis=0)  # (n_times,)

    # Mean amplitude in P3b window (in uV)
    p3b_amp = erp[t_mask].mean() * 1e6

    # Peak latency (in ms)
    window_erp = erp[t_mask]
    peak_idx = np.argmax(window_erp)
    p3b_lat = times[t_mask][peak_idx] * 1000

    return {
        'p3b_amplitude_uV': p3b_amp,
        'p3b_latency_ms': p3b_lat,
    }


def main():
    cfg = load_config()
    blocks = cfg.get('blocks', [1, 5])
    epochs_dir = pipeline_dir / "outputs" / "derivatives" / "epochs_clean"

    if not epochs_dir.exists():
        print(f"Epochs directory not found: {epochs_dir}. Run preprocessing first.")
        return

    subjects = get_subjects_with_blocks(epochs_dir, 'p3b', blocks)

    if not subjects:
        legacy = sorted(epochs_dir.glob("*_p3b_clean-epo.fif"))
        subjects = sorted(set(f.stem.split("_")[0] for f in legacy))
        if subjects:
            print(f"Using legacy single-block files for {len(subjects)} subjects")

    if not subjects:
        print("No P3b epoch files found.")
        return

    rows = []

    for subj in subjects:
        print(f"--- P3b Analysis: {subj} ---")

        for block in blocks:
            epochs = load_block_epochs(subj, block, 'p3b', epochs_dir)
            if epochs is None:
                print(f"  Block {block}: no data")
                continue

            if 'eeg' not in epochs.get_channel_types():
                print(f"  Block {block}: no EEG channels")
                continue

            # Filter to target trials if available (supports BrainVision "Comment/" prefix)
            target_keys = [k for k in epochs.event_id.keys() if 'stim/target' in k]
            if target_keys:
                target_epochs = epochs[target_keys]
            else:
                target_epochs = epochs

            p3b = extract_p3b(target_epochs)
            rows.append({
                'subject': subj,
                'block': block,
                'p3b_amp_uV': p3b['p3b_amplitude_uV'],
                'p3b_lat_ms': p3b['p3b_latency_ms'],
            })
            print(f"  Block {block}: amp={p3b['p3b_amplitude_uV']:.2f} uV, "
                  f"lat={p3b['p3b_latency_ms']:.0f} ms")

    if rows:
        df = pd.DataFrame(rows)
        output_file = OUTPUT_DIR / "p3b_features.csv"
        df.to_csv(output_file, index=False)
        print(f"\nSaved P3b features (long format) to {output_file}")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
