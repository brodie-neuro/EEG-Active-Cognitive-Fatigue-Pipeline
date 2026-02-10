# eeg_pipeline/analysis/12_peak_frequencies.py
"""
Steps 2 & 3 — IAF (H6) and Theta Peak Frequency (H2) via specparam

Step 2: Individual Alpha Frequency from resting-state posterior electrodes.
Step 3: Task-related theta peak frequency from CF node during maintenance.

Both use specparam to separate periodic from aperiodic components.
Outputs long format: one row per subject × block.

Reference: post_processing_EEG_plan_v2.docx, Steps 2 & 3
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
    load_block_epochs, get_subjects_with_blocks,
    available_channels, get_node_channels
)

try:
    from specparam import SpectralModel
    SPECPARAM_AVAILABLE = True
except ImportError:
    SPECPARAM_AVAILABLE = False
    print("Warning: specparam not installed. Install with: pip install specparam")

OUTPUT_DIR = pipeline_dir / "outputs" / "features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_peak_frequency(psd_data, freqs, freq_range, cfg):
    """
    Use specparam to extract periodic peak frequency from a power spectrum.

    Returns
    -------
    dict with 'peak_freq', 'peak_power', 'aperiodic_exponent'
    """
    sp_cfg = cfg.get('specparam', {})
    freq_fit_range = sp_cfg.get('freq_range', [1, 30])

    sm = SpectralModel(
        peak_width_limits=sp_cfg.get('peak_width_limits', [1, 8]),
        max_n_peaks=sp_cfg.get('max_n_peaks', 6),
        min_peak_height=sp_cfg.get('min_peak_height', 0.1),
        aperiodic_mode='fixed',
        verbose=False
    )

    try:
        sm.fit(freqs, psd_data, freq_fit_range)
    except Exception as e:
        print(f"  specparam fit failed: {e}")
        return {'peak_freq': np.nan, 'peak_power': np.nan, 'aperiodic_exponent': np.nan}

    # New specparam API: results.params.aperiodic.params = [offset, exponent]
    try:
        ap = sm.results.params.aperiodic.params
        aperiodic_exp = ap[-1] if len(ap) > 1 else np.nan
    except Exception:
        aperiodic_exp = np.nan

    # peaks: results.params.periodic.params = [[freq, power, width], ...]
    try:
        peaks = sm.results.params.periodic.params
        if peaks.ndim == 1:
            peaks = peaks.reshape(1, -1) if len(peaks) > 0 else np.array([]).reshape(0, 3)
    except Exception:
        peaks = np.array([]).reshape(0, 3)

    if len(peaks) == 0:
        return {'peak_freq': np.nan, 'peak_power': np.nan, 'aperiodic_exponent': aperiodic_exp}

    in_range = [(p[0], p[1]) for p in peaks if freq_range[0] <= p[0] <= freq_range[1]]
    if not in_range:
        return {'peak_freq': np.nan, 'peak_power': np.nan, 'aperiodic_exponent': aperiodic_exp}

    best = max(in_range, key=lambda x: x[1])
    return {
        'peak_freq': best[0],
        'peak_power': best[1],
        'aperiodic_exponent': aperiodic_exp
    }


def compute_psd_for_channels(data_source, ch_picks, fmin=1, fmax=30):
    """Compute PSD and return (mean_power, freqs) for specparam."""
    avail = available_channels(ch_picks, data_source.ch_names)
    if not avail:
        avail = [data_source.ch_names[0]]

    psd_obj = data_source.compute_psd(
        fmin=fmin, fmax=fmax, picks=avail, verbose=False
    )
    power = psd_obj.get_data().mean(axis=tuple(range(psd_obj.get_data().ndim - 1)))
    freqs = psd_obj.freqs
    return power, freqs


def extract_iaf(raw, cfg):
    """Extract IAF from resting-state data using specparam on posterior channels."""
    posterior_chs = ['Oz', 'O1', 'O2', 'POz', 'PO3', 'PO4', 'PO7', 'PO8']
    power, freqs = compute_psd_for_channels(raw, posterior_chs, fmin=1, fmax=30)

    if SPECPARAM_AVAILABLE:
        result = extract_peak_frequency(power, freqs, (7, 14), cfg)
        return result['peak_freq'], result['aperiodic_exponent']
    else:
        mask = (freqs >= 7) & (freqs <= 14)
        if mask.sum() == 0:
            return np.nan, np.nan
        peak_idx = np.argmax(power[mask])
        return freqs[mask][peak_idx], np.nan


def extract_theta_freq(epochs, cfg):
    """Extract theta peak frequency from CF node using specparam."""
    cf_channels = get_node_channels('CF', cfg)
    if not cf_channels:
        cf_channels = ['Fz', 'FCz', 'Cz']

    power, freqs = compute_psd_for_channels(epochs, cf_channels, fmin=1, fmax=30)

    if SPECPARAM_AVAILABLE:
        result = extract_peak_frequency(power, freqs, (4, 8), cfg)
        return result['peak_freq'], result['aperiodic_exponent']
    else:
        mask = (freqs >= 4) & (freqs <= 8)
        if mask.sum() == 0:
            return np.nan, np.nan
        peak_idx = np.argmax(power[mask])
        return freqs[mask][peak_idx], np.nan


def main():
    cfg = load_config()
    blocks = cfg.get('blocks', [1, 5])

    epochs_dir = pipeline_dir / "outputs" / "derivatives" / "epochs_clean"
    raw_dir = pipeline_dir / "outputs" / "derivatives" / "asr_cleaned_raw"

    if not epochs_dir.exists():
        print(f"Epochs directory not found: {epochs_dir}")
        return

    subjects = get_subjects_with_blocks(epochs_dir, 'pac', blocks)
    if not subjects:
        legacy = sorted(epochs_dir.glob("*_pac_clean-epo.fif"))
        subjects = sorted(set(f.stem.split("_")[0] for f in legacy))
    if not subjects:
        print("No epoch files found.")
        return

    # --- Theta frequency: long format (subject × block) ---
    theta_rows = []

    for subj in subjects:
        print(f"--- Frequency Analysis: {subj} ---")

        for block in blocks:
            epochs = load_block_epochs(subj, block, 'pac', epochs_dir)
            if epochs is None or 'eeg' not in epochs.get_channel_types():
                continue

            f_theta, aperiodic_exp = extract_theta_freq(epochs, cfg)
            theta_rows.append({
                'subject': subj,
                'block': block,
                'f_theta': f_theta,
                'aperiodic_exp': aperiodic_exp,
            })
            if not np.isnan(f_theta):
                print(f"  Block {block}: fθ={f_theta:.2f} Hz, aperiodic={aperiodic_exp:.3f}")
            else:
                print(f"  Block {block}: fθ N/A")

    if theta_rows:
        df_theta = pd.DataFrame(theta_rows)
        output_theta = OUTPUT_DIR / "theta_freq_features.csv"
        df_theta.to_csv(output_theta, index=False)
        print(f"\nSaved theta frequency features (long format) to {output_theta}")
        print(df_theta.to_string(index=False))

    # --- IAF: separate output (pre/post rest, not block-based) ---
    iaf_rows = []

    for subj in subjects:
        for label, suffix in [('pre', f'block{blocks[0]}'), ('post', f'block{blocks[-1]}')]:
            raw_file = raw_dir / f"{subj}_{suffix}_asr-raw.fif"
            if not raw_file.exists():
                raw_file = raw_dir / f"{subj}_asr-raw.fif"

            if raw_file.exists():
                try:
                    raw = mne.io.read_raw_fif(raw_file, preload=True, verbose=False)
                    iaf, aperiodic = extract_iaf(raw, cfg)
                    iaf_rows.append({
                        'subject': subj,
                        'timepoint': label,
                        'iaf': iaf,
                        'aperiodic_exp': aperiodic,
                    })
                except Exception as e:
                    print(f"  IAF ({label}) failed: {e}")

    if iaf_rows:
        df_iaf = pd.DataFrame(iaf_rows)
        output_iaf = OUTPUT_DIR / "iaf_features.csv"
        df_iaf.to_csv(output_iaf, index=False)
        print(f"\nSaved IAF features (long format) to {output_iaf}")
        print(df_iaf.to_string(index=False))


if __name__ == "__main__":
    main()
