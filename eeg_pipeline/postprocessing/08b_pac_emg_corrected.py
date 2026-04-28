# eeg_pipeline/postprocessing/08b_pac_emg_corrected.py
"""
Step 08b - PAC with EMG-corrected gamma (sensitivity analysis).

Identical to step 08 PAC computation EXCEPT:
  Before computing gamma amplitude, regress out EMG PC1 from each trial's
  parietal signal at every time point. This removes the trial-level EMG
  contribution from the gamma envelope BEFORE the Hilbert transform.

Method:
  For each time point t across N trials:
    parietal_signal[trial, t] = beta0[t] + beta1[t] * emg_pc1[trial] + residual[trial, t]
    corrected_signal[trial, t] = residual[trial, t] + mean(parietal_signal[:, t])

  Then: gamma_amp = |Hilbert(bandpass(corrected_signal, 55-85 Hz))|
  PAC is computed identically to step 08 using corrected gamma amp.

Outputs:
  - pac_between_emg_corrected.csv (same format as pac_between_features.csv)
  - Figure: pac_emg_corrected_{subj}.png

Does NOT modify step 08 or any existing outputs.
"""
import os
import sys
import copy
import argparse
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
from scipy import stats as sp_stats
from scipy.signal import hilbert as sp_hilbert

pipeline_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pipeline_dir))

from src.utils_io import load_config, discover_subjects
from src.utils_config import get_param
from src.utils_determinism import file_sha256, save_step_qc
from src.utils_features import (
    load_block_epochs, get_subjects_with_blocks,
    available_channels, filter_excluded_channels,
)

OUTPUT_DIR = pipeline_dir / "outputs" / "features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = pipeline_dir / "outputs" / "figures" / "pac"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PAC_ANALYSIS_WINDOW = (0.0, 0.6)
H1_PAIRS = [("C_broad_F", "C_broad_P")]


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


def _theta_phase(signal_1d, sfreq, lo, hi):
    filt = mne.filter.filter_data(
        signal_1d, sfreq, lo, hi, verbose=False,
        method='iir', iir_params=dict(order=4, ftype='butter'))
    return np.angle(sp_hilbert(filt))


def _gamma_amp(signal_1d, sfreq, lo, hi):
    filt = mne.filter.filter_data(
        signal_1d, sfreq, lo, hi, verbose=False,
        method='fir')
    return np.abs(sp_hilbert(filt))


def _regress_emg_from_signal(node_signal, emg_pc1):
    """Regress EMG PC1 out of the node signal at each time point.

    Parameters
    ----------
    node_signal : (n_trials, n_times)
    emg_pc1 : (n_trials,)

    Returns
    -------
    corrected : (n_trials, n_times) — residual + column mean
    """
    n_trials, n_times = node_signal.shape
    corrected = np.zeros_like(node_signal)

    for t in range(n_times):
        y = node_signal[:, t]
        x = emg_pc1
        mask = np.isfinite(y) & np.isfinite(x)
        if mask.sum() < 5:
            corrected[:, t] = y
            continue
        slope, intercept, _, _, _ = sp_stats.linregress(x[mask], y[mask])
        predicted = intercept + slope * x
        residual = y - predicted
        corrected[:, t] = residual + np.mean(y)  # preserve scale

    return corrected


def _modulation_index(theta_phase, gamma_amp, n_bins=12):
    """Tort et al. (2010) MI."""
    phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    mean_amp = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (theta_phase >= phase_bins[i]) & (theta_phase < phase_bins[i + 1])
        if mask.sum() > 0:
            mean_amp[i] = gamma_amp[mask].mean()
    if mean_amp.sum() == 0:
        return 0
    mean_amp = mean_amp / mean_amp.sum()
    mean_amp = np.clip(mean_amp, 1e-10, None)
    uniform = np.ones(n_bins) / n_bins
    kl_div = np.sum(mean_amp * np.log(mean_amp / uniform))
    return kl_div / np.log(n_bins)


def _pac_from_precomputed(theta_phase, gamma_amp, n_surr):
    """Trial-concatenated MI with surrogates."""
    n_epochs = min(theta_phase.shape[0], gamma_amp.shape[0])
    if n_epochs == 0:
        return np.nan

    concat_phase = theta_phase[:n_epochs].ravel()
    concat_amp = gamma_amp[:n_epochs].ravel()

    mi_real = _modulation_index(concat_phase, concat_amp)

    n_samples = len(concat_phase)
    shift_range = n_samples // 4
    rng = np.random.RandomState(42)
    surr_mis = []
    for _ in range(n_surr):
        shift = rng.randint(shift_range, n_samples - shift_range)
        shifted_amp = np.roll(concat_amp, shift)
        surr_mis.append(_modulation_index(concat_phase, shifted_amp))

    surr_mis = np.array(surr_mis)
    mean_surr = np.mean(surr_mis)
    std_surr = np.std(surr_mis)
    z = (mi_real - mean_surr) / std_surr if std_surr > 0 else 0
    return z


def main():
    parser = argparse.ArgumentParser(
        description="PAC with EMG-corrected gamma (step 08b sensitivity analysis)"
    )
    parser.add_argument(
        "--subject",
        type=str,
        default="",
        help="Optional subject filter, e.g. sub-p001 or sub-p001,sub-p002.",
    )
    args = parser.parse_args()
    if args.subject.strip():
        os.environ["EEG_SUBJECT_FILTER"] = args.subject.strip()

    cfg = load_config()
    blocks = cfg.get('blocks', [1, 5])
    epochs_dir = pipeline_dir / "outputs" / "derivatives" / "epochs_clean"

    subjects = get_subjects_with_blocks(epochs_dir, 'pac', blocks)
    if not subjects:
        subjects = discover_subjects(
            epochs_dir=epochs_dir, blocks=blocks,
            epoch_type='pac', require_all_blocks=False,
        )
    if not subjects:
        print("No PAC epoch files found.")
        return

    # Load node channels
    h1_nodes = cfg.get('h1_nodes', {})
    if not h1_nodes:
        raise KeyError("Missing h1_nodes in study.yml.")

    # Load EMG covariates
    emg_path = OUTPUT_DIR / "emg_covariates.csv"
    if not emg_path.exists():
        raise FileNotFoundError(f"{emg_path} not found. Run step 13 first.")
    emg_df = pd.read_csv(emg_path)

    pac_cfg = get_param('pac', default={})
    f_pha = pac_cfg.get('phase_band', [4, 8])
    f_amp = pac_cfg.get('amp_band', [55, 85])
    n_surr = pac_cfg.get('n_surrogates', 500)
    print(f"  PAC params (from config): phase={f_pha}, amp={f_amp}, surrogates={n_surr}")

    between_rows = []

    for subj in subjects:
        print(f"\n{'='*60}")
        print(f"  PAC EMG-Corrected: {subj}")
        print(f"{'='*60}")

        for block in blocks:
            print(f"\n  --- Block {block} ---")
            epochs = load_block_epochs(subj, block, 'pac', epochs_dir)
            if epochs is None or 'eeg' not in epochs.get_channel_types():
                print(f"  No data for block {block}")
                continue

            sfreq = epochs.info['sfreq']
            times = epochs.times

            # Get EMG PC1 for this block
            emg_block = emg_df[
                (emg_df["subject"] == subj) & (emg_df["block"] == block)
            ].sort_values("trial").reset_index(drop=True)

            if emg_block.empty:
                print(f"  No EMG data for block {block}")
                continue

            emg_pc1 = emg_block["emg_pc1"].values

            # Extract node signals
            for phase_node, amp_node in H1_PAIRS:
                phase_chs = h1_nodes.get(phase_node, [])
                amp_chs = h1_nodes.get(amp_node, [])

                phase_signal = get_node_signal(epochs, phase_chs)
                amp_signal = get_node_signal(epochs, amp_chs)

                if phase_signal is None or amp_signal is None:
                    print(f"  Missing node signals for {phase_node}->{amp_node}")
                    continue

                n_epochs = min(phase_signal.shape[0], amp_signal.shape[0], len(emg_pc1))
                phase_signal = phase_signal[:n_epochs]
                amp_signal = amp_signal[:n_epochs]
                emg_pc1_matched = emg_pc1[:n_epochs]

                print(f"  {n_epochs} trials matched (epochs + EMG)")

                # --- Regress EMG out of parietal signal ---
                amp_corrected = _regress_emg_from_signal(amp_signal, emg_pc1_matched)
                print(f"  EMG regressed out of {amp_node} signal at each time point")

                # --- Compute theta phase (unchanged) ---
                theta_ph = np.zeros((n_epochs, phase_signal.shape[1]))
                for ep in range(n_epochs):
                    theta_ph[ep] = _theta_phase(phase_signal[ep], sfreq, f_pha[0], f_pha[1])

                # --- Compute gamma amplitude from CORRECTED signal ---
                gamma_am = np.zeros((n_epochs, amp_corrected.shape[1]))
                for ep in range(n_epochs):
                    gamma_am[ep] = _gamma_amp(amp_corrected[ep], sfreq, f_amp[0], f_amp[1])

                # Crop to analysis window
                crop_mask = (times >= PAC_ANALYSIS_WINDOW[0]) & (times <= PAC_ANALYSIS_WINDOW[1])
                theta_ph = theta_ph[:, crop_mask]
                gamma_am = gamma_am[:, crop_mask]

                if theta_ph.shape[1] < 10:
                    print(f"  Too few samples after crop.")
                    continue

                # --- Compute PAC ---
                z = _pac_from_precomputed(theta_ph, gamma_am, n_surr=n_surr)
                col = f"pac_between_{phase_node}_{amp_node}"
                print(f"  EMG-corrected PAC ({phase_node}->{amp_node}): Z = {z:.3f}")

                between_rows.append({
                    "subject": subj,
                    "block": block,
                    col: z,
                })

    # Save
    if between_rows:
        df = pd.DataFrame(between_rows)
        out = OUTPUT_DIR / "pac_between_emg_corrected.csv"
        df.to_csv(out, index=False)
        print(f"\nSaved EMG-corrected PAC to {out}")
        print(df.to_string(index=False))

        # Compare with original
        orig_path = OUTPUT_DIR / "pac_between_features.csv"
        if orig_path.exists():
            orig_df = pd.read_csv(orig_path)
            pac_col = [c for c in orig_df.columns if c.startswith("pac_between")][0]
            corr_col = [c for c in df.columns if c.startswith("pac_between")][0]

            comp = pd.merge(
                orig_df[["subject", "block", pac_col]].rename(columns={pac_col: "pac_original"}),
                df[["subject", "block", corr_col]].rename(columns={corr_col: "pac_corrected"}),
                on=["subject", "block"],
            )
            comp["delta"] = comp["pac_corrected"] - comp["pac_original"]

            print(f"\n{'='*60}")
            print("  Comparison: Original vs EMG-Corrected PAC")
            print(f"{'='*60}")
            print(comp.to_string(index=False))

            r, p = sp_stats.pearsonr(comp["pac_original"], comp["pac_corrected"])
            print(f"\n  Correlation original vs corrected: r={r:.3f}, p={p:.4f}")
            print(f"  Mean absolute delta: {comp['delta'].abs().mean():.3f}")


if __name__ == "__main__":
    main()
