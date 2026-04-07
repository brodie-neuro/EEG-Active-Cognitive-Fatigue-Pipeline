# eeg_pipeline/analysis/20c_emg_pca_covariates.py
"""
EMG PCA covariates for gamma contamination control.

Uses the actual dedicated off-cap EMG sensors to construct a single robust
'Global Muscle Tension' covariate (PC1) to control for jaw clenching and
neck stiffening without over-cleaning the neural data.

Target EMG Sensors (from montage_channel_plan.md):
  Temporalis (BIPOLAR, 2 per side):
    F7 + FT7: Left Temporalis  -> L_TEMP = F7 - FT7
    F8 + FT8: Right Temporalis -> R_TEMP = F8 - FT8
  Posterior Neck (MONOPOLAR, 1 per side):
    TP7: Left Splenius Capitis
    TP8: Right Splenius Capitis

PCA input: 4 derived signals (L_TEMP, R_TEMP, TP7, TP8) at 55-85 Hz envelope.
PC1 captures coordinated global muscle tension across all 4.

Outputs:
  - CSV: outputs/features/emg_covariates.csv (trial-by-trial PC1 scores)
  - CSV: outputs/features/emg_covariates_block.csv (block-level summary)
  - Figure: outputs/analysis_figures/emg_pca_diagnostics_{subj}.png
"""
import os
import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import hilbert as sp_hilbert
from sklearn.decomposition import PCA

pipeline_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pipeline_dir))

from src.utils_io import load_config, discover_subjects, iter_derivative_files
from src.utils_features import load_block_epochs, filter_excluded_channels


def _load_bad_emg(subj):
    """Load bad EMG channels from participant config."""
    cfg_path = pipeline_dir / "config" / "participant_configs" / f"{subj}.json"
    if cfg_path.exists():
        import json
        with open(cfg_path) as f:
            pcfg = json.load(f)
        return set(pcfg.get("known_bad_emg", []))
    return set()

OUTPUT_DIR = pipeline_dir / "outputs" / "features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = pipeline_dir / "outputs" / "analysis_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_TAG = os.environ.get("EEG_OUTPUT_TAG", "").strip()

# --- Constants ---
GAMMA_BAND = (55.0, 85.0)
BUTTER_ORDER = 4
STIM_TMIN, STIM_TMAX = 0.0, 1.0  # PAC window

# Raw EMG channels on the cap (6 physical electrodes)
EMG_RAW = {"L_TEMP_A": "F7", "L_TEMP_B": "FT7",
           "R_TEMP_A": "F8", "R_TEMP_B": "FT8",
           "L_NECK": "TP7", "R_NECK": "TP8"}

# Bipolar derivations + monopolar channels -> 4 PCA inputs
BIPOLAR_PAIRS = [("L_TEMP", "F7", "FT7"),    # L_TEMP = F7 - FT7
                 ("R_TEMP", "F8", "FT8")]    # R_TEMP = F8 - FT8
MONOPOLAR = [("L_NECK", "TP7"), ("R_NECK", "TP8")]


def _tag_path(base: str) -> Path:
    s, ext = Path(base).stem, Path(base).suffix
    if OUTPUT_TAG:
        s = f"{s}_{OUTPUT_TAG}"
    return FIG_DIR / f"{s}{ext}"


def _make_epochs(subj, block, raw_dir, epochs_dir, tmin=0.0, tmax=1.5):
    """Create onset-locked epochs from ICA-cleaned continuous data."""
    raw_candidates = [
        p for p in iter_derivative_files(
            "ica_cleaned_raw", "*_ica-raw.fif", subject=subj
        )
        if f"_block{block}_" in p.name
    ]
    if not raw_candidates:
        raw_candidates = [
            p for p in iter_derivative_files(
                "ica_cleaned_raw", "*_ica*.fif", subject=subj
            )
            if f"_block{block}_" in p.name
        ]
    raw_file = raw_candidates[0] if raw_candidates else None
    if raw_file is None:
        print(f"    No ICA-cleaned raw file for {subj} block {block}")
        return None

    raw = mne.io.read_raw_fif(raw_file, preload=True, verbose=False)
    
    # We load the pac epochs to get exactly the trials kept after autoreject
    pac_epo = load_block_epochs(subj, block, "pac", epochs_dir)
    if pac_epo is None:
        print(f"    No clean pac epochs found")
        return None

    onset_codes = set(pac_epo.events[:, 2])
    clean_samples = set(pac_epo.events[:, 0])
    
    events_raw, event_id = mne.events_from_annotations(raw, verbose=False)
    onset_dict = {k: v for k, v in event_id.items() if v in onset_codes}
    
    if not onset_dict:
        return None

    epochs = mne.Epochs(
        raw, events_raw, onset_dict,
        tmin=tmin, tmax=tmax,
        baseline=None, preload=True, reject=None, verbose=False,
    )

    epoch_samples = epochs.events[:, 0]
    keep = np.array([s in clean_samples for s in epoch_samples])
    drop_idx = np.where(~keep)[0].tolist()
    if drop_idx:
        epochs.drop(drop_idx, reason="not in autoreject clean set")

    print(f"    Created {len(epochs)} onset-locked epochs from {raw_file.name}")
    return epochs


def _bandpass_envelope_trials(epochs, ch_names, sfreq, band, times, tmin_crop, tmax_crop):
    """Compute per-trial mean envelope across specified channels."""
    data = epochs.copy().pick(ch_names).get_data()
    n_trials, n_ch, _ = data.shape
    crop_mask = (times >= tmin_crop) & (times <= tmax_crop)
    
    per_chan_trial = np.full((n_trials, n_ch), np.nan)

    for ch in range(n_ch):
        for i in range(n_trials):
            try:
                sig = data[i, ch, :].astype(np.float64)
                filt = mne.filter.filter_data(
                    sig, sfreq,
                    l_freq=band[0], h_freq=band[1],
                    method="iir",
                    iir_params=dict(order=BUTTER_ORDER, ftype="butter"),
                    verbose=False)
                env = np.abs(sp_hilbert(filt))
                per_chan_trial[i, ch] = float(np.mean(env[crop_mask])) * 1e6 # in microvolts
            except Exception:
                pass
                
    return per_chan_trial


def _plot_pca_diagnostics(subj, block, per_chan_trial, pca, channels, pc1_scores):
    """Diagnostic figure showing PCA loadings and variance explained."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Variance Explained
    ax = axes[0]
    bars = ax.bar(range(1, len(pca.explained_variance_ratio_) + 1), 
                  pca.explained_variance_ratio_ * 100, color='#4CAF50')
    ax.set_xticks(range(1, len(pca.explained_variance_ratio_) + 1))
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Variance Explained (%)')
    ax.set_title('EMG Variance Explained by PCs')
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 2: PC1 Loadings across Sensors
    ax = axes[1]
    pc1_loadings = pca.components_[0]
    colors = ['#1E88E5' if l > 0 else '#E53935' for l in pc1_loadings]
    ax.bar(channels, pc1_loadings, color=colors)
    ax.set_title('PC1 Loadings (Global Tension)')
    ax.set_ylabel('Weight')
    ax.axhline(0, color='k', linewidth=0.8)
    
    # Plot 3: PC1 Score Distribution
    ax = axes[2]
    ax.hist(pc1_scores, bins=20, color='#9C27B0', alpha=0.7, edgecolor='k')
    ax.set_title('Trial PC1 Scores\n(Covariate for Regression)')
    ax.set_xlabel('PC1 Score (Global Muscle Tension)')
    ax.set_ylabel('Trial Count')
    
    plt.suptitle(f"{subj} Block {block} — EMG PCA Diagnostics (55-85 Hz)", fontweight="bold")
    plt.tight_layout()
    out = _tag_path(f"emg_pca_diagnostics_{subj}_block{block}.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main():
    cfg = load_config()
    blocks = cfg.get("blocks", [1, 5])
    raw_dir = pipeline_dir / "outputs" / "derivatives" / "ica_cleaned_raw"
    epochs_dir = pipeline_dir / "outputs" / "derivatives" / "epochs_clean"

    subjects = discover_subjects(
        epochs_dir=epochs_dir,
        blocks=blocks,
        epoch_type="p3b",
        require_all_blocks=False,
    )
    if not subjects:
        print("No subjects found.")
        return

    csv_rows = []

    for subj in subjects:
        print(f"\n{'='*60}")
        print(f"  EMG Covariates (PCA): {subj}")
        print(f"{'='*60}")

        for block in blocks:
            epochs = _make_epochs(subj, block, raw_dir, epochs_dir)
            if epochs is None:
                continue

            sfreq = epochs.info["sfreq"]
            times = epochs.times
            
            # Check which raw EMG channels exist, excluding per-participant bads
            bad_emg = _load_bad_emg(subj)
            if bad_emg:
                print(f"    Excluding bad EMG channels: {bad_emg}")
            all_raw_chs = list(EMG_RAW.values())
            avail_raw = [ch for ch in all_raw_chs if ch in epochs.ch_names and ch not in bad_emg]
            if len(avail_raw) < 2:
                print(f"  Block {block}: Fewer than 2 EMG sensors found. Skipping.")
                continue
            
            # Extract raw data for all available EMG channels
            raw_data = epochs.copy().pick(avail_raw).get_data()  # (trials, ch, time)
            ch_idx = {ch: i for i, ch in enumerate(avail_raw)}
            
            # Build derived signals: bipolar for temporalis, monopolar for neck
            derived_labels = []
            derived_trials = []  # list of (n_trials, n_times) arrays
            
            for label, ch_a, ch_b in BIPOLAR_PAIRS:
                if ch_a in ch_idx and ch_b in ch_idx:
                    bipolar = raw_data[:, ch_idx[ch_a], :] - raw_data[:, ch_idx[ch_b], :]
                    derived_trials.append(bipolar)
                    derived_labels.append(label)
            
            for label, ch in MONOPOLAR:
                if ch in ch_idx:
                    derived_trials.append(raw_data[:, ch_idx[ch], :])
                    derived_labels.append(label)
            
            if len(derived_labels) < 2:
                print(f"  Block {block}: Fewer than 2 derived EMG signals. Skipping.")
                continue
                
            print(f"  Block {block}: Extracting 55-85Hz envelopes for {derived_labels}")
            
            # Stack into (n_trials, n_derived, n_times)
            derived_data = np.stack(derived_trials, axis=1)
            n_trials, n_derived, _ = derived_data.shape
            crop_mask = (times >= STIM_TMIN) & (times <= STIM_TMAX)
            
            per_chan_trial = np.full((n_trials, n_derived), np.nan)
            for d in range(n_derived):
                for i in range(n_trials):
                    try:
                        sig = derived_data[i, d, :].astype(np.float64)
                        filt = mne.filter.filter_data(
                            sig, sfreq,
                            l_freq=GAMMA_BAND[0], h_freq=GAMMA_BAND[1],
                            method="iir",
                            iir_params=dict(order=BUTTER_ORDER, ftype="butter"),
                            verbose=False)
                        env = np.abs(sp_hilbert(filt))
                        per_chan_trial[i, d] = float(np.mean(env[crop_mask])) * 1e6
                    except Exception:
                        pass
            
            # Impute NaNs gently if any single trial failed a channel
            df_chan = pd.DataFrame(per_chan_trial, columns=derived_labels)
            if df_chan.isna().all().all():
                print(f"  Block {block}: Failed to extract envelopes. Skipping.")
                continue
            df_chan = df_chan.fillna(df_chan.mean())
            
            # --- Perform PCA ---
            if len(derived_labels) >= 2:
                pca = PCA()
                pca_scores = pca.fit_transform(df_chan)
                pc1_scores = pca_scores[:, 0]
                var_explained = pca.explained_variance_ratio_[0] * 100
                
                print(f"    PCA calculated: PC1 explains {var_explained:.1f}% of variance across sensors")
                _plot_pca_diagnostics(subj, block, df_chan.values, pca, derived_labels, pc1_scores)
            else:
                print("    WARNING: Only 1 EMG signal. PCA bypassed; using raw envelope as PC1 proxy.")
                pc1_scores = df_chan.iloc[:, 0].values
                var_explained = 100.0
                
            # --- Store PC1 mapped back to each exact trial ---
            for i, trial_val in enumerate(pc1_scores):
                csv_rows.append({
                    "subject": subj,
                    "block": block,
                    "trial": i+1,
                    "emg_pc1": float(trial_val),
                    "pc1_var_explained": var_explained,
                    "n_emg_derived": len(derived_labels),
                    "n_emg_raw": len(avail_raw)
                })

    # Save CSV
    if csv_rows:
        df = pd.DataFrame(csv_rows)
        # We also create a block-level aggregation to slip neatly into merged_wide.csv
        df_block = df.groupby(["subject", "block"]).agg(
            emg_pc1_mean=("emg_pc1", "mean"),
            emg_pc1_std=("emg_pc1", "std"),
            emg_pc1_var_explained=("pc1_var_explained", "mean")
        ).reset_index()
        
        out = OUTPUT_DIR / "emg_covariates.csv"
        df.to_csv(out, index=False)
        print(f"\nSaved trial-level covariates to {out}")

        df_block_out = OUTPUT_DIR / "emg_covariates_block.csv"
        df_block.to_csv(df_block_out, index=False)
        print(f"Saved block-level summary to {df_block_out}")
        
    else:
        print("No robust EMG channels found across recordings.")

if __name__ == "__main__":
    main()
