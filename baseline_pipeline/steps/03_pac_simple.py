# baseline_pipeline/steps/03_pac_simple.py
"""
Baseline PAC analysis: Simple electrode-level Modulation Index.
No surrogate normalization, no node aggregation, no trimmed means.
"""
from pathlib import Path
import mne
import numpy as np
from scipy.signal import hilbert

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PIPELINE_ROOT / "outputs" / "epochs"
OUTPUT_DIR = PIPELINE_ROOT / "outputs" / "pac"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def modulation_index(phase_signal, amplitude_signal, n_bins=18):
    """
    Compute Modulation Index (Tort et al., 2010).
    Simple electrode-level PAC without surrogate normalization.
    """
    # Bin amplitude by phase
    phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    mean_amp = np.zeros(n_bins)
    
    for i in range(n_bins):
        mask = (phase_signal >= phase_bins[i]) & (phase_signal < phase_bins[i + 1])
        if mask.sum() > 0:
            mean_amp[i] = amplitude_signal[mask].mean()
    
    # Normalize
    mean_amp = mean_amp / mean_amp.sum() if mean_amp.sum() > 0 else mean_amp
    
    # Kullback-Leibler divergence from uniform
    uniform = np.ones(n_bins) / n_bins
    # Avoid log(0)
    mean_amp = np.clip(mean_amp, 1e-10, None)
    kl_div = np.sum(mean_amp * np.log(mean_amp / uniform))
    
    # Modulation index
    mi = kl_div / np.log(n_bins)
    return mi


def compute_pac_electrode(epochs, theta_band=(4, 8), gamma_band=(30, 80)):
    """Compute PAC for each electrode separately."""
    sfreq = epochs.info['sfreq']
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    n_epochs, n_channels, n_times = data.shape
    
    pac_values = np.zeros((n_epochs, n_channels))
    
    for ep in range(n_epochs):
        for ch in range(n_channels):
            signal = data[ep, ch, :]
            
            # Filter for theta phase
            theta = mne.filter.filter_data(signal, sfreq, theta_band[0], theta_band[1], verbose=False)
            theta_phase = np.angle(hilbert(theta))
            
            # Filter for gamma amplitude
            gamma = mne.filter.filter_data(signal, sfreq, gamma_band[0], gamma_band[1], verbose=False)
            gamma_amp = np.abs(hilbert(gamma))
            
            # Compute MI
            pac_values[ep, ch] = modulation_index(theta_phase, gamma_amp)
    
    return pac_values


def main():
    files = sorted(list(INPUT_DIR.glob("*_resp-epo.fif")))
    if not files:
        print(f"No epoch files found in {INPUT_DIR}")
        return
    
    for f in files:
        subj = f.stem.split("_")[0]
        print(f"--- Baseline PAC analysis: {subj} ---")
        
        epochs = mne.read_epochs(f, preload=True, verbose=False)
        
        # Check for EEG channels
        eeg_picks = mne.pick_types(epochs.info, eeg=True)
        if len(eeg_picks) == 0:
            print("No EEG channels found")
            continue
        
        epochs_eeg = epochs.copy().pick_types(eeg=True)
        
        print(f"Computing PAC for {len(epochs_eeg)} epochs, {len(epochs_eeg.ch_names)} channels...")
        pac = compute_pac_electrode(epochs_eeg)
        
        # Save raw PAC values (no normalization)
        np.save(OUTPUT_DIR / f"{subj}_pac_raw.npy", pac)
        
        # Summary: mean PAC per channel
        mean_pac = pac.mean(axis=0)
        print(f"Mean PAC across epochs: {mean_pac.mean():.4f} (range: {mean_pac.min():.4f} - {mean_pac.max():.4f})")
        
        # Save summary
        import pandas as pd
        df = pd.DataFrame({
            'channel': epochs_eeg.ch_names,
            'mean_pac': mean_pac
        })
        df.to_csv(OUTPUT_DIR / f"{subj}_pac_summary.csv", index=False)
        print(f"Saved PAC results to {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()
