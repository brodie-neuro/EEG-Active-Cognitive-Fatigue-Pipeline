# eeg_pipeline/analysis/13_pac_nodal.py
"""
Step 13: Nodal PAC with Z-Score Normalization
Computes theta-gamma PAC per electrode, normalizes with surrogates, aggregates to nodes.
"""
import sys
from pathlib import Path
import mne
import numpy as np
import pandas as pd
from scipy.signal import hilbert
from scipy.stats import trim_mean

pipeline_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pipeline_dir))

EPOCHS_DIR = pipeline_dir / "outputs" / "derivatives" / "epochs_clean"
OUTPUT_DIR = pipeline_dir / "outputs" / "features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# PAC parameters
THETA_BAND = (4, 8)
GAMMA_BAND = (55, 85)  # High gamma, avoids 50Hz artifact
N_SURROGATES = 200
TRIM_PROPORTION = 0.1  # 10% trim

# Node definitions
NODES = {
    'LF': ['F7', 'F3', 'FC5'],
    'CF': ['Fz', 'FCz', 'Cz'],
    'RF': ['F8', 'F4', 'FC6'],
    'LC': ['T7', 'C3', 'CP5'],
    'C': ['Cz', 'CPz'],
    'RC': ['T8', 'C4', 'CP6'],
    'LP': ['P7', 'P3', 'PO7'],
    'CP': ['Pz', 'POz'],
    'RP': ['P8', 'P4', 'PO8']
}


def modulation_index(theta_phase, gamma_amp, n_bins=18):
    """Compute Modulation Index (Tort et al., 2010)."""
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


def compute_pac_zscore(signal, sfreq, n_surrogates=200):
    """Compute surrogate-normalized PAC for a single channel."""
    # Filter for theta phase
    theta = mne.filter.filter_data(signal, sfreq, THETA_BAND[0], THETA_BAND[1], verbose=False)
    theta_phase = np.angle(hilbert(theta))
    
    # Filter for gamma amplitude
    gamma = mne.filter.filter_data(signal, sfreq, GAMMA_BAND[0], GAMMA_BAND[1], verbose=False)
    gamma_amp = np.abs(hilbert(gamma))
    
    # Real PAC
    mi_real = modulation_index(theta_phase, gamma_amp)
    
    # Surrogate distribution (time-shift method)
    surr_mis = []
    shift_range = len(signal) // 4
    for _ in range(n_surrogates):
        shift = np.random.randint(shift_range, len(signal) - shift_range)
        shifted_amp = np.roll(gamma_amp, shift)
        surr_mis.append(modulation_index(theta_phase, shifted_amp))
    
    # Z-score
    mean_surr = np.mean(surr_mis)
    std_surr = np.std(surr_mis)
    z_pac = (mi_real - mean_surr) / std_surr if std_surr > 0 else 0
    
    return z_pac


def compute_nodal_pac(epochs, node_channels):
    """Compute PAC for a node using trimmed mean of electrode Z-scores."""
    available = [ch for ch in node_channels if ch in epochs.ch_names]
    if not available:
        return np.nan
    
    data = epochs.copy().pick(available).get_data()  # (n_epochs, n_ch, n_times)
    sfreq = epochs.info['sfreq']
    
    # Compute Z-scored PAC for each electrode (averaged across epochs)
    electrode_pacs = []
    for ch_idx in range(data.shape[1]):
        epoch_pacs = []
        for ep_idx in range(min(data.shape[0], 20)):  # Limit epochs for speed
            try:
                z = compute_pac_zscore(data[ep_idx, ch_idx, :], sfreq, n_surrogates=50)
                epoch_pacs.append(z)
            except:
                pass
        if epoch_pacs:
            electrode_pacs.append(np.mean(epoch_pacs))
    
    if not electrode_pacs:
        return np.nan
    
    # 10% trimmed mean across electrodes
    return trim_mean(electrode_pacs, proportiontocut=TRIM_PROPORTION)


def main():
    files = sorted(list(EPOCHS_DIR.glob("*_pac_clean-epo.fif")))
    if not files:
        print(f"No epoch files found in {EPOCHS_DIR}")
        return
    
    results = []
    
    for f in files:
        subj = f.stem.split("_")[0]
        print(f"--- Nodal PAC Analysis: {subj} ---")
        
        epochs = mne.read_epochs(f, preload=True, verbose=False)
        
        if len(mne.pick_types(epochs.info, eeg=True)) == 0:
            print(f"No EEG channels for {subj} - skipping")
            continue
        
        row = {'subject': subj}
        
        for node_name, node_chs in NODES.items():
            pac = compute_nodal_pac(epochs, node_chs)
            row[f'pac_{node_name}'] = pac
            print(f"  {node_name}: Z={pac:.2f}" if not np.isnan(pac) else f"  {node_name}: N/A")
        
        results.append(row)
    
    if results:
        df = pd.DataFrame(results)
        output_file = OUTPUT_DIR / "pac_nodal_features.csv"
        df.to_csv(output_file, index=False)
        print(f"\nSaved nodal PAC features to {output_file}")


if __name__ == "__main__":
    main()
