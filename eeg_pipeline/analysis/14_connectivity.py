# eeg_pipeline/analysis/14_connectivity.py
"""
Step 14: Frontal-Parietal Connectivity
Computes PLV at theta and cross-frequency PAC between frontal and parietal regions.
"""
import sys
from pathlib import Path
import mne
import numpy as np
import pandas as pd
from scipy.signal import hilbert

pipeline_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pipeline_dir))

EPOCHS_DIR = pipeline_dir / "outputs" / "derivatives" / "epochs_clean"
OUTPUT_DIR = pipeline_dir / "outputs" / "features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Region pairs for connectivity
REGION_PAIRS = {
    'CF_CP': (['Fz', 'FCz'], ['Pz', 'POz']),
    'LF_LP': (['F7', 'F3'], ['P7', 'P3']),
    'RF_RP': (['F8', 'F4'], ['P8', 'P4'])
}

THETA_BAND = (4, 7)
GAMMA_BAND = (55, 85)


def compute_plv(signal1, signal2, sfreq, fmin, fmax):
    """Compute Phase Locking Value between two signals."""
    # Filter both signals
    filt1 = mne.filter.filter_data(signal1, sfreq, fmin, fmax, verbose=False)
    filt2 = mne.filter.filter_data(signal2, sfreq, fmin, fmax, verbose=False)
    
    # Get instantaneous phase
    phase1 = np.angle(hilbert(filt1))
    phase2 = np.angle(hilbert(filt2))
    
    # PLV = |mean(exp(i * phase_diff))|
    phase_diff = phase1 - phase2
    plv = np.abs(np.mean(np.exp(1j * phase_diff)))
    
    return plv


def compute_cross_freq_pac(frontal_signal, parietal_signal, sfreq):
    """Compute frontal theta phase → parietal gamma amplitude coupling."""
    # Frontal theta phase
    theta_filt = mne.filter.filter_data(frontal_signal, sfreq, THETA_BAND[0], THETA_BAND[1], verbose=False)
    theta_phase = np.angle(hilbert(theta_filt))
    
    # Parietal gamma amplitude
    gamma_filt = mne.filter.filter_data(parietal_signal, sfreq, GAMMA_BAND[0], GAMMA_BAND[1], verbose=False)
    gamma_amp = np.abs(hilbert(gamma_filt))
    
    # Modulation Index
    n_bins = 18
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


def get_region_signal(epochs, ch_names):
    """Get mean signal across region channels."""
    available = [ch for ch in ch_names if ch in epochs.ch_names]
    if not available:
        return None
    data = epochs.copy().pick(available).get_data()
    return data.mean(axis=1)  # Mean across channels: (n_epochs, n_times)


def main():
    files = sorted(list(EPOCHS_DIR.glob("*_pac_clean-epo.fif")))
    if not files:
        print(f"No epoch files found in {EPOCHS_DIR}")
        return
    
    results = []
    
    for f in files:
        subj = f.stem.split("_")[0]
        print(f"--- Connectivity Analysis: {subj} ---")
        
        epochs = mne.read_epochs(f, preload=True, verbose=False)
        sfreq = epochs.info['sfreq']
        
        if len(mne.pick_types(epochs.info, eeg=True)) == 0:
            print(f"No EEG channels for {subj} - skipping")
            continue
        
        row = {'subject': subj}
        
        for pair_name, (frontal_chs, parietal_chs) in REGION_PAIRS.items():
            frontal_data = get_region_signal(epochs, frontal_chs)
            parietal_data = get_region_signal(epochs, parietal_chs)
            
            if frontal_data is None or parietal_data is None:
                row[f'plv_theta_{pair_name}'] = np.nan
                row[f'pac_cross_{pair_name}'] = np.nan
                continue
            
            # Compute for each epoch, then average
            plvs = []
            pacs = []
            for ep in range(min(frontal_data.shape[0], 20)):
                try:
                    plvs.append(compute_plv(frontal_data[ep], parietal_data[ep], sfreq, *THETA_BAND))
                    pacs.append(compute_cross_freq_pac(frontal_data[ep], parietal_data[ep], sfreq))
                except:
                    pass
            
            row[f'plv_theta_{pair_name}'] = np.mean(plvs) if plvs else np.nan
            row[f'pac_cross_{pair_name}'] = np.mean(pacs) if pacs else np.nan
            print(f"  {pair_name}: PLV={row[f'plv_theta_{pair_name}']:.3f}, Cross-PAC={row[f'pac_cross_{pair_name}']:.4f}")
        
        results.append(row)
    
    if results:
        df = pd.DataFrame(results)
        output_file = OUTPUT_DIR / "connectivity_features.csv"
        df.to_csv(output_file, index=False)
        print(f"\nSaved connectivity features to {output_file}")


if __name__ == "__main__":
    main()
