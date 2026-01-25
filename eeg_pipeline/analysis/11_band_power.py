# eeg_pipeline/analysis/11_band_power.py
"""
Step 11: Band Power Analysis
Extracts theta, alpha, beta, gamma power from task epochs.
"""
import sys
from pathlib import Path
import mne
import numpy as np
import pandas as pd

pipeline_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pipeline_dir))

EPOCHS_DIR = pipeline_dir / "outputs" / "derivatives" / "epochs_clean"
OUTPUT_DIR = pipeline_dir / "outputs" / "features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Band definitions (per postprocessing_analysis.md)
BANDS = {
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma_low': (35, 48),   # Low gamma - attention, local processing (below 50Hz line noise)
    'gamma_high': (55, 85)   # High gamma - WM, cognition (above 50Hz line noise)
}

# Region-specific channels for each band
BAND_CHANNELS = {
    'theta': ['Fz', 'FCz', 'Cz'],          # Frontal theta
    'alpha': ['Oz', 'POz', 'O1', 'O2'],     # Posterior alpha
    'beta': ['Cz', 'C3', 'C4'],             # Central beta
    'gamma_low': None,                      # All EEG channels
    'gamma_high': None                      # All EEG channels
}


def compute_band_power(epochs, band_name):
    """Compute mean power in specified frequency band."""
    fmin, fmax = BANDS[band_name]
    
    # Select channels for this band
    ch_picks = BAND_CHANNELS.get(band_name)
    if ch_picks:
        available = [ch for ch in ch_picks if ch in epochs.ch_names]
        if not available:
            available = None
    else:
        available = None
    
    # Compute PSD
    try:
        psd = epochs.compute_psd(fmin=fmin, fmax=fmax, picks=available, verbose=False)
        power = psd.get_data().mean()  # Mean across epochs, channels, frequencies
        return power * 1e12  # Convert to µV²
    except:
        return np.nan


def main():
    files = sorted(list(EPOCHS_DIR.glob("*_pac_clean-epo.fif")))
    if not files:
        print(f"No epoch files found in {EPOCHS_DIR}")
        return
    
    results = []
    
    for f in files:
        subj = f.stem.split("_")[0]
        print(f"--- Band Power Analysis: {subj} ---")
        
        epochs = mne.read_epochs(f, preload=True, verbose=False)
        
        if len(mne.pick_types(epochs.info, eeg=True)) == 0:
            print(f"No EEG channels for {subj} - skipping")
            continue
        
        row = {'subject': subj}
        
        for band_name in BANDS:
            power = compute_band_power(epochs, band_name)
            row[f'{band_name}_power'] = power
            print(f"  {band_name}: {power:.4f} µV²")
        
        results.append(row)
    
    if results:
        df = pd.DataFrame(results)
        output_file = OUTPUT_DIR / "band_power_features.csv"
        df.to_csv(output_file, index=False)
        print(f"\nSaved band power features to {output_file}")


if __name__ == "__main__":
    main()
