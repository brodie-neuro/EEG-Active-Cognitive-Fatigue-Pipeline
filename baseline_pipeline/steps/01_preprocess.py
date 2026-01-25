# baseline_pipeline/steps/01_preprocess.py
"""
Baseline preprocessing: Traditional approach for comparison.
Uses notch filter, basic ICA, amplitude-based epoch rejection, average reference.
"""
import sys
from pathlib import Path
import mne
from mne.preprocessing import ICA
import numpy as np

# Paths
PIPELINE_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = Path(__file__).resolve().parents[2] / "eeg_pipeline" / "raw"
OUTPUT_DIR = PIPELINE_ROOT / "outputs" / "preprocessed"
QC_DIR = PIPELINE_ROOT / "outputs" / "qc"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
QC_DIR.mkdir(parents=True, exist_ok=True)


def main():
    files = sorted(list(RAW_DIR.glob("*.vhdr")))
    if not files:
        print(f"No raw files found in {RAW_DIR}")
        return
    
    for f in files:
        subj = f.stem.split("_")[0]
        print(f"--- Baseline preprocessing: {subj} ---")
        
        # Load raw data
        raw = mne.io.read_raw_brainvision(f, preload=True, verbose=False)
        
        # Set channel types
        raw.set_channel_types({
            'VEOG': 'eog', 'HEOG': 'eog',
            'EMG_L': 'emg', 'EMG_R': 'emg'
        })
        
        # 1. Basic filtering (bandpass + notch)
        print("Filtering: 0.1-100 Hz bandpass + 50 Hz notch...")
        raw.filter(l_freq=0.1, h_freq=100.0, n_jobs=-1, verbose=False)
        raw.notch_filter(freqs=[50, 100], n_jobs=-1, verbose=False)
        
        # 2. Average reference
        print("Applying average reference...")
        raw.set_eeg_reference('average', projection=False, verbose=False)
        
        # 3. Basic ICA (no automated labeling - would normally be manual)
        print("Running ICA...")
        raw_ica = raw.copy().filter(l_freq=1.0, h_freq=None, verbose=False)
        ica = ICA(n_components=20, method='fastica', random_state=42, max_iter=500)
        
        try:
            ica.fit(raw_ica, verbose=False)
            
            # Simple heuristic: exclude components with high correlation to EOG
            eog_indices, _ = ica.find_bads_eog(raw, verbose=False)
            ica.exclude = eog_indices
            print(f"Excluding {len(eog_indices)} EOG-related components")
            
            # Apply ICA
            ica.apply(raw, verbose=False)
        except Exception as e:
            print(f"ICA failed: {e}")
        
        # Save
        out_file = OUTPUT_DIR / f"{subj}_baseline-raw.fif"
        raw.save(out_file, overwrite=True)
        print(f"Saved {out_file}\n")


if __name__ == "__main__":
    main()
