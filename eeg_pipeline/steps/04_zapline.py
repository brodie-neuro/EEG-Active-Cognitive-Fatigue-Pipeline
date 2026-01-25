# steps/04_zapline.py
"""
Step 04: Line noise removal using Zapline-plus spatial filtering.
Falls back to notch filter if Zapline fails (e.g., on rank-deficient synthetic data).
"""
import sys
from pathlib import Path
import mne
import numpy as np
from meegkit.dss import dss_line

pipeline_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pipeline_dir))
from src.utils_io import load_config, save_clean_raw


def main():
    cfg = load_config()
    
    pipeline_root = Path(__file__).resolve().parents[1]
    input_dir = pipeline_root / "outputs" / "derivatives" / "referenced_raw"
    output_dir = pipeline_root / "outputs" / "derivatives" / "zapline_raw"
    qc_dir = pipeline_root / "outputs" / "qc_figs" / "zapline"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
    files = sorted(list(input_dir.glob("*_referenced-raw.fif")))
    if not files:
        print("No files found to process.")
        return
    
    fline = 50.0  # Line frequency (Hz)
    nremove = 1   # Number of components to remove
    
    for f in files:
        subj = f.name.split("_")[0]
        print(f"--- Processing {subj} (Zapline) ---")
        
        raw = mne.io.read_raw_fif(f, preload=True)
        sfreq = raw.info["sfreq"]
        
        # Get EEG data in microvolts for numerical stability
        data = raw.get_data(picks="eeg") * 1e6
        
        print("Running Zapline (removing 50Hz)...")
        try:
            # Replace any NaNs before processing
            if np.isnan(data).any():
                print("Warning: NaNs found, replacing with zeros...")
                data[np.isnan(data)] = 0
                
            out, _ = dss_line(data.T, fline=fline, sfreq=sfreq, nremove=nremove)
            
            # Remove 100Hz harmonic if sampling rate allows
            if sfreq > 200:
                print("Removing 100Hz harmonic...")
                out, _ = dss_line(out, fline=100.0, sfreq=sfreq, nremove=nremove)
            
            # Put cleaned data back into Raw object
            raw_zap = raw.copy()
            eeg_picks = mne.pick_types(raw.info, eeg=True)
            raw_zap._data[eeg_picks, :] = out.T / 1e6  # Convert back to Volts

        except Exception as e:
            print(f"Zapline failed: {e}")
            print("Falling back to MNE notch filter...")
            raw_zap = raw.copy().notch_filter(freqs=[50, 100], method='fir', n_jobs=-1, verbose=False)
        
        # Save QC plot comparing before/after
        try:
            fig = raw.compute_psd(fmin=1, fmax=120).plot(show=False)
            fig.axes[0].set_title(f"{subj} - Before (blue) vs After (red)")
            raw_zap.compute_psd(fmin=1, fmax=120).plot(axes=fig.axes[0], color='r', show=False)
            fig.savefig(qc_dir / f"{subj}_zapline_psd.png")
            print(f"Saved QC plot to {qc_dir}")
        except Exception as e:
            print(f"Could not generate QC plot: {e}")
        
        save_clean_raw(raw_zap, output_dir, subj, "zapline")
        print(f"Saved {subj} zapline data.\n")


if __name__ == "__main__":
    main()
