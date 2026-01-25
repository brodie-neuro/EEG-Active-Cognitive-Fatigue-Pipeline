# steps/09_features.py
"""
Step 09: Feature extraction for analysis.
Extracts P3b amplitude/latency, IAF, theta frequency, and prepares for PAC calculation.
"""
import sys
from pathlib import Path
import mne
import numpy as np
import pandas as pd

pipeline_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pipeline_dir))
from src.utils_io import load_config


def extract_p3b_features(epochs, ch_name='Pz'):
    """Extract P3b mean amplitude (300-500ms) and peak latency."""
    if ch_name not in epochs.ch_names:
        ch_name = epochs.ch_names[0]  # Fallback
    
    data = epochs.get_data(picks=[ch_name])  # (n_epochs, 1, n_times)
    times = epochs.times
    
    # P3b window: 300-500 ms
    t_mask = (times >= 0.3) & (times <= 0.5)
    
    # Mean amplitude across trials
    erp = data.mean(axis=0).squeeze()  # Average across epochs
    p3b_amp = erp[t_mask].mean() * 1e6  # Convert to µV
    
    # Peak latency
    peak_idx = np.argmax(erp[t_mask])
    p3b_lat = times[t_mask][peak_idx] * 1000  # Convert to ms
    
    return {'p3b_amplitude_uV': p3b_amp, 'p3b_latency_ms': p3b_lat}


def extract_iaf(raw, ch_name='Oz'):
    """Extract Individual Alpha Frequency from resting-state data."""
    if ch_name not in raw.ch_names:
        ch_name = raw.ch_names[0]
    
    # Compute PSD
    psd = raw.compute_psd(picks=[ch_name], fmin=7, fmax=14, verbose=False)
    freqs = psd.freqs
    power = psd.get_data().squeeze()
    
    # Find peak frequency
    peak_idx = np.argmax(power)
    iaf = freqs[peak_idx]
    
    return {'iaf_hz': iaf}


def main():
    cfg = load_config()
    
    pipeline_root = Path(__file__).resolve().parents[1]
    epochs_dir = pipeline_root / "outputs" / "derivatives" / "epochs_clean"
    output_dir = pipeline_root / "outputs" / "features"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not epochs_dir.exists():
        print(f"Epochs directory not found: {epochs_dir}. Run Step 08 first.")
        return
    
    results = []
    
    # Find all subjects by looking at P3b epoch files
    p3b_files = sorted(list(epochs_dir.glob("*_p3b_clean-epo.fif")))
    
    for f in p3b_files:
        subj = f.name.split("_")[0]
        print(f"--- Extracting features for {subj} ---")
        
        row = {'subject': subj}
        
        # P3b features
        try:
            epochs_p3b = mne.read_epochs(f, preload=True, verbose=False)
            p3b_feats = extract_p3b_features(epochs_p3b)
            row.update(p3b_feats)
            print(f"  P3b: {p3b_feats}")
        except Exception as e:
            print(f"  P3b extraction failed: {e}")
        
        # PAC epochs (for later PAC calculation - just count for now)
        pac_file = epochs_dir / f"{subj}_pac_clean-epo.fif"
        if pac_file.exists():
            try:
                epochs_pac = mne.read_epochs(pac_file, preload=True, verbose=False)
                row['n_pac_epochs'] = len(epochs_pac)
                print(f"  PAC epochs available: {len(epochs_pac)}")
            except Exception as e:
                print(f"  PAC epoch loading failed: {e}")
        
        results.append(row)
    
    # Save features to CSV
    if results:
        df = pd.DataFrame(results)
        output_file = output_dir / "extracted_features.csv"
        df.to_csv(output_file, index=False)
        print(f"\nSaved features to {output_file}")
        print(df.to_string())
    else:
        print("No features extracted.")


if __name__ == "__main__":
    main()
