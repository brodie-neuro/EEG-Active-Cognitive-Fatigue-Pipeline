# eeg_pipeline/analysis/10_erp_p3b.py
"""
Step 10: P3b ERP Analysis
Extracts P3b amplitude (300-500ms) and peak latency from target trials.
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

# P3b analysis parameters
P3B_CHANNELS = ['Pz', 'CPz']  # Primary and backup
P3B_TMIN = 0.300  # 300ms
P3B_TMAX = 0.500  # 500ms


def extract_p3b(epochs, ch_name='Pz'):
    """Extract P3b mean amplitude and peak latency."""
    if ch_name not in epochs.ch_names:
        # Try backup
        ch_name = 'CPz' if 'CPz' in epochs.ch_names else epochs.ch_names[0]
    
    # Get data for target channel
    data = epochs.copy().pick([ch_name]).get_data()  # (n_epochs, 1, n_times)
    times = epochs.times
    
    # P3b window mask
    t_mask = (times >= P3B_TMIN) & (times <= P3B_TMAX)
    
    # Grand average ERP
    erp = data.mean(axis=0).squeeze()  # Average across epochs
    
    # Mean amplitude in P3b window (in µV)
    p3b_amp = erp[t_mask].mean() * 1e6
    
    # Peak latency (in ms)
    window_erp = erp[t_mask]
    peak_idx = np.argmax(window_erp)
    p3b_lat = times[t_mask][peak_idx] * 1000
    
    return {
        'p3b_amplitude_uV': p3b_amp,
        'p3b_latency_ms': p3b_lat,
        'channel': ch_name
    }


def main():
    files = sorted(list(EPOCHS_DIR.glob("*_p3b_clean-epo.fif")))
    if not files:
        print(f"No epoch files found in {EPOCHS_DIR}")
        return
    
    results = []
    
    for f in files:
        subj = f.stem.split("_")[0]
        print(f"--- P3b Analysis: {subj} ---")
        
        epochs = mne.read_epochs(f, preload=True, verbose=False)
        
        # Check for EEG channels
        if len(mne.pick_types(epochs.info, eeg=True)) == 0:
            print(f"No EEG channels for {subj} - skipping")
            continue
        
        # Filter to target trials only if possible
        target_epochs = epochs['stim/target'] if 'stim/target' in epochs.event_id else epochs
        
        p3b = extract_p3b(target_epochs)
        p3b['subject'] = subj
        results.append(p3b)
        print(f"  P3b amplitude: {p3b['p3b_amplitude_uV']:.2f} µV, latency: {p3b['p3b_latency_ms']:.0f} ms")
    
    if results:
        df = pd.DataFrame(results)
        output_file = OUTPUT_DIR / "p3b_features.csv"
        df.to_csv(output_file, index=False)
        print(f"\nSaved P3b features to {output_file}")


if __name__ == "__main__":
    main()
