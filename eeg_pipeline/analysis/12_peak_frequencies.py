# eeg_pipeline/analysis/12_peak_frequencies.py
"""
Step 12: Oscillatory Frequency Analysis
Extracts IAF, theta frequency, alpha frequency, beta frequency.
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
RAW_DIR = pipeline_dir / "outputs" / "derivatives" / "asr_cleaned_raw"
OUTPUT_DIR = pipeline_dir / "outputs" / "features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_peak_frequency(psd_data, freqs, fmin, fmax):
    """Find peak frequency in specified band."""
    mask = (freqs >= fmin) & (freqs <= fmax)
    band_power = psd_data[mask]
    band_freqs = freqs[mask]
    
    if len(band_power) == 0 or np.all(np.isnan(band_power)):
        return np.nan
    
    peak_idx = np.argmax(band_power)
    return band_freqs[peak_idx]


def compute_instantaneous_frequency(signal, sfreq, fmin, fmax):
    """Compute instantaneous frequency using Hilbert transform."""
    # Bandpass filter
    filtered = mne.filter.filter_data(signal, sfreq, fmin, fmax, verbose=False)
    
    # Hilbert transform
    analytic = hilbert(filtered)
    phase = np.unwrap(np.angle(analytic))
    
    # Instantaneous frequency
    inst_freq = np.diff(phase) * sfreq / (2 * np.pi)
    
    # Return median (robust to outliers)
    return np.median(inst_freq[(inst_freq > fmin) & (inst_freq < fmax)])


def extract_iaf(raw, ch_name='Oz'):
    """Extract Individual Alpha Frequency from continuous data."""
    if ch_name not in raw.ch_names:
        ch_name = raw.ch_names[0]
    
    psd = raw.compute_psd(picks=[ch_name], fmin=7, fmax=14, verbose=False)
    power = psd.get_data().squeeze()
    freqs = psd.freqs
    
    return find_peak_frequency(power, freqs, 7, 14)


def extract_task_frequencies(epochs, iaf, ch_names=['Fz', 'FCz']):
    """
    Extract task-related oscillatory frequencies using INDIVIDUAL bands.
    Individual theta = IAF - 4 Hz (±2 Hz)
    Individual alpha = IAF (±2 Hz)
    """
    if np.isnan(iaf):
        iaf = 10.0  # Default if IAF not available
    
    # Individual frequency bands based on IAF
    i_theta = (iaf - 6, iaf - 2)  # e.g., if IAF=10, theta = 4-8 Hz
    i_alpha = (iaf - 2, iaf + 2)  # e.g., if IAF=10, alpha = 8-12 Hz
    
    available = [ch for ch in ch_names if ch in epochs.ch_names]
    if not available:
        available = [epochs.ch_names[0]]
    
    data = epochs.copy().pick(available).get_data()  # (n_epochs, n_ch, n_times)
    sfreq = epochs.info['sfreq']
    
    # Average across channels
    mean_signal = data.mean(axis=1)  # (n_epochs, n_times)
    
    # Compute instantaneous frequency for each epoch
    theta_freqs = []
    alpha_freqs = []
    
    for ep in range(mean_signal.shape[0]):
        try:
            theta_freqs.append(compute_instantaneous_frequency(
                mean_signal[ep], sfreq, max(1, i_theta[0]), i_theta[1]))
            alpha_freqs.append(compute_instantaneous_frequency(
                mean_signal[ep], sfreq, i_alpha[0], i_alpha[1]))
        except:
            pass
    
    return {
        'f_theta': np.nanmedian(theta_freqs) if theta_freqs else np.nan,
        'f_alpha': np.nanmedian(alpha_freqs) if alpha_freqs else np.nan,
        'i_theta_band': f"{i_theta[0]:.1f}-{i_theta[1]:.1f}",
        'i_alpha_band': f"{i_alpha[0]:.1f}-{i_alpha[1]:.1f}"
    }


def main():
    epoch_files = sorted(list(EPOCHS_DIR.glob("*_pac_clean-epo.fif")))
    raw_files = sorted(list(RAW_DIR.glob("*_asr-raw.fif")))
    
    if not epoch_files:
        print(f"No epoch files found in {EPOCHS_DIR}")
        return
    
    results = []
    
    for f in epoch_files:
        subj = f.stem.split("_")[0]
        print(f"--- Frequency Analysis: {subj} ---")
        
        row = {'subject': subj}
        
        # IAF from continuous data
        raw_file = RAW_DIR / f"{subj}_asr-raw.fif"
        if raw_file.exists():
            try:
                raw = mne.io.read_raw_fif(raw_file, preload=True, verbose=False)
                iaf = extract_iaf(raw)
                row['iaf'] = iaf
                print(f"  IAF: {iaf:.2f} Hz")
            except Exception as e:
                print(f"  IAF extraction failed: {e}")
                row['iaf'] = np.nan
        else:
            row['iaf'] = np.nan
        
        # Task-related frequencies from epochs
        epochs = mne.read_epochs(f, preload=True, verbose=False)
        
        if len(mne.pick_types(epochs.info, eeg=True)) == 0:
            print(f"No EEG channels for {subj} - skipping task frequencies")
            row['f_theta'] = np.nan
            row['f_alpha'] = np.nan
            row['i_theta_band'] = np.nan
            row['i_alpha_band'] = np.nan
        else:
            iaf_val = row.get('iaf', np.nan)
            freqs = extract_task_frequencies(epochs, iaf_val)
            row.update(freqs)
            if not np.isnan(freqs['f_theta']):
                print(f"  fθ: {freqs['f_theta']:.2f} Hz (band: {freqs['i_theta_band']})")
                print(f"  fα: {freqs['f_alpha']:.2f} Hz (band: {freqs['i_alpha_band']})")
        
        results.append(row)
    
    if results:
        df = pd.DataFrame(results)
        output_file = OUTPUT_DIR / "frequency_features.csv"
        df.to_csv(output_file, index=False)
        print(f"\nSaved frequency features to {output_file}")


if __name__ == "__main__":
    main()
