# steps/04_zapline.py
"""
Step 04: Line noise removal using Zapline-plus spatial filtering.
Falls back to notch filter if Zapline fails (e.g., on rank-deficient synthetic data).
"""
import sys
from pathlib import Path
import mne
import numpy as np
try:
    from meegkit.dss import dss_line
    MEEGKIT_AVAILABLE = True
except ImportError:
    dss_line = None
    MEEGKIT_AVAILABLE = False

pipeline_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pipeline_dir))
from src.utils_io import load_config, save_clean_raw, subj_id_from_derivative
from src.utils_config import get_param
from src.utils_report import QCReport, qc_psd_overlay
from src.utils_logging import setup_pipeline_logger


def main():
    cfg = load_config()
    logger = setup_pipeline_logger('04_zapline')
    
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
    
    # Read parameters from config
    zap_params = get_param('zapline')
    filt_params = get_param('filtering')
    fline = filt_params.get('notch_freq', 50.0)
    n_harmonics = zap_params.get('n_harmonics', 4)
    nremove = 1   # DSS components to remove per harmonic
    
    for f in files:
        subj = subj_id_from_derivative(f)
        print(f"--- Processing {subj} (Zapline) ---")
        
        raw = mne.io.read_raw_fif(f, preload=True)
        sfreq = raw.info["sfreq"]
        
        # Extract block number from filename for QC report
        block_str = ''.join(c for c in f.stem if c.isdigit())
        block_num = int(block_str[-1]) if block_str else 1
        qc = QCReport(subj, block_num)
        
        # Get EEG data in microvolts for numerical stability
        data = raw.get_data(picks="eeg") * 1e6
        
        print("Running Zapline (removing 50Hz)...")
        try:
            if not MEEGKIT_AVAILABLE:
                raise ImportError("meegkit not installed")

            # Replace any NaNs before processing
            if np.isnan(data).any():
                print("Warning: NaNs found, replacing with zeros...")
                data[np.isnan(data)] = 0
                
            out, _ = dss_line(data.T, fline=fline, sfreq=sfreq, nremove=nremove)
            
            # Remove harmonics as configured
            harmonics_removed = [fline]
            for h in range(2, n_harmonics + 1):
                harmonic_freq = fline * h
                if harmonic_freq < sfreq / 2:  # Below Nyquist
                    print(f"Removing {harmonic_freq:.0f}Hz harmonic...")
                    out, _ = dss_line(out, fline=harmonic_freq, sfreq=sfreq, nremove=nremove)
                    harmonics_removed.append(harmonic_freq)
            
            # Put cleaned data back into Raw object
            raw_zap = raw.copy()
            eeg_picks = mne.pick_types(raw.info, eeg=True)
            raw_zap._data[eeg_picks, :] = out.T / 1e6  # Convert back to Volts

        except Exception as e:
            print(f"Zapline failed: {e}")
            print("Falling back to MNE notch filter...")
            notch_freqs = [fline * h for h in range(1, n_harmonics + 1) if fline * h < sfreq / 2]
            raw_zap = raw.copy().notch_filter(freqs=notch_freqs, method='fir', n_jobs=-1, verbose=False)
        
        # Save QC plot comparing before/after
        try:
            fig = qc_psd_overlay(raw, raw_zap, f"{subj} - ZapLine PSD (Before/After)")
            qc.add_figure('04_zapline_psd', fig)
        except Exception as e:
            print(f"Could not generate QC plot: {e}")
        
        # Log QC step
        qc.log_step('04_zapline', status='PASS',
                     metrics={'line_freq': fline, 'n_harmonics': n_harmonics},
                     params_used=zap_params)
        
        save_clean_raw(raw_zap, output_dir, subj, "zapline")
        qc.save_report()
        print(f"Saved {subj} zapline data.\n")


if __name__ == "__main__":
    main()
