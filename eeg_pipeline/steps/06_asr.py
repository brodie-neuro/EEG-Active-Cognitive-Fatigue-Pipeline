# steps/06_asr.py
"""
Step 06: Artifact Subspace Reconstruction (ASR) for transient artefact repair.
Cleans high-amplitude bursts (muscle twitches, head movements) without removing data.
"""
import sys
from pathlib import Path
import mne
import numpy as np

pipeline_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pipeline_dir))
from src.utils_io import load_config, save_clean_raw, subj_id_from_derivative

# ASR from meegkit
try:
    from meegkit.asr import ASR
    ASR_AVAILABLE = True
except ImportError:
    ASR_AVAILABLE = False
    print("Warning: meegkit.asr not available. Install with: pip install meegkit")


def main():
    cfg = load_config()
    
    pipeline_root = Path(__file__).resolve().parents[1]
    input_dir = pipeline_root / "outputs" / "derivatives" / "ica_cleaned_raw"
    output_dir = pipeline_root / "outputs" / "derivatives" / "asr_cleaned_raw"
    qc_dir = pipeline_root / "outputs" / "qc_figs" / "asr"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}. Run Step 05 first.")
        
    files = sorted(list(input_dir.glob("*_ica-raw.fif")))
    if not files:
        print("No files found to process.")
        return
    
    for f in files:
        subj = subj_id_from_derivative(f)
        print(f"--- Processing {subj} (ASR) ---")
        
        raw = mne.io.read_raw_fif(f, preload=True)
        
        if not ASR_AVAILABLE:
            print("ASR not available - copying input to output unchanged.")
            save_clean_raw(raw, output_dir, subj, "asr")
            continue
        
        # Get EEG data only
        eeg_picks = mne.pick_types(raw.info, eeg=True)
        
        if len(eeg_picks) == 0:
            print("No EEG channels found - copying input unchanged.")
            save_clean_raw(raw, output_dir, subj, "asr")
            continue
        
        data = raw.get_data(picks=eeg_picks)
        sfreq = raw.info['sfreq']
        
        # Check for flat/NaN channels (skip ASR if data is bad)
        variances = np.var(data, axis=1)
        if np.any(variances < 1e-15) or np.any(np.isnan(data)):
            print("Data has flat or NaN channels - skipping ASR.")
            save_clean_raw(raw, output_dir, subj, "asr")
            continue
        
        # Run ASR
        print("Fitting ASR on calibration data...")
        asr = ASR(sfreq=sfreq, cutoff=20)  # cutoff=20 is standard, lower = more aggressive
        
        # Use first 60 seconds as calibration (should be relatively clean)
        calib_samples = min(int(60 * sfreq), data.shape[1] // 2)
        asr.fit(data[:, :calib_samples])
        
        print("Transforming full recording...")
        data_clean = asr.transform(data)
        
        # Put cleaned data back
        raw_clean = raw.copy()
        raw_clean._data[eeg_picks, :] = data_clean
        
        # Save QC plot: variance before/after
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 1, figsize=(12, 6))
            
            # Time series of one frontal channel
            ch_idx = raw.ch_names.index('Fz') if 'Fz' in raw.ch_names else 0
            t = np.arange(data.shape[1]) / sfreq
            
            axes[0].plot(t[:int(10*sfreq)], data[ch_idx, :int(10*sfreq)] * 1e6, 'b', alpha=0.7, label='Before')
            axes[0].plot(t[:int(10*sfreq)], data_clean[ch_idx, :int(10*sfreq)] * 1e6, 'r', alpha=0.7, label='After')
            axes[0].set_ylabel('µV')
            axes[0].set_title(f'{subj} - First 10 seconds (Fz)')
            axes[0].legend()
            
            # Global variance comparison
            var_before = np.var(data, axis=1) * 1e12
            var_after = np.var(data_clean, axis=1) * 1e12
            axes[1].bar(range(len(var_before)), var_before, alpha=0.5, label='Before')
            axes[1].bar(range(len(var_after)), var_after, alpha=0.5, label='After')
            axes[1].set_ylabel('Variance (µV²)')
            axes[1].set_xlabel('Channel')
            axes[1].set_title('Channel variance before/after ASR')
            axes[1].legend()
            
            plt.tight_layout()
            fig.savefig(qc_dir / f"{subj}_asr_comparison.png", dpi=100)
            plt.close(fig)
            print(f"Saved QC plot to {qc_dir}")
        except Exception as e:
            print(f"Could not generate QC plot: {e}")
        
        save_clean_raw(raw_clean, output_dir, subj, "asr")
        print(f"Saved {subj} ASR cleaned data.\n")


if __name__ == "__main__":
    main()
