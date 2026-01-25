# baseline_pipeline/steps/04_compare_pipelines.py
"""
Compare baseline vs advanced pipeline outputs.
Generates metrics for methods comparison paper.
"""
from pathlib import Path
import mne
import numpy as np
import pandas as pd

# Paths
BASELINE_ROOT = Path(__file__).resolve().parents[1]
ADVANCED_ROOT = Path(__file__).resolve().parents[2] / "eeg_pipeline"

BASELINE_EPOCHS = BASELINE_ROOT / "outputs" / "epochs"
ADVANCED_EPOCHS = ADVANCED_ROOT / "outputs" / "derivatives" / "epochs_clean"

BASELINE_PAC = BASELINE_ROOT / "outputs" / "pac"
ADVANCED_FEATURES = ADVANCED_ROOT / "outputs" / "features"

OUTPUT_DIR = BASELINE_ROOT / "outputs" / "comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def count_epochs(epochs_dir, pattern):
    """Count epochs across all files matching pattern."""
    files = list(epochs_dir.glob(pattern))
    total = 0
    for f in files:
        try:
            epochs = mne.read_epochs(f, preload=False, verbose=False)
            total += len(epochs)
        except:
            pass
    return total


def compute_gamma_snr(epochs_file):
    """Compute gamma-band (30-80 Hz) signal-to-noise ratio."""
    try:
        epochs = mne.read_epochs(epochs_file, preload=True, verbose=False)
        psd = epochs.compute_psd(fmin=30, fmax=80, verbose=False)
        power = psd.get_data().mean()  # Mean power in gamma band
        
        # SNR = signal power / noise floor (approximated by lowest power)
        noise = psd.get_data().min()
        snr = power / noise if noise > 0 else 0
        return snr
    except:
        return np.nan


def main():
    print("=" * 60)
    print("PIPELINE COMPARISON: Baseline vs Advanced")
    print("=" * 60)
    
    results = {}
    
    # 1. Epoch retention comparison
    print("\n--- Epoch Retention ---")
    
    baseline_stim = count_epochs(BASELINE_EPOCHS, "*_stim-epo.fif")
    baseline_resp = count_epochs(BASELINE_EPOCHS, "*_resp-epo.fif")
    
    advanced_p3b = count_epochs(ADVANCED_EPOCHS, "*_p3b_clean-epo.fif")
    advanced_pac = count_epochs(ADVANCED_EPOCHS, "*_pac_clean-epo.fif")
    
    print(f"Baseline - Stimulus epochs: {baseline_stim}")
    print(f"Advanced - P3b epochs: {advanced_p3b}")
    print(f"Baseline - Response epochs: {baseline_resp}")
    print(f"Advanced - PAC epochs: {advanced_pac}")
    
    results['baseline_stim_epochs'] = baseline_stim
    results['advanced_p3b_epochs'] = advanced_p3b
    results['baseline_resp_epochs'] = baseline_resp
    results['advanced_pac_epochs'] = advanced_pac
    
    # 2. Gamma SNR comparison
    print("\n--- Gamma-Band SNR ---")
    
    baseline_files = list(BASELINE_EPOCHS.glob("*_resp-epo.fif"))
    advanced_files = list(ADVANCED_EPOCHS.glob("*_pac_clean-epo.fif"))
    
    baseline_snr = np.nanmean([compute_gamma_snr(f) for f in baseline_files]) if baseline_files else np.nan
    advanced_snr = np.nanmean([compute_gamma_snr(f) for f in advanced_files]) if advanced_files else np.nan
    
    print(f"Baseline gamma SNR: {baseline_snr:.2f}")
    print(f"Advanced gamma SNR: {advanced_snr:.2f}")
    
    results['baseline_gamma_snr'] = baseline_snr
    results['advanced_gamma_snr'] = advanced_snr
    
    # 3. PAC values comparison (if available)
    print("\n--- PAC Values ---")
    
    baseline_pac_files = list(BASELINE_PAC.glob("*_pac_summary.csv"))
    if baseline_pac_files:
        df_base = pd.read_csv(baseline_pac_files[0])
        baseline_mean_pac = df_base['mean_pac'].mean()
        print(f"Baseline mean PAC (raw MI): {baseline_mean_pac:.6f}")
        results['baseline_mean_pac'] = baseline_mean_pac
    else:
        print("Baseline PAC not computed yet")
    
    # Save comparison results
    df_results = pd.DataFrame([results])
    df_results.to_csv(OUTPUT_DIR / "pipeline_comparison.csv", index=False)
    print(f"\nSaved comparison to {OUTPUT_DIR / 'pipeline_comparison.csv'}")
    
    print("\n" + "=" * 60)
    print("To demonstrate improvement, run both pipelines on the same")
    print("real dataset and compare these metrics with statistical tests.")
    print("=" * 60)


if __name__ == "__main__":
    main()
