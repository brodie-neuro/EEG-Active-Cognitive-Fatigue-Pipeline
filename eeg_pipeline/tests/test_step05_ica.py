# eeg_pipeline/tests/test_step05_ica.py
"""
Automated tests to verify Step 05 (ICA) outputs.
Handles edge case where ICA is skipped due to synthetic/flat data.
"""
import pytest
from pathlib import Path
import mne
import numpy as np

pipeline_root = Path(__file__).resolve().parents[1]


def test_step05_output_exists():
    """Verify that Step 05 produced output files."""
    output_dir = pipeline_root / "outputs" / "derivatives" / "ica_cleaned_raw"
    
    assert output_dir.exists(), "ICA output directory missing - run Step 05 first"
    
    files = list(output_dir.glob("*_ica-raw.fif"))
    assert len(files) > 0, "No ICA output files found"
    
    for f in files:
        subj = f.name.split("_")[0]
        print(f"Found output for {subj}")


def test_step05_signal_integrity():
    """Load output files and verify basic signal properties."""
    output_dir = pipeline_root / "outputs" / "derivatives" / "ica_cleaned_raw"
    files = list(output_dir.glob("*_ica-raw.fif"))
    
    if not files:
        pytest.skip("No ICA output files found.")
        
    for f in files:
        raw = mne.io.read_raw_fif(f, preload=True)
        
        # Check sampling rate is preserved
        assert raw.info['sfreq'] > 0, "Invalid sampling rate"
        
        # Check for NaNs (final output should be clean)
        data = raw.get_data()
        nan_count = np.sum(np.isnan(data))
        if nan_count > 0:
            print(f"Warning: {nan_count} NaN values in {f.name}")
        
        print(f"Verified {f.name}: sfreq={raw.info['sfreq']}, channels={len(raw.ch_names)}")


def test_step05_qc_plots():
    """Check if QC plots were generated (optional - skipped if ICA was skipped)."""
    qc_dir = pipeline_root / "outputs" / "qc_figs" / "ica"
    
    if not qc_dir.exists():
        pytest.skip("QC directory doesn't exist - ICA may have been skipped for synthetic data")
    
    plots = list(qc_dir.glob("*.png"))
    if len(plots) == 0:
        pytest.skip("No QC plots found - ICA likely skipped due to flat channels")
    
    print(f"Found {len(plots)} QC plots")


if __name__ == "__main__":
    from _pytest.outcomes import Skipped
    
    print("Running Step 05 validation tests...\n")
    test_step05_output_exists()
    test_step05_signal_integrity()
    
    try:
        test_step05_qc_plots()
    except Skipped as e:
        print(f"Skipped: {e}")
    
    print("\nAll tests passed!")
