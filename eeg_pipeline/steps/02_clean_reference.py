# steps/02_clean_reference.py
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy
print(f"!!! CURRENT NUMPY VERSION: {numpy.__version__} !!!")

# --- FIX: Point to 'eeg_pipeline' (parents[1]) ---
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils_io import load_config, read_raw, save_clean_raw
from src.utils_clean import run_robust_reference, plot_before_after


def main():
    cfg = load_config()

    # 1. Define folders (using parents[1] as root for 'steps')
    pipeline_root = Path(__file__).resolve().parents[1]
    clean_dir = pipeline_root / "outputs" / "derivatives" / "cleaned_raw"
    out_dir = pipeline_root / "outputs" / "derivatives" / "referenced_raw"
    qc_dir = pipeline_root / "outputs" / "qc_figs" / "reference_check"

    # Create output folders if they don't exist
    out_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    # 2. Check input files
    if not clean_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {clean_dir}. Run Step 01 first.")

    files = sorted(list(clean_dir.glob("*_cleaned-raw.fif")))
    if not files:
        print("No files found! Make sure Step 01 ran successfully.")
        return

    # 3. Loop over subjects
    for f in files:
        subj = f.name.split("_")[0]
        print(f"--- Processing {subj} ---")

        # Load the data from Step 01
        # It already has the correct channel types (EEG vs EMG) set
        raw = read_raw(f, fmt="fif", montage=cfg['montage'])

        # --- RUN PYPREP (Robust Referencing) ---
        # This uses RANSAC to find bad EEG channels and interpolates them.
        # It calculates the average reference excluding those bad channels.
        print("Running PyPREP/RANSAC (this takes ~30-60s)...")
        raw_ref, bads = run_robust_reference(raw)

        # Save QC Plot (Before vs After)
        # We save this to 'qc_figs/reference_check'
        qc_plot_path = qc_dir / f"{subj}_ref_check.png"
        plot_before_after(raw, raw_ref, bads, qc_plot_path)
        print(f"Saved QC plot to {qc_plot_path}")

        # Save Final Data to 'referenced_raw'
        save_clean_raw(raw_ref, out_dir, subj, "referenced")
        print(f"Saved {subj} referenced data.")


if __name__ == "__main__":
    main()