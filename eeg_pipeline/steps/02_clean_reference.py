# steps/02_clean_reference.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Fix path

from src.utils_io import load_config, glob_subjects, read_raw, save_clean_raw
from src.utils_clean import run_robust_reference, plot_before_after


def main():
    cfg = load_config()

    # 1. Grab the "cleaned" files from Step 01 (Import)
    # Note: We look in the 'derivatives' folder now, not 'raw'
    clean_dir = Path("outputs/derivatives/cleaned_raw")
    files = sorted(list(clean_dir.glob("*_cleaned-raw.fif")))

    out_dir = Path("outputs/derivatives/referenced_raw")
    qc_dir = Path("outputs/qc_figs/reference_check")
    out_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        subj = f.name.split("_")[0]
        print(f"Processing {subj}...")

        # Load data (preload needed for processing)
        raw = read_raw(f, fmt="fif", montage=cfg['montage'])

        # --- RUN PYPREP ---
        raw_ref, bads = run_robust_reference(raw)

        # Save QC Plot
        plot_before_after(raw, raw_ref, bads, qc_dir / f"{subj}_ref_check.png")

        # Save Data
        save_clean_raw(raw_ref, out_dir, subj, "referenced")
        print(f"Saved {subj} referenced data.")


if __name__ == "__main__":
    main()