# steps/03_notch_filter.py
"""
Simplified Step 03: FIR notch filter at 50 Hz.

a standard FIR notch filter. Pure frequency-domain operation, fully
deterministic on any hardware.
"""
import argparse
import os
import sys
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MNE_DONTWRITE_HOME"] = "true"
os.environ.setdefault("_MNE_FAKE_HOME_DIR", os.path.dirname(os.path.dirname(__file__)))
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils_io import (
    load_config,
    read_raw,
    save_clean_raw,
    subj_id_from_derivative,
    iter_derivative_files,
)
from src.utils_determinism import file_sha256, save_step_qc
from src.utils_logging import setup_pipeline_logger


def main():
    parser = argparse.ArgumentParser(description="Step 03 (simple): notch filter 50 Hz")
    parser.add_argument("--subject", type=str, default="")
    args = parser.parse_args()

    cfg = load_config()
    logger = setup_pipeline_logger('03_notch_filter')

    pipeline_root = Path(__file__).resolve().parents[1]
    out_dir = pipeline_root / "outputs" / "derivatives" / "notch_raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = iter_derivative_files("referenced_raw", "*_referenced-raw.fif", subject=args.subject)
    if not files:
        print("No files found! Make sure Step 02 ran successfully.")
        return

    for f in files:
        subj = subj_id_from_derivative(f)
        logger.info("--- Processing %s (NOTCH FILTER) ---", subj)

        raw = read_raw(f, fmt="fif", montage=cfg['montage'])

        # Simple notch filter at 50 Hz + harmonics (100 Hz)
        # Uses FIR filter — pure convolution, no eigendecomposition
        freqs = [50.0, 100.0]
        logger.info("Applying notch filter at %s Hz...", freqs)
        raw.notch_filter(
            freqs=freqs,
            picks='eeg',
            method='fir',
            fir_design='firwin',
            verbose=False,
        )

        block_str = ''.join(c for c in f.stem if c.isdigit())
        block_num = int(block_str[-1]) if block_str else 1

        # Save notch-filtered output
        out_file = save_clean_raw(raw, out_dir, subj, "notch")

        step_qc_path = save_step_qc(
            "03_notch_filter",
            subj,
            block_num,
            {
                "status": "PASS",
                "input_file": str(f),
                "input_hash": file_sha256(f),
                "output_file": str(out_file),
                "output_hash": file_sha256(out_file),
                "parameters_used": {
                    "method": "notch_fir",
                    "freqs_hz": freqs,
                },
                "step_specific": {
                    "method": "notch_fir",
                    "freqs_removed": freqs,
                    "note": "Simple FIR notch filter, fully deterministic across hardware",
                },
            },
        )
        logger.info("Saved %s", out_file)

    logger.info("Step 03 (notch) complete.")


if __name__ == "__main__":
    main()
