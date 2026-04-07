# steps/02_simple_reference.py
"""
Simplified Step 02: Simple average reference (no RANSAC, no eigendecomposition).

Replaces the robust RANSAC reference with a simple arithmetic average reference.
Only interpolates known bad channels from per-participant configs.
This removes eigendecomposition #1 from the preprocessing chain.
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
import mne

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils_io import (
    load_config,
    load_participant_config,
    read_raw,
    save_clean_raw,
    subj_id_from_derivative,
    iter_derivative_files,
)
from src.utils_determinism import file_sha256, save_step_qc, stable_json_hash
from src.utils_logging import setup_pipeline_logger


def main():
    parser = argparse.ArgumentParser(description="Step 02 (simple): average reference")
    parser.add_argument("--subject", type=str, default="")
    args = parser.parse_args()

    cfg = load_config()
    logger = setup_pipeline_logger('02_simple_reference')

    pipeline_root = Path(__file__).resolve().parents[1]
    out_dir = pipeline_root / "outputs" / "derivatives" / "referenced_raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = iter_derivative_files("cleaned_raw", "*_cleaned-raw.fif", subject=args.subject)
    if not files:
        print("No files found! Make sure Step 01 ran successfully.")
        return

    for f in files:
        subj = subj_id_from_derivative(f)
        logger.info("--- Processing %s (SIMPLE REFERENCE) ---", subj)

        raw = read_raw(f, fmt="fif", montage=cfg['montage'])

        # Load and apply known bad channels from per-participant config
        pconfig = load_participant_config(subj)
        known_bad_eeg = pconfig.get('known_bad_eeg', [])
        if known_bad_eeg:
            valid_bads = [ch for ch in known_bad_eeg if ch in raw.ch_names]
            if valid_bads:
                raw.info['bads'] = list(set(raw.info['bads'] + valid_bads))
                logger.info("Marked known bad channels: %s", valid_bads)

        # Interpolate bad channels (uses spherical splines, deterministic)
        bads = list(raw.info['bads'])
        if bads:
            logger.info("Interpolating %d bad channels: %s", len(bads), bads)
            raw.interpolate_bads(reset_bads=True)

        # Simple average reference — pure arithmetic, no eigendecomposition
        logger.info("Applying simple average reference...")
        raw.set_eeg_reference('average', projection=False)

        block_str = ''.join(c for c in f.stem if c.isdigit())
        block_num = int(block_str[-1]) if block_str else 1

        out_file = save_clean_raw(raw, out_dir, subj, "referenced")

        step_qc_path = save_step_qc(
            "02_reference",
            subj,
            block_num,
            {
                "status": "PASS",
                "input_file": str(f),
                "input_hash": file_sha256(f),
                "output_file": str(out_file),
                "output_hash": file_sha256(out_file),
                "parameters_used": {"reference_method": "simple_average"},
                "step_specific": {
                    "reference_method": "simple_average",
                    "known_bads_interpolated": bads,
                    "n_bads": len(bads),
                },
            },
        )
        logger.info("Saved %s", out_file)

    logger.info("Step 02 (simple) complete.")


if __name__ == "__main__":
    main()
