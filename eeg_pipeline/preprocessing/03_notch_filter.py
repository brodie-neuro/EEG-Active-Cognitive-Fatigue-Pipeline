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
from src.utils_report import QCReport


def _process_notch_branch(cfg, logger, input_subdir, output_subdir, step_name, branch_label, subject):
    pipeline_root = Path(__file__).resolve().parents[1]
    out_dir = pipeline_root / "outputs" / "derivatives" / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    files = iter_derivative_files(input_subdir, "*_referenced-raw.fif", subject=subject)
    if not files:
        logger.info("No files found for %s branch (%s).", branch_label, input_subdir)
        return

    for f in files:
        subj = subj_id_from_derivative(f)
        logger.info("--- Processing %s (%s) ---", subj, branch_label)

        raw = read_raw(f, fmt="fif", montage=cfg['montage'])

        freqs = [50.0, 100.0]
        logger.info("Applying notch filter at %s Hz (%s)...", freqs, branch_label)
        raw.notch_filter(
            freqs=freqs,
            picks='eeg',
            method='fir',
            fir_design='firwin',
            verbose=False,
        )

        block_str = ''.join(c for c in f.stem if c.isdigit())
        block_num = int(block_str[-1]) if block_str else 1
        out_file = save_clean_raw(raw, out_dir, subj, "notch")

        qc = QCReport(subj, block_num)
        qc.log_step(
            step_name,
            status="PASS",
            metrics={
                "method": "notch_fir",
                "freqs_removed": str(freqs),
            },
            params_used={"method": "notch_fir", "freqs_hz": freqs, "branch": branch_label},
            input_file=str(f),
            output_file=out_file,
        )
        qc.save_report()

        step_qc_path = save_step_qc(
            step_name,
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
                    "branch": branch_label,
                },
                "step_specific": {
                    "method": "notch_fir",
                    "freqs_removed": freqs,
                    "note": "Simple FIR notch filter, deterministic across hardware",
                },
            },
        )
        logger.info("Saved step QC log: %s", step_qc_path)
        logger.info("Saved %s", out_file)


def main():
    parser = argparse.ArgumentParser(description="Step 03 (simple): notch filter 50 Hz")
    parser.add_argument("--subject", type=str, default="")
    args = parser.parse_args()

    cfg = load_config()
    logger = setup_pipeline_logger('03_notch_filter')
    _process_notch_branch(
        cfg,
        logger,
        input_subdir="referenced_raw",
        output_subdir="notch_raw",
        step_name="03_notch_filter",
        branch_label="main_oscillatory_path",
        subject=args.subject,
    )
    _process_notch_branch(
        cfg,
        logger,
        input_subdir="erp_referenced_raw",
        output_subdir="erp_notch_raw",
        step_name="03_notch_filter_erp",
        branch_label="erp_p3b_branch",
        subject=args.subject,
    )

    logger.info("Step 03 (notch) complete.")


if __name__ == "__main__":
    main()
