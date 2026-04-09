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
from src.utils_report import QCReport


def _process_reference_branch(cfg, logger, input_subdir, output_subdir, step_name, branch_label, subject):
    pipeline_root = Path(__file__).resolve().parents[1]
    out_dir = pipeline_root / "outputs" / "derivatives" / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    files = iter_derivative_files(input_subdir, "*_cleaned-raw.fif", subject=subject)
    if not files:
        logger.info("No files found for %s branch (%s).", branch_label, input_subdir)
        return

    for f in files:
        subj = subj_id_from_derivative(f)
        logger.info("--- Processing %s (%s) ---", subj, branch_label)

        raw = read_raw(f, fmt="fif", montage=cfg['montage'])

        pconfig = load_participant_config(subj)
        known_bad_eeg = pconfig.get('known_bad_eeg', [])
        if known_bad_eeg:
            valid_bads = [ch for ch in known_bad_eeg if ch in raw.ch_names]
            if valid_bads:
                raw.info['bads'] = list(set(raw.info['bads'] + valid_bads))
                logger.info("Marked known bad channels: %s", valid_bads)

        bads = list(raw.info['bads'])
        if bads:
            logger.info("Interpolating %d bad channels: %s", len(bads), bads)
            raw.interpolate_bads(reset_bads=True)

        logger.info("Applying simple average reference (%s)...", branch_label)
        raw.set_eeg_reference('average', projection=False)

        block_str = ''.join(c for c in f.stem if c.isdigit())
        block_num = int(block_str[-1]) if block_str else 1
        out_file = save_clean_raw(raw, out_dir, subj, "referenced")

        qc = QCReport(subj, block_num)
        qc.log_step(
            step_name,
            status="PASS",
            metrics={
                "reference_method": "simple_average",
                "n_channels_interpolated": len(bads),
                "known_bads_interpolated": str(bads),
            },
            params_used={"reference_method": "simple_average", "branch": branch_label},
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
                    "reference_method": "simple_average",
                    "branch": branch_label,
                },
                "step_specific": {
                    "reference_method": "simple_average",
                    "known_bads_interpolated": bads,
                    "known_bads_hash": stable_json_hash(sorted(bads)),
                    "n_bads": len(bads),
                },
            },
        )
        logger.info("Saved step QC log: %s", step_qc_path)
        logger.info("Saved %s", out_file)


def main():
    parser = argparse.ArgumentParser(description="Step 02 (simple): average reference")
    parser.add_argument("--subject", type=str, default="")
    args = parser.parse_args()

    cfg = load_config()
    logger = setup_pipeline_logger('02_simple_reference')
    _process_reference_branch(
        cfg,
        logger,
        input_subdir="cleaned_raw",
        output_subdir="referenced_raw",
        step_name="02_reference",
        branch_label="main_oscillatory_path",
        subject=args.subject,
    )
    _process_reference_branch(
        cfg,
        logger,
        input_subdir="erp_cleaned_raw",
        output_subdir="erp_referenced_raw",
        step_name="02_reference_erp",
        branch_label="erp_p3b_branch",
        subject=args.subject,
    )

    logger.info("Step 02 (simple) complete.")


if __name__ == "__main__":
    main()
