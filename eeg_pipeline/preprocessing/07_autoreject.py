# steps/07_autoreject.py
"""
Step 07: Autoreject for final epoch-level quality control.
Automatically finds optimal rejection thresholds and repairs bad epochs.
"""
import argparse
import json
import sys
from pathlib import Path
import re

pipeline_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pipeline_dir))

from src.utils_determinism import (
    array_hash,
    file_sha256,
    get_runtime_info,
    log_runtime_determinism_info,
    save_step_qc,
    set_determinism_env,
    set_random_seeds,
    stable_json_hash,
    threadpool_limit_context,
)

# Deterministic BLAS/LAPACK thread limits: set before scientific imports.
set_determinism_env()
set_random_seeds()

import mne
import numpy as np

from autoreject import AutoReject

from src.utils_io import (
    load_config,
    subj_id_from_derivative,
    iter_derivative_files,
    subject_derivatives_dir,
)
from src.utils_config import get_param
from src.utils_report import QCReport, qc_epoch_summary
from src.utils_qc import is_synthetic_subject
from src.utils_logging import setup_pipeline_logger

_BLOCK_RE = re.compile(r"block(\d+)", re.IGNORECASE)


def _parse_block_num(filename_stem: str) -> int:
    m = _BLOCK_RE.search(filename_stem)
    return int(m.group(1)) if m else 1


def _reject_log_payload(reject_log):
    if reject_log is None:
        return None
    return {
        "bad_epochs": np.asarray(reject_log.bad_epochs, dtype=np.uint8).tolist(),
        "labels": np.asarray(reject_log.labels).tolist(),
        "ch_names": list(getattr(reject_log, "ch_names", [])),
    }


def _run_autoreject_once(epochs, ar_params):
    """Run AutoReject once and return output plus deterministic diagnostics."""
    # `cv` is AutoReject's internal cross-validation fold count only.
    # It does not affect ICLabel or any ICA component classification logic.
    ar = AutoReject(
        n_interpolate=ar_params.get("n_interpolate", [1, 4, 8, 16]),
        consensus=ar_params.get("consensus", [0.1, 0.5, 1.0]),
        cv=ar_params.get("cv", 10),
        random_state=ar_params.get("random_state", 42),
        n_jobs=1,
        verbose=False,
    )

    input_data = epochs.get_data()
    diag = {
        "config": {
            "n_interpolate": ar_params.get("n_interpolate", [1, 4, 8, 16]),
            "consensus": ar_params.get("consensus", [0.1, 0.5, 1.0]),
            "cv": ar_params.get("cv", 10),
            "random_state": ar_params.get("random_state", 42),
            "n_jobs": 1,
        },
        "runtime_info": get_runtime_info(),
        "threadpool_limited": True,
        "input_hash": array_hash(input_data),
        "input_shape": list(input_data.shape),
    }

    fit_error = ""
    reject_log = None

    try:
        with threadpool_limit_context():
            epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)
    except Exception as exc:
        fit_error = f"{type(exc).__name__}: {exc}"
        raise RuntimeError(f"Autoreject fit_transform failed: {fit_error}") from exc

    output_data = epochs_clean.get_data()
    reject_log_payload = _reject_log_payload(reject_log)
    diag["output_hash"] = array_hash(output_data)
    diag["output_shape"] = list(output_data.shape)
    diag["reject_log_hash"] = (
        stable_json_hash(reject_log_payload) if reject_log_payload is not None else "NONE"
    )
    diag["reject_log_n_bad_epochs"] = (
        int(np.sum(reject_log.bad_epochs)) if reject_log is not None else 0
    )
    diag["fit_error"] = fit_error

    return epochs_clean, reject_log, diag


def main():
    parser = argparse.ArgumentParser(description="Step 07: autoreject")
    parser.add_argument(
        "--subject",
        type=str,
        default="",
        help="Optional subject filter (e.g. sub-dario or sub-a,sub-b).",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=0,
        help="Run Autoreject N additional times per block and compare deterministic hashes.",
    )
    args = parser.parse_args()

    cfg = load_config()
    logger = setup_pipeline_logger("07_autoreject")
    runtime_info = log_runtime_determinism_info(logger)

    pipeline_root = Path(__file__).resolve().parents[1]
    input_dir = pipeline_root / "outputs" / "derivatives" / "epochs"
    qc_dir = pipeline_root / "outputs" / "qc_figs" / "autoreject"
    debug_dir = pipeline_root / "outputs" / "features" / "autoreject_debug"

    qc_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"Legacy input directory not found: {input_dir}. Trying per-subject layout.")

    for epoch_type in ["p3b", "pac"]:
        files = iter_derivative_files("epochs", f"*_{epoch_type}-epo.fif", subject=args.subject)

        for f in files:
            subj = subj_id_from_derivative(f)
            block_num = _parse_block_num(f.stem)
            subj_output_dir = subject_derivatives_dir(subj, "epochs_clean")
            subj_output_dir.mkdir(parents=True, exist_ok=True)
            print(f"--- Processing {subj} block {block_num} ({epoch_type} epochs, Autoreject) ---")

            epochs = mne.read_epochs(f, preload=True)
            n_before = len(epochs)

            if is_synthetic_subject(subj):
                print("Synthetic data -- skipping autoreject by design.")
                out_file = subj_output_dir / f"{subj}_{epoch_type}_clean-epo.fif"
                epochs.save(out_file, overwrite=True)
                save_step_qc(
                    "07_autoreject",
                    subj,
                    block_num,
                    {
                        "status": "SKIPPED",
                        "input_file": str(f),
                        "input_hash": file_sha256(f),
                        "output_file": str(out_file),
                        "output_hash": file_sha256(out_file),
                        "parameters_used": {"epoch_type": epoch_type},
                        "step_specific": {
                            "skip_reason": "synthetic_subject",
                            "epoch_type": epoch_type,
                        },
                    },
                )
                continue

            if n_before < 10:
                raise RuntimeError(
                    f"Only {n_before} epochs available for {subj} {epoch_type}; "
                    "autoreject requires at least 10."
                )

            if "eeg" not in epochs.get_channel_types():
                raise ValueError(f"No EEG channels found for {subj} {epoch_type}; cannot run autoreject.")

            ar_params = get_param("autoreject")
            print(f"Running Autoreject on {n_before} epochs (n_jobs=1, deterministic controls applied)...")

            epochs_clean, reject_log, diag = _run_autoreject_once(epochs, ar_params)
            diag["subject"] = subj
            diag["block"] = block_num
            diag["epoch_type"] = epoch_type
            diag["runtime_info"] = runtime_info

            n_after = len(epochs_clean)
            n_rejected = n_before - n_after
            print(f"  Rejected {n_rejected}/{n_before} epochs ({100*n_rejected/n_before:.1f}%)")
            print(
                f"  Autoreject hashes: input={diag['input_hash']} "
                f"output={diag['output_hash']} reject_log={diag['reject_log_hash']}"
            )

            if args.repeat > 0:
                print(f"\n  === REPRODUCIBILITY TEST: {args.repeat} additional Autoreject runs ===")
                all_diags = [diag]
                for rep in range(args.repeat):
                    print(f"  Run {rep + 2}/{args.repeat + 1}...")
                    _, _, rep_diag = _run_autoreject_once(epochs, ar_params)
                    rep_diag["subject"] = subj
                    rep_diag["block"] = block_num
                    rep_diag["epoch_type"] = epoch_type
                    rep_diag["runtime_info"] = runtime_info
                    all_diags.append(rep_diag)

                input_hashes = [d.get("input_hash", "N/A") for d in all_diags]
                output_hashes = [d.get("output_hash", "N/A") for d in all_diags]
                reject_hashes = [d.get("reject_log_hash", "N/A") for d in all_diags]
                repeat_summary = {
                    "subject": subj,
                    "block": block_num,
                    "epoch_type": epoch_type,
                    "n_runs": len(all_diags),
                    "input_identical": len(set(input_hashes)) == 1,
                    "output_identical": len(set(output_hashes)) == 1,
                    "reject_log_identical": len(set(reject_hashes)) == 1,
                    "runs": all_diags,
                }
                repeat_path = (
                    debug_dir / f"{subj}_block{block_num}_{epoch_type}_repeat_test.json"
                )
                with open(repeat_path, "w", encoding="utf-8") as fp:
                    json.dump(repeat_summary, fp, indent=2, default=str)
                print(
                    f"  input identical: {repeat_summary['input_identical']} | "
                    f"output identical: {repeat_summary['output_identical']} | "
                    f"reject log identical: {repeat_summary['reject_log_identical']}"
                )
                print(f"  Saved repeat diagnostics to {repeat_path}")

            min_trials = get_param("qc", default={}).get("min_trials_per_block", 50)
            if n_after < min_trials:
                print(f"  WARNING: Only {n_after} trials remain (below minimum {min_trials})")

            try:
                if reject_log is not None:
                    fig = reject_log.plot(orientation="horizontal", show=False)
                    fig.savefig(qc_dir / f"{subj}_{epoch_type}_reject_log.png", dpi=100)
                    print(f"  Saved reject log to {qc_dir}")
            except Exception as exc:
                print(f"  Could not save reject log: {exc}")

            diag_path = debug_dir / f"{subj}_block{block_num}_{epoch_type}_autoreject_diag.json"
            with open(diag_path, "w", encoding="utf-8") as fp:
                json.dump(diag, fp, indent=2, default=str)

            out_file = subj_output_dir / f"{subj}_{epoch_type}_clean-epo.fif"
            epochs_clean.save(
                out_file,
                overwrite=True,
            )

            qc = QCReport(subj, block_num)
            reject_pct = 100 * n_rejected / n_before if n_before > 0 else 0
            status = qc.assess_metric(
                "Epoch rejection",
                reject_pct,
                "max_epoch_rejection_pct",
                "<=",
            )
            if n_after < min_trials:
                status = "FAIL"

            try:
                fig, metrics = qc_epoch_summary(
                    epochs_clean,
                    title=f"{subj} {epoch_type} ({n_after}/{n_before} kept)",
                )
                qc.add_figure(f"08_autoreject_{epoch_type}", fig)
            except Exception as exc:
                print(f"  Could not save epoch QC figure: {exc}")
                metrics = {}

            qc.log_step(
                f"08_autoreject_{epoch_type}",
                status=status,
                metrics={
                    "n_before": n_before,
                    "n_after": n_after,
                    "n_rejected": n_rejected,
                    "rejection_pct": round(reject_pct, 1),
                    "trials_remaining": n_after,
                    "min_trial_threshold": min_trials,
                    "below_min_threshold": n_after < min_trials,
                    "fit_error": diag["fit_error"],
                    "input_hash": diag["input_hash"],
                    "output_hash": diag["output_hash"],
                    "reject_log_hash": diag["reject_log_hash"],
                    **metrics,
                },
                params_used=ar_params,
            )
            qc.save_report()
            step_qc_path = save_step_qc(
                "07_autoreject",
                subj,
                block_num,
                {
                    "status": status,
                    "input_file": str(f),
                    "input_hash": file_sha256(f),
                    "output_file": str(out_file),
                    "output_hash": file_sha256(out_file),
                    "parameters_used": {**ar_params, "epoch_type": epoch_type},
                    "step_specific": {
                        "epoch_type": epoch_type,
                        "n_before": int(n_before),
                        "n_after": int(n_after),
                        "n_rejected": int(n_rejected),
                        "reject_log_hash": diag["reject_log_hash"],
                        "input_hash": diag["input_hash"],
                        "output_hash": diag["output_hash"],
                    },
                },
            )
            logger.info("Saved step QC log: %s", step_qc_path)
            print(f"  Saved clean epochs for {subj}.\n")


if __name__ == "__main__":
    main()
