# steps/05_ica_iclabel.py
"""
Step 05: ICA artefact removal using ICLabel classification.
Fits ICA directly on the shared 1 Hz high-pass preprocessing stream, then applies
the solution back to that same stream.
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
from mne.preprocessing import ICA
from mne_icalabel import label_components
import numpy as np

from src.utils_io import (
    load_config,
    save_clean_raw,
    subj_id_from_derivative,
    iter_derivative_files,
)
from src.utils_config import get_param
from src.utils_report import QCReport
from src.utils_qc import is_synthetic_subject
from src.utils_logging import setup_pipeline_logger

_BLOCK_RE = re.compile(r"block(\d+)", re.IGNORECASE)


def _default_n_jobs() -> int:
    # LOCKED to 1 for deterministic reproducibility across all platforms/core counts.
    return 1


def _parse_block_num(filename_stem: str) -> int:
    m = _BLOCK_RE.search(filename_stem)
    return int(m.group(1)) if m else 1


def _matrix_hash(mat) -> str:
    if mat is None:
        return "NONE"
    return array_hash(np.ascontiguousarray(mat))


def _score_value(prob) -> float:
    arr = np.asarray(prob)
    if arr.ndim == 0:
        return float(arr)
    return float(arr.max())


def _run_ica_once(
    raw,
    n_components,
    ica_method,
    ica_fit_params,
    ica_seed,
    iclabel_thresholds,
    verbose=True,
):
    """Run ICA fit/classify/apply once and return output plus diagnostics."""
    label_to_config = {
        "eye blink": "eye",
        "eye": "eye",
        "heart beat": "heart",
        "heart": "heart",
        "muscle artifact": "muscle",
        "muscle": "muscle",
        "channel noise": "channel_noise",
        "line noise": "line_noise",
        "other": "other",
    }

    diag = {
        "config": {
            "n_components_requested": int(n_components),
            "method": ica_method,
            "fit_params": dict(ica_fit_params),
            "random_state": int(ica_seed),
            "thresholds": iclabel_thresholds,
        },
        "runtime_info": get_runtime_info(),
        "threadpool_limited": True,
        "raw_input_hash": array_hash(raw.get_data()),
        "raw_input_shape": list(raw.get_data().shape),
    }

    raw_ica_fit = raw.copy()
    raw_ica_fit.pick_types(eeg=True, exclude="bads")

    ica_input = raw_ica_fit.get_data()
    n_good_channels = len(raw_ica_fit.ch_names)
    actual_n_components = min(n_components, n_good_channels - 1)

    diag["ica_input_hash"] = array_hash(ica_input)
    diag["ica_input_shape"] = list(ica_input.shape)
    diag["n_good_channels"] = int(n_good_channels)
    diag["n_components_fit"] = int(actual_n_components)
    is_extended_infomax = (
        ica_method == "infomax" and bool(ica_fit_params.get("extended", False))
    )

    if actual_n_components < 5:
        raise RuntimeError(
            f"ICA requires at least 5 fit components, got {actual_n_components} "
            f"from {n_good_channels} good channels."
        )

    if verbose:
        if is_extended_infomax:
            print(
                f"Computing Extended Infomax ICA "
                f"(n={actual_n_components} components on {n_good_channels} channels, "
                f"method={ica_method}, fit_params={dict(ica_fit_params)})..."
            )
        else:
            print(
                f"Fitting ICA (n={actual_n_components} components on "
                f"{n_good_channels} channels, method={ica_method}, "
                f"fit_params={dict(ica_fit_params)})..."
            )

    ica = ICA(
        n_components=actual_n_components,
        method=ica_method,
        fit_params=ica_fit_params or None,
        random_state=ica_seed,
        max_iter=500,
    )
    with threadpool_limit_context():
        ica.fit(raw_ica_fit)

    diag["mixing_matrix_hash"] = _matrix_hash(getattr(ica, "mixing_matrix_", None))
    diag["unmixing_matrix_hash"] = _matrix_hash(getattr(ica, "unmixing_matrix_", None))
    diag["n_iter"] = int(getattr(ica, "n_iter_", -1))

    if verbose:
        print("Running ICLabel classifier...")
    try:
        with threadpool_limit_context():
            ic_labels = label_components(raw_ica_fit, ica, method="iclabel")
    except ImportError as exc:
        raise RuntimeError(
            "ICA cleaning failed: Missing onnxruntime or pytorch. Terminating pipeline."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"ICA cleaning failed during ICLabel classification: {exc}") from exc

    labels = list(ic_labels["labels"])
    probs = [_score_value(prob) for prob in ic_labels["y_pred_proba"]]
    exclude_idx = []

    if verbose:
        print("\n--- Component Classification ---")
    for i, (label, prob) in enumerate(zip(labels, probs)):
        if verbose:
            print(f"IC {i:02d}: {label} ({prob:.2f})")
        config_key = label_to_config.get(label, "other")
        threshold = iclabel_thresholds.get(config_key, 0.80)
        if label != "brain" and prob > threshold:
            exclude_idx.append(i)
            if verbose:
                print(f"  -> EXCLUDED (threshold: {threshold:.2f})")

    # ICLabel is the sole method for component classification.
    # No EOG correlation safety net: this study has no dedicated EOG electrodes,
    # and AF7/AF8 proxy screening was removed to avoid altering rejection decisions.
    eog_indices = []
    eog_bads = []
    exclude_idx = sorted(exclude_idx)
    if verbose:
        print(f"\nFinal exclusion list (ICLabel only): {exclude_idx}")

    ica.exclude = exclude_idx
    raw_clean = raw.copy()
    with threadpool_limit_context():
        ica.apply(raw_clean, exclude=exclude_idx)

    cleaned_eeg = raw_clean.get_data(picks=mne.pick_types(raw_clean.info, eeg=True, exclude=[]))
    diag["cleaned_output_hash"] = array_hash(cleaned_eeg)
    diag["cleaned_output_shape"] = list(cleaned_eeg.shape)
    diag["exclude_hash"] = stable_json_hash(exclude_idx)
    diag["component_labels"] = labels
    diag["component_scores"] = probs
    diag["exclude_idx"] = exclude_idx
    diag["eog_indices"] = list(eog_indices)
    diag["n_brain_remaining"] = int(sum(1 for label in labels if label == "brain"))
    diag["n_caught_by_eog_only"] = int(len(eog_bads))

    summary = {
        "n_components_fit": actual_n_components,
        "exclude_idx": exclude_idx,
        "n_brain_remaining": diag["n_brain_remaining"],
        "n_caught_by_eog_only": len(eog_bads),
        "labels": labels,
    }
    return raw_clean, ica, summary, diag


def main():
    parser = argparse.ArgumentParser(description="Step 05: ICA + ICLabel")
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
        help="Run ICA N additional times per block and compare deterministic hashes.",
    )
    args = parser.parse_args()

    cfg = load_config()
    logger = setup_pipeline_logger("05_ica_iclabel")
    runtime_info = log_runtime_determinism_info(logger)

    pipeline_root = Path(__file__).resolve().parents[1]
    input_dir = pipeline_root / "outputs" / "derivatives" / "asr_cleaned_raw"
    output_dir = pipeline_root / "outputs" / "derivatives" / "ica_cleaned_raw"
    qc_dir = pipeline_root / "outputs" / "qc_figs" / "ica"
    debug_dir = pipeline_root / "outputs" / "features" / "ica_debug"

    output_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"Legacy input directory not found: {input_dir}. Trying per-subject layout.")

    files = iter_derivative_files("asr_cleaned_raw", "*_asr-raw.fif", subject=args.subject)
    if not files:
        print("No files found to process.")
        return

    n_components = get_param("ica", "n_components", default=25)
    ica_method = get_param("ica", "method", default="infomax")
    ica_fit_params = dict(get_param("ica", "fit_params", default={}) or {})
    ica_seed = get_param("ica", "random_state", default=42)
    iclabel_thresholds = get_param("ica", "iclabel_thresholds", default={})

    for f in files:
        subj = subj_id_from_derivative(f)
        block_num = _parse_block_num(f.stem)
        print(f"--- Processing {subj} block {block_num} (ICA) ---")

        raw = mne.io.read_raw_fif(f, preload=True)

        if is_synthetic_subject(subj):
            print("Synthetic data detected -- skipping ICA, passing through.")
            out_file = save_clean_raw(raw, output_dir, subj, "ica")
            save_step_qc(
                "05_ica",
                subj,
                block_num,
                {
                    "status": "SKIPPED",
                    "input_file": str(f),
                    "input_hash": file_sha256(f),
                    "output_file": str(out_file),
                    "output_hash": file_sha256(out_file),
                    "parameters_used": {
                        "method": ica_method,
                        "fit_params": ica_fit_params,
                        "n_components": n_components,
                        "thresholds": iclabel_thresholds,
                        "random_state": ica_seed,
                    },
                    "step_specific": {
                        "skip_reason": "synthetic_subject",
                    },
                },
            )
            continue

        n_good = len([ch for ch in raw.ch_names if ch not in raw.info["bads"]])
        if n_good < 10:
            raise RuntimeError(f"Only {n_good} good channels available for ICA in {subj}.")

        raw_clean, ica, summary, diag = _run_ica_once(
            raw,
            n_components=n_components,
            ica_method=ica_method,
            ica_fit_params=ica_fit_params,
            ica_seed=ica_seed,
            iclabel_thresholds=iclabel_thresholds,
            verbose=True,
        )
        diag["subject"] = subj
        diag["block"] = block_num
        diag["runtime_info"] = runtime_info

        print(
            f"  ICA hashes: input={diag['ica_input_hash']} "
            f"mix={diag['mixing_matrix_hash']} "
            f"unmix={diag['unmixing_matrix_hash']} "
            f"output={diag['cleaned_output_hash']}"
        )

        if args.repeat > 0:
            print(f"\n  === REPRODUCIBILITY TEST: {args.repeat} additional ICA runs ===")
            all_diags = [diag]
            for rep in range(args.repeat):
                print(f"  Run {rep + 2}/{args.repeat + 1}...")
                _, _, _, rep_diag = _run_ica_once(
                    raw,
                    n_components=n_components,
                    ica_method=ica_method,
                    ica_fit_params=ica_fit_params,
                    ica_seed=ica_seed,
                    iclabel_thresholds=iclabel_thresholds,
                    verbose=False,
                )
                rep_diag["subject"] = subj
                rep_diag["block"] = block_num
                rep_diag["runtime_info"] = runtime_info
                all_diags.append(rep_diag)

            input_hashes = [d.get("ica_input_hash", "N/A") for d in all_diags]
            mix_hashes = [d.get("mixing_matrix_hash", "N/A") for d in all_diags]
            unmix_hashes = [d.get("unmixing_matrix_hash", "N/A") for d in all_diags]
            out_hashes = [d.get("cleaned_output_hash", "N/A") for d in all_diags]
            exclude_hashes = [d.get("exclude_hash", "N/A") for d in all_diags]

            repeat_summary = {
                "subject": subj,
                "block": block_num,
                "n_runs": len(all_diags),
                "input_identical": len(set(input_hashes)) == 1,
                "mixing_identical": len(set(mix_hashes)) == 1,
                "unmixing_identical": len(set(unmix_hashes)) == 1,
                "output_identical": len(set(out_hashes)) == 1,
                "exclude_identical": len(set(exclude_hashes)) == 1,
                "runs": all_diags,
            }
            repeat_path = debug_dir / f"{subj}_block{block_num}_repeat_test.json"
            with open(repeat_path, "w", encoding="utf-8") as fp:
                json.dump(repeat_summary, fp, indent=2, default=str)
            print(
                f"  input identical: {repeat_summary['input_identical']} | "
                f"mixing identical: {repeat_summary['mixing_identical']} | "
                f"unmixing identical: {repeat_summary['unmixing_identical']} | "
                f"output identical: {repeat_summary['output_identical']} | "
                f"exclude identical: {repeat_summary['exclude_identical']}"
            )
            print(f"  Saved repeat diagnostics to {repeat_path}")

        fig_topo = ica.plot_components(show=False)
        if isinstance(fig_topo, list):
            for i, fig in enumerate(fig_topo):
                fig.savefig(qc_dir / f"{subj}_ica_topo_part{i}.png")
        else:
            fig_topo.savefig(qc_dir / f"{subj}_ica_topo.png")

        if summary["exclude_idx"]:
            fig_src = ica.plot_sources(raw.copy().pick_types(eeg=True, exclude="bads"), picks=summary["exclude_idx"], show=False)
            fig_src.savefig(qc_dir / f"{subj}_ica_excluded_time.png")

        diag_path = debug_dir / f"{subj}_block{block_num}_ica_diag.json"
        with open(diag_path, "w", encoding="utf-8") as fp:
            json.dump(diag, fp, indent=2, default=str)

        qc = QCReport(subj, block_num)
        status = qc.assess_metric(
            "ICs rejected",
            len(summary["exclude_idx"]),
            "max_ica_components_rejected",
            "<=",
        )
        if summary["n_brain_remaining"] < get_param("qc", "min_brain_ics_remaining", default=15):
            status = "WARNING"

        qc.log_step(
            "05_ica",
            status=status,
            metrics={
                "n_components_fit": summary["n_components_fit"],
                "n_excluded": len(summary["exclude_idx"]),
                "n_brain_remaining": summary["n_brain_remaining"],
                "n_caught_by_eog_only": summary["n_caught_by_eog_only"],
                "excluded_indices": str(summary["exclude_idx"]),
                "ica_input_hash": diag["ica_input_hash"],
                "ica_output_hash": diag["cleaned_output_hash"],
            },
            params_used={
                "method": ica_method,
                "fit_params": ica_fit_params,
                "n_components": n_components,
                "thresholds": iclabel_thresholds,
                "random_state": ica_seed,
            },
        )
        qc.save_report()

        out_file = save_clean_raw(raw_clean, output_dir, subj, "ica")
        step_qc_path = save_step_qc(
            "05_ica",
            subj,
            block_num,
            {
                "status": status,
                "input_file": str(f),
                "input_hash": file_sha256(f),
                "output_file": str(out_file),
                "output_hash": file_sha256(out_file),
                "parameters_used": {
                    "method": ica_method,
                    "fit_params": ica_fit_params,
                    "n_components": n_components,
                    "thresholds": iclabel_thresholds,
                    "random_state": ica_seed,
                },
                "step_specific": {
                    "n_components": summary["n_components_fit"],
                    "excluded_indices": summary["exclude_idx"],
                    "labels": summary["labels"],
                    "mixing_hash": diag["mixing_matrix_hash"],
                    "unmixing_hash": diag["unmixing_matrix_hash"],
                    "input_hash": diag["ica_input_hash"],
                    "output_hash": diag["cleaned_output_hash"],
                    "exclude_hash": diag["exclude_hash"],
                },
            },
        )
        logger.info("Saved step QC log: %s", step_qc_path)
        print(f"Saved {subj} ICA cleaned data.\n")


if __name__ == "__main__":
    main()
