# steps/06_epoch.py
"""
Step 06: Epoch creation for P3b and PAC analysis.
Creates two separate epoch sets with different time-locking.
"""
import argparse
import os
import sys
from pathlib import Path
# Deterministic BLAS/LAPACK thread limits: set before scientific imports.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MNE_DONTWRITE_HOME"] = "true"
os.environ.setdefault("_MNE_FAKE_HOME_DIR", os.path.dirname(os.path.dirname(__file__)))
import mne

pipeline_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pipeline_dir))
from src.utils_io import (
    load_config,
    subj_id_from_derivative,
    iter_derivative_files,
    subject_derivatives_dir,
)
from src.utils_config import get_param
from src.utils_determinism import file_sha256, save_step_qc, stable_json_hash
from src.utils_report import QCReport
from src.utils_logging import setup_pipeline_logger
from src.utils_adapter import map_event_id_labels


def _require_onset_events(event_id: dict[str, int]) -> dict[str, int]:
    """Return explicit stimulus-onset events or fail hard."""
    onset = {
        k: v for k, v in event_id.items()
        if "stim" in str(k).lower() and "offset" not in str(k).lower()
    }
    if onset:
        return onset
    raise ValueError(
        "Could not identify explicit stimulus onset events after adapter mapping. "
        f"Available event ids: {event_id}"
    )


def _select_onset_event_rows(events, onset_event_id: dict[str, int]):
    """Return only stimulus-onset event rows used for epoching."""
    import numpy as np

    onset_codes = sorted({int(code) for code in onset_event_id.values()})
    if not onset_codes:
        return events[:0].copy()
    return events[np.isin(events[:, 2], onset_codes)].copy()


def _trim_practice_onsets(onset_events, block_num: int, practice_cfg: dict):
    """Trim prepended practice onsets by keeping the final task trials."""
    enabled = bool(practice_cfg.get("enabled", False))
    trim_blocks = {int(b) for b in practice_cfg.get("blocks", [1])}
    expected_task_onsets = int(practice_cfg.get("expected_task_onsets", 0) or 0)
    max_extra_onsets = int(practice_cfg.get("max_extra_onsets", 0) or 0)

    before_n = int(len(onset_events))
    meta = {
        "enabled": enabled,
        "applied": False,
        "before_n": before_n,
        "after_n": before_n,
        "dropped_n": 0,
        "expected_task_onsets": expected_task_onsets,
    }

    if not enabled or block_num not in trim_blocks or expected_task_onsets <= 0:
        return onset_events, meta

    if before_n <= expected_task_onsets:
        return onset_events, meta

    extra_n = before_n - expected_task_onsets
    if max_extra_onsets > 0 and extra_n > max_extra_onsets:
        raise RuntimeError(
            f"Block {block_num}: found {before_n} onset events, which is "
            f"{extra_n} above expected {expected_task_onsets}. "
            f"This exceeds epoching.practice_trim.max_extra_onsets={max_extra_onsets}."
        )

    trimmed = onset_events[-expected_task_onsets:].copy()
    meta["applied"] = True
    meta["after_n"] = int(len(trimmed))
    meta["dropped_n"] = extra_n
    return trimmed, meta


def _prepare_epoch_inputs(raw, cfg: dict):
    """Return events, mapped event ids, and stimulus-onset rows for epoching."""
    events, event_id = mne.events_from_annotations(raw)
    mapped_event_id = map_event_id_labels(event_id, cfg.get('adapter', {}))
    if mapped_event_id != event_id:
        print("Applied adapter event label mapping.")
        event_id = mapped_event_id

    if not any("stim" in str(k).lower() for k in event_id):
        default_map = {"1": "stim", "2": "stim/offset"}
        event_id = {default_map.get(str(k), str(k)): v for k, v in event_id.items()}
        print(f"Applied default WAND event mapping: {event_id}")

    if len(events) == 0:
        raise RuntimeError("No events found; cannot epoch data.")

    stim_events = _require_onset_events(event_id)
    onset_event_rows = _select_onset_event_rows(events, stim_events)
    return events, event_id, stim_events, onset_event_rows


def main():
    parser = argparse.ArgumentParser(description="Step 06: epoch creation")
    parser.add_argument(
        "--subject",
        type=str,
        default="",
        help="Optional subject filter (e.g. sub-dario or sub-a,sub-b).",
    )
    args = parser.parse_args()

    cfg = load_config()
    logger = setup_pipeline_logger("06_epoch")
    
    pipeline_root = Path(__file__).resolve().parents[1]
    input_dir = pipeline_root / "outputs" / "derivatives" / "ica_cleaned_raw"
    output_dir = pipeline_root / "outputs" / "derivatives" / "epochs"
    erp_branch_cfg = get_param("erp_branch", default={}) or {}
    erp_branch_enabled = bool(erp_branch_cfg.get("enabled", False))
    erp_epoch_type = str(erp_branch_cfg.get("epoch_type", "p3b_erp"))

    if not input_dir.exists():
        print(f"Legacy input directory not found: {input_dir}. Trying per-subject layout.")

    files = iter_derivative_files("ica_cleaned_raw", "*_ica-raw.fif", subject=args.subject)
    if not files:
        print("No files found to process.")
        return
    
    for f in files:
        subj = subj_id_from_derivative(f)
        print(f"--- Processing {subj} (Epoching) ---")
        subj_epochs_dir = subject_derivatives_dir(subj, "epochs")
        subj_epochs_dir.mkdir(parents=True, exist_ok=True)
        epochs_p3b = None
        epochs_pac = None
        epochs_p3b_erp = None
        
        raw = mne.io.read_raw_fif(f, preload=True)
        events, event_id, stim_events, onset_event_rows = _prepare_epoch_inputs(raw, cfg)
        
        # Extract block number for QC
        block_str = ''.join(c for c in f.stem if c.isdigit())
        block_num = int(block_str[-1]) if block_str else 1
        qc = QCReport(subj, block_num)

        p3b_cfg = get_param('epoching', 'p3b', default={})
        pac_cfg_params = get_param('epoching', 'pac', default={})
        if not p3b_cfg or not pac_cfg_params:
            raise KeyError("Missing required epoching.p3b or epoching.pac config.")

        practice_cfg = get_param('epoching', 'practice_trim', default={}) or {}
        
        # P3b Epochs: Stimulus ONSET locked only (Trigger 1), -200 to +800 ms
        # Exclude stim/offset events (Trigger 2) -- those mark end of stimulus
        print("Creating P3b epochs (stimulus-onset-locked)...")
        onset_event_rows, practice_meta = _trim_practice_onsets(
            onset_event_rows, block_num, practice_cfg
        )
        if practice_meta["applied"]:
            print(
                f"  Trimmed {practice_meta['dropped_n']} prepended practice onset trials "
                f"from block {block_num}; keeping final {practice_meta['after_n']} task onsets."
            )
        p3b_tmin = p3b_cfg.get('tmin', -0.2)
        p3b_tmax = p3b_cfg.get('tmax', 0.8)
        p3b_baseline = tuple(p3b_cfg.get('baseline', [-0.2, 0.0]))
        epochs_p3b = mne.Epochs(
            raw, onset_event_rows, stim_events,
            tmin=p3b_tmin, tmax=p3b_tmax,
            baseline=p3b_baseline,
            preload=True,
            reject=None,  # Autoreject handles this in Step 08
            verbose=False
        )
        p3b_out = subj_epochs_dir / f"{subj}_p3b-epo.fif"
        epochs_p3b.save(p3b_out, overwrite=True)
        print(f"  P3b epochs: {len(epochs_p3b)} trials")

        if erp_branch_enabled:
            erp_raw_path = subject_derivatives_dir(subj, "erp_notch_raw") / f"{subj}_notch-raw.fif"
            if not erp_raw_path.exists():
                raise FileNotFoundError(
                    f"ERP branch is enabled but no ERP-notch file was found for {subj}: {erp_raw_path}"
                )

            print("Creating ERP-branch P3b epochs from dedicated 0.1 Hz stream...")
            raw_erp = mne.io.read_raw_fif(erp_raw_path, preload=True)
            _, event_id_erp, stim_events_erp, onset_event_rows_erp = _prepare_epoch_inputs(raw_erp, cfg)
            onset_event_rows_erp, practice_meta_erp = _trim_practice_onsets(
                onset_event_rows_erp, block_num, practice_cfg
            )
            if practice_meta_erp["applied"]:
                print(
                    f"  ERP branch trimmed {practice_meta_erp['dropped_n']} prepended practice onset trials "
                    f"from block {block_num}; keeping final {practice_meta_erp['after_n']} task onsets."
                )
            epochs_p3b_erp = mne.Epochs(
                raw_erp,
                onset_event_rows_erp,
                stim_events_erp,
                tmin=p3b_tmin,
                tmax=p3b_tmax,
                baseline=p3b_baseline,
                preload=True,
                reject=None,
                verbose=False,
            )
            p3b_erp_out = subj_epochs_dir / f"{subj}_{erp_epoch_type}-epo.fif"
            epochs_p3b_erp.save(p3b_erp_out, overwrite=True)
            print(f"  P3b ERP-branch epochs: {len(epochs_p3b_erp)} trials")
        else:
            p3b_erp_out = None
            practice_meta_erp = None
        
        # PAC Epochs:
        # Stimulus-ONSET locked, with pre-stimulus buffer for filter settling.
        # tmin/tmax from config: -0.5 to 1.8s (buffer + stimulus + maintenance).
        # PAC analysis script filters on full epoch, then crops to 0-0.6s.
        pac_tmin = pac_cfg_params.get('tmin', -0.5)
        pac_tmax = pac_cfg_params.get('tmax', 1.8)
        print(f'Creating PAC epochs (onset-locked, {pac_tmin} to {pac_tmax}s)...')

        epochs_pac = mne.Epochs(
            raw, onset_event_rows, stim_events,
            tmin=pac_tmin, tmax=pac_tmax,
            baseline=None,  # No baseline for PAC
            preload=True,
            reject=None,
            verbose=False
        )
        pac_out = subj_epochs_dir / f"{subj}_pac-epo.fif"
        epochs_pac.save(pac_out, overwrite=True)
        print(f"  PAC epochs: {len(epochs_pac)} trials")
        
        # QC report
        n_p3b = len(epochs_p3b) if epochs_p3b is not None else 0
        n_pac = len(epochs_pac) if epochs_pac is not None else 0
        n_p3b_erp = len(epochs_p3b_erp) if epochs_p3b_erp is not None else 0
        qc.log_step('06_epoch', status='PASS',
                     metrics={
                          'n_events_total': len(events),
                         'n_onset_events_before_trim': int(practice_meta['before_n']),
                         'n_onset_events_after_trim': int(practice_meta['after_n']),
                         'n_practice_onsets_dropped': int(practice_meta['dropped_n']),
                         'n_p3b_epochs': n_p3b,
                         'n_p3b_erp_epochs': n_p3b_erp,
                         'n_pac_epochs': n_pac,
                      })
        qc.save_report()
        output_file = {
            "p3b": str(p3b_out),
            "pac": str(pac_out),
        }
        output_hash = {
            "p3b": file_sha256(p3b_out),
            "pac": file_sha256(pac_out),
        }
        if p3b_erp_out is not None:
            output_file["p3b_erp"] = str(p3b_erp_out)
            output_hash["p3b_erp"] = file_sha256(p3b_erp_out)
        step_qc_path = save_step_qc(
            "06_epoch",
            subj,
            block_num,
            {
                "status": "PASS",
                "input_file": str(f),
                "input_hash": file_sha256(f),
                "output_file": output_file,
                "output_hash": output_hash,
                "parameters_used": {
                    "p3b": p3b_cfg,
                    "pac": pac_cfg_params,
                    "erp_branch": erp_branch_cfg if erp_branch_enabled else None,
                },
                "step_specific": {
                    "n_epochs": {
                        "p3b": int(n_p3b),
                        "p3b_erp": int(n_p3b_erp),
                        "pac": int(n_pac),
                    },
                    "practice_trim": practice_meta,
                    "practice_trim_erp": practice_meta_erp,
                    "event_id_map": event_id,
                    "event_id_hash": stable_json_hash(event_id),
                    "tmin": {"p3b": p3b_tmin, "pac": pac_tmin},
                    "tmax": {"p3b": p3b_tmax, "pac": pac_tmax},
                    "erp_epoch_type": erp_epoch_type if erp_branch_enabled else None,
                    "erp_event_id_map": event_id_erp if erp_branch_enabled else None,
                },
            },
        )
        logger.info("Saved step QC log: %s", step_qc_path)
        print(f"Saved epochs for {subj}.\n")


if __name__ == "__main__":
    main()
