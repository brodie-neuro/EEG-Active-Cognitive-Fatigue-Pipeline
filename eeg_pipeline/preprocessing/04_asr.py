# preprocessing/04_asr.py
"""
Step 04: Artifact Subspace Reconstruction (ASR) for transient artefact repair.
Cleans high-amplitude bursts (muscle twitches, head movements) without removing data.
"""
# ─── DETERMINISM: lock BLAS/LAPACK to 1 thread BEFORE numpy/scipy load ───
# OpenBLAS multi-threading causes non-deterministic floating-point summation
# order in eigendecomposition (scipy.linalg.eigh), which changes ASR
# calibration window selection and downstream cleaning.  Must be set before
# any import that triggers the shared library load.
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MNE_DONTWRITE_HOME"] = "true"
os.environ.setdefault("_MNE_FAKE_HOME_DIR", os.path.dirname(os.path.dirname(__file__)))
# ─────────────────────────────────────────────────────────────────────────

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from datetime import datetime
import mne
import numpy as np
import pandas as pd

pipeline_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pipeline_dir))
from src.utils_io import (
    load_config,
    save_clean_raw,
    subj_id_from_derivative,
    iter_derivative_files,
)
from src.utils_config import get_param
from src.utils_determinism import file_sha256, save_step_qc
from src.utils_report import QCReport, qc_psd_overlay
from src.utils_logging import setup_pipeline_logger

# ASR using asrpy
try:
    from asrpy import ASR
    ASR_AVAILABLE = True
except ImportError:
    raise ImportError("FAILED: asrpy not available. Install with: pip install asrpy")

# ─── (A) threadpoolctl as belt-and-suspenders safeguard ───
try:
    from threadpoolctl import threadpool_limits
    THREADPOOLCTL_AVAILABLE = True
except ImportError:
    threadpool_limits = None
    THREADPOOLCTL_AVAILABLE = False


# ─── (E) Robust block number parsing ───
_BLOCK_RE = re.compile(r'block(\d+)', re.IGNORECASE)


def _parse_block_num(filename_stem: str) -> int:
    """Extract block number from filename like 'sub-p003_block1_notch-raw'.

    Handles multi-digit subject IDs and multi-digit block numbers correctly.
    Falls back to 1 with a warning if parsing fails.
    """
    m = _BLOCK_RE.search(filename_stem)
    if m:
        return int(m.group(1))
    raise ValueError(f"Could not parse block number from '{filename_stem}'.")


# ─── (C) Diagnostic helpers ───
def _array_hash(arr: np.ndarray) -> str:
    """Deterministic SHA-256 hex digest of a numpy array."""
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def _matrix_hash(mat: np.ndarray) -> str:
    """Hash a mixing/threshold matrix."""
    if mat is None:
        return "NONE"
    return _array_hash(np.ascontiguousarray(mat))


# ─── (D) Multi-metric QC ───
def _compute_modification_metrics(data_before: np.ndarray,
                                   data_after: np.ndarray) -> dict:
    """Compute multiple modification metrics between pre- and post-ASR data.

    Returns dict with physically meaningful metrics, not just the fragile 1e-15.
    """
    diff = np.abs(data_before - data_after)

    # Element-wise metrics
    modified_1e15 = 100.0 * np.mean(diff > 1e-15)
    modified_1e9 = 100.0 * np.mean(diff > 1e-9)

    # Sample-wise: % of time samples where ANY channel changed > 1e-9 V
    modified_sample_any = 100.0 * np.mean(np.any(diff > 1e-9, axis=0))

    # Physical metrics in microvolts
    diff_uv = diff * 1e6
    mean_abs_uv = float(np.mean(diff_uv))
    rms_uv = float(np.sqrt(np.mean(diff_uv ** 2)))
    max_abs_uv = float(np.max(diff_uv))

    return {
        'modified_pct_element_1e15': round(modified_1e15, 4),
        'modified_pct_element_1e9': round(modified_1e9, 4),
        'modified_pct_sample_any_1e9': round(modified_sample_any, 4),
        'mean_abs_change_uv': round(mean_abs_uv, 6),
        'rms_change_uv': round(rms_uv, 6),
        'max_abs_change_uv': round(max_abs_uv, 4),
    }


def _run_asr_once(raw, eeg_picks, sfreq, asr_params, use_clean_windows):
    """Run a single ASR fit+transform cycle and return diagnostics.

    Returns (raw_clean, diagnostics_dict).
    """
    # ─── (B) Explicit ASR parameters from config ───
    asr_cutoff = asr_params.get('cutoff', 20)
    blocksize = asr_params.get('blocksize', 100)
    win_len = asr_params.get('win_len', 0.5)
    win_overlap = asr_params.get('win_overlap', 0.66)
    max_dropout_fraction = asr_params.get('max_dropout_fraction', 0.1)
    min_clean_fraction = asr_params.get('min_clean_fraction', 0.25)
    max_bad_chans = asr_params.get('max_bad_chans', 0.1)
    method = asr_params.get('method', 'euclid')

    # Transform parameters
    lookahead = asr_params.get('lookahead', 0.25)
    stepsize = asr_params.get('stepsize', 32)
    maxdims = asr_params.get('maxdims', 0.66)
    mem_splits = asr_params.get('mem_splits', 3)

    # Build ASR with explicit parameters
    asr_kwargs = dict(
        sfreq=sfreq,
        cutoff=asr_cutoff,
        blocksize=blocksize,
        win_len=win_len,
        win_overlap=win_overlap,
        max_dropout_fraction=max_dropout_fraction,
        min_clean_fraction=min_clean_fraction,
        max_bad_chans=max_bad_chans,
        method=method,
    )

    asr = ASR(**asr_kwargs)

    config_used = {
        'cutoff': asr_cutoff,
        'blocksize': blocksize,
        'win_len': win_len,
        'win_overlap': win_overlap,
        'max_dropout_fraction': max_dropout_fraction,
        'min_clean_fraction': min_clean_fraction,
        'max_bad_chans': max_bad_chans,
        'method': method,
        'use_clean_windows': use_clean_windows,
        'lookahead': lookahead,
        'stepsize': stepsize,
        'maxdims': maxdims,
        'mem_splits': mem_splits,
    }

    # ─── FIT with diagnostics ───
    diag = {'config': config_used}

    if use_clean_windows:
        clean_data, sample_mask = asr.fit(
            raw, picks=eeg_picks, return_clean_window=True
        )
        diag['sample_mask_n_kept'] = int(np.sum(sample_mask))
        diag['sample_mask_n_total'] = int(sample_mask.size)
        diag['sample_mask_pct_kept'] = round(
            100.0 * np.sum(sample_mask) / sample_mask.size, 4
        )
        diag['sample_mask_hash'] = _array_hash(sample_mask.astype(np.uint8))
        diag['calib_data_shape'] = list(clean_data.shape)
        diag['calib_data_hash'] = _array_hash(clean_data)
    else:
        calib_samples = min(int(60 * sfreq), raw.n_times // 2)
        asr.fit(raw, picks=eeg_picks, start=0, stop=calib_samples)
        diag['calib_method'] = 'first_60s'
        diag['calib_samples'] = calib_samples

    # Capture M and T matrix hashes
    diag['M_hash'] = _matrix_hash(asr.M)
    diag['T_hash'] = _matrix_hash(asr.T)

    # ─── TRANSFORM ───
    data_before = raw.get_data(picks=eeg_picks)

    raw_clean = asr.transform(
        raw, picks=eeg_picks,
        lookahead=lookahead,
        stepsize=stepsize,
        maxdims=maxdims,
        mem_splits=mem_splits,
    )

    if isinstance(raw_clean, tuple):
        raw_clean = raw_clean[0]

    data_after = raw_clean.get_data(picks=eeg_picks)

    # Compute all QC metrics
    metrics = _compute_modification_metrics(data_before, data_after)
    diag['metrics'] = metrics
    diag['cleaned_data_hash'] = _array_hash(data_after)
    diag['n_channels'] = int(len(eeg_picks))
    diag['n_samples'] = int(data_before.shape[1])
    diag['sfreq'] = float(sfreq)

    return raw_clean, data_before, data_after, metrics, diag


def main():
    parser = argparse.ArgumentParser(description="Step 04: ASR cleaning")
    parser.add_argument(
        "--subject",
        type=str,
        default="",
        help="Optional subject filter (e.g. sub-dario or sub-a,sub-b).",
    )
    # ─── (F) Reproducibility test mode ───
    parser.add_argument(
        "--repeat",
        type=int,
        default=0,
        help="Run ASR N times on each block and report reproducibility diagnostics.",
    )
    args = parser.parse_args()

    cfg = load_config()
    logger = setup_pipeline_logger('04_asr')

    pipeline_root = Path(__file__).resolve().parents[1]
    output_dir = pipeline_root / "outputs" / "derivatives" / "asr_cleaned_raw"
    qc_dir = pipeline_root / "outputs" / "qc_figs" / "asr"
    debug_dir = pipeline_root / "outputs" / "features" / "asr_debug"

    output_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    files = iter_derivative_files("notch_raw", "*_notch-raw.fif", subject=args.subject)
    if not files:
        print("No files found to process.")
        return

    asr_params = get_param('asr', default={}) or {}
    asr_enabled = bool(asr_params.get('enabled', True))

    if not THREADPOOLCTL_AVAILABLE:
        print("  NOTE: threadpoolctl not installed. BLAS thread limits rely on env vars only.")

    for f in files:
        subj = subj_id_from_derivative(f)
        block_num = _parse_block_num(f.stem)
        print(f"--- Processing {subj} block {block_num} (ASR) ---")

        raw = mne.io.read_raw_fif(f, preload=True)
        qc = QCReport(subj, block_num)

        if not asr_enabled:
            print("ASR disabled in parameters; copying input to output unchanged.")
            qc.log_step('04_asr', status='SKIPPED',
                        metrics={'enabled': False},
                        notes='ASR disabled via parameters.json',
                        params_used=asr_params)
            qc.save_report()
            out_file = save_clean_raw(raw, output_dir, subj, "asr")
            save_step_qc(
                "04_asr",
                subj,
                block_num,
                {
                    "status": "SKIPPED",
                    "input_file": str(f),
                    "input_hash": file_sha256(f),
                    "output_file": str(out_file),
                    "output_hash": file_sha256(out_file),
                    "parameters_used": asr_params,
                    "step_specific": {
                        "enabled": False,
                        "skip_reason": "disabled_in_config",
                    },
                },
            )
            continue

        # Get EEG data only
        eeg_picks = mne.pick_types(raw.info, eeg=True)

        if len(eeg_picks) == 0:
            raise ValueError(f"No EEG channels found in {f.name}; cannot run ASR.")

        data = raw.get_data(picks=eeg_picks)
        sfreq = raw.info['sfreq']

        # Check for flat/NaN channels
        variances = np.var(data, axis=1)
        if np.any(variances < 1e-15) or np.any(np.isnan(data)):
            raise ValueError(
                f"Data has flat or NaN EEG channels for {subj} block {block_num}; "
                "aborting ASR."
            )

        use_clean_windows = bool(asr_params.get('use_clean_windows', True))
        asr_cutoff = asr_params.get('cutoff', 20)
        print(f"Fitting ASR (cutoff={asr_cutoff}, clean_windows={use_clean_windows})...")

        # ─── (A) Wrap in threadpoolctl context ───
        def _do_asr():
            return _run_asr_once(raw, eeg_picks, sfreq, asr_params, use_clean_windows)

        if THREADPOOLCTL_AVAILABLE:
            with threadpool_limits(limits=1, user_api='blas'):
                with threadpool_limits(limits=1, user_api='openmp'):
                    raw_clean, data_before, data_after, metrics, diag = _do_asr()
        else:
            raw_clean, data_before, data_after, metrics, diag = _do_asr()

        # ─── (F) Reproducibility test mode ───
        if args.repeat > 0:
            print(f"\n  ═══ REPRODUCIBILITY TEST: {args.repeat} additional runs ═══")
            all_diags = [diag]

            for rep in range(args.repeat):
                print(f"  Run {rep + 2}/{args.repeat + 1}...")
                if THREADPOOLCTL_AVAILABLE:
                    with threadpool_limits(limits=1, user_api='blas'):
                        with threadpool_limits(limits=1, user_api='openmp'):
                            _, _, _, rep_metrics, rep_diag = _run_asr_once(
                                raw, eeg_picks, sfreq, asr_params, use_clean_windows
                            )
                else:
                    _, _, _, rep_metrics, rep_diag = _run_asr_once(
                        raw, eeg_picks, sfreq, asr_params, use_clean_windows
                    )
                all_diags.append(rep_diag)

            # ─── Reproducibility summary ───
            print(f"\n  ─── Reproducibility Summary ({len(all_diags)} runs) ───")

            mask_hashes = [d.get('sample_mask_hash', 'N/A') for d in all_diags]
            m_hashes = [d.get('M_hash', 'N/A') for d in all_diags]
            t_hashes = [d.get('T_hash', 'N/A') for d in all_diags]
            data_hashes = [d.get('cleaned_data_hash', 'N/A') for d in all_diags]

            mask_ok = len(set(mask_hashes)) == 1
            m_ok = len(set(m_hashes)) == 1
            t_ok = len(set(t_hashes)) == 1
            data_ok = len(set(data_hashes)) == 1

            print(f"  sample_mask identical: {'✅ YES' if mask_ok else '❌ NO'} ({set(mask_hashes)})")
            print(f"  M matrix identical:    {'✅ YES' if m_ok else '❌ NO'} ({set(m_hashes)})")
            print(f"  T matrix identical:    {'✅ YES' if t_ok else '❌ NO'} ({set(t_hashes)})")
            print(f"  cleaned data identical:{'✅ YES' if data_ok else '❌ NO'} ({set(data_hashes)})")

            # Metric ranges
            for metric_key in ['modified_pct_element_1e15', 'modified_pct_element_1e9',
                               'mean_abs_change_uv', 'rms_change_uv']:
                values = [d['metrics'].get(metric_key, 0) for d in all_diags]
                print(f"  {metric_key}: range [{min(values):.4f}, {max(values):.4f}]")

            # Save repeat diagnostics
            repeat_path = debug_dir / f"{subj}_block{block_num}_repeat_test.json"
            with open(repeat_path, 'w') as fp:
                json.dump({
                    'subject': subj,
                    'block': block_num,
                    'n_runs': len(all_diags),
                    'mask_identical': mask_ok,
                    'M_identical': m_ok,
                    'T_identical': t_ok,
                    'data_identical': data_ok,
                    'runs': all_diags,
                }, fp, indent=2, default=str)
            print(f"  Saved to {repeat_path}")
            print(f"  ═══════════════════════════════════════════\n")

        # ─── Print primary metrics ───
        modified_pct = metrics['modified_pct_element_1e15']
        modified_pct_1e9 = metrics['modified_pct_element_1e9']
        print(f"  Data modified (>1e-15): {modified_pct:.2f}%")
        print(f"  Data modified (>1e-9):  {modified_pct_1e9:.2f}%")
        print(f"  Mean abs change:        {metrics['mean_abs_change_uv']:.4f} µV")
        print(f"  RMS change:             {metrics['rms_change_uv']:.4f} µV")
        print(f"  Max abs change:         {metrics['max_abs_change_uv']:.2f} µV")
        print(f"  M hash: {diag['M_hash']}  T hash: {diag['T_hash']}")
        if 'sample_mask_pct_kept' in diag:
            print(f"  Calibration windows: {diag['sample_mask_pct_kept']:.1f}% kept")

        # ─── Save per-subject diagnostic JSON ───
        diag_path = debug_dir / f"{subj}_block{block_num}_asr_diag.json"
        diag['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        diag['subject'] = subj
        diag['block'] = block_num
        try:
            with open(diag_path, 'w') as fp:
                json.dump(diag, fp, indent=2, default=str)
        except Exception as e:
            print(f"  WARNING: Could not write diagnostic JSON: {e}")

        # ─── Persistent ASR log (append, never overwrite) ───
        asr_log_path = pipeline_root / "outputs" / "features" / "asr_modification_log.csv"
        log_row = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'subject': subj,
            'block': block_num,
            'modified_pct': round(modified_pct, 2),
            'modified_pct_1e9': round(modified_pct_1e9, 2),
            'mean_abs_change_uv': round(metrics['mean_abs_change_uv'], 4),
            'rms_change_uv': round(metrics['rms_change_uv'], 4),
            'max_abs_change_uv': round(metrics['max_abs_change_uv'], 2),
            'cutoff': asr_params.get('cutoff', 20),
            'clean_windows': use_clean_windows,
            'n_eeg_channels': len(eeg_picks),
            'n_samples': data_before.shape[1],
            'M_hash': diag['M_hash'],
            'T_hash': diag['T_hash'],
            'status': 'PASS' if modified_pct_1e9 <= asr_params.get('max_modification_pct', 30.0) else 'EXCEED_THRESHOLD',
        }
        try:
            if asr_log_path.exists():
                df_log = pd.read_csv(asr_log_path)
                df_log = pd.concat([df_log, pd.DataFrame([log_row])], ignore_index=True)
            else:
                df_log = pd.DataFrame([log_row])
            df_log.to_csv(asr_log_path, index=False)
            print(f"  ASR log appended to {asr_log_path}")
        except Exception as e:
            print(f"  WARNING: Could not write ASR log: {e}")

        if modified_pct_1e9 == 0.0:
            raise RuntimeError(
                "ASR produced EXACTLY 0% modifications at 1e-9 threshold "
                "(pass-through error). Aborting to prevent silent failure."
            )

        status = qc.assess_metric('ASR modified %', modified_pct_1e9,
                                  'max_asr_modified_pct', '<=')

        # Save QC plot: PSD before/after
        try:
            fig = qc_psd_overlay(raw, raw_clean, f"{subj} - ASR PSD (Before/After)")
            qc.add_figure('04_asr_psd', fig)
        except Exception as e:
            print(f"Could not generate PSD QC plot: {e}")

        # Save QC plot: variance comparison
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 1, figsize=(12, 6))

            ch_idx = raw.ch_names.index('Fz') if 'Fz' in raw.ch_names else 0
            t = np.arange(data.shape[1]) / sfreq

            axes[0].plot(t[:int(10*sfreq)], data[ch_idx, :int(10*sfreq)] * 1e6,
                         'b', alpha=0.7, label='Before')
            axes[0].plot(t[:int(10*sfreq)], data_after[ch_idx, :int(10*sfreq)] * 1e6,
                         'r', alpha=0.7, label='After')
            axes[0].set_ylabel('µV')
            axes[0].set_title(f'{subj} - First 10 seconds (Fz)')
            axes[0].legend()

            var_before = np.var(data, axis=1) * 1e12
            var_after = np.var(data_after, axis=1) * 1e12
            axes[1].bar(range(len(var_before)), var_before, alpha=0.5, label='Before')
            axes[1].bar(range(len(var_after)), var_after, alpha=0.5, label='After')
            axes[1].set_ylabel('Variance (µV²)')
            axes[1].set_xlabel('Channel')
            axes[1].set_title('Channel variance before/after ASR')
            axes[1].legend()

            plt.tight_layout()
            qc.add_figure('04_asr_variance', fig)
        except Exception as e:
            print(f"Could not generate variance QC plot: {e}")

        qc.log_step('04_asr', status=status,
                     metrics={
                         'cutoff': asr_params.get('cutoff', 20),
                         'modified_pct_1e15': round(modified_pct, 1),
                         'modified_pct_1e9': round(modified_pct_1e9, 1),
                         'mean_abs_change_uv': round(metrics['mean_abs_change_uv'], 4),
                     },
                     params_used=asr_params)
        qc.save_report()

        out_file = save_clean_raw(raw_clean, output_dir, subj, "asr")
        step_qc_path = save_step_qc(
            "04_asr",
            subj,
            block_num,
            {
                "status": status,
                "input_file": str(f),
                "input_hash": file_sha256(f),
                "output_file": str(out_file),
                "output_hash": file_sha256(out_file),
                "parameters_used": asr_params,
                "step_specific": {
                    "M_hash": diag["M_hash"],
                    "T_hash": diag["T_hash"],
                    "sample_mask_hash": diag.get("sample_mask_hash", "UNAVAILABLE"),
                    "sample_mask_pct_kept": diag.get("sample_mask_pct_kept"),
                    "calib_data_hash": diag.get("calib_data_hash", "UNAVAILABLE"),
                    "cleaned_data_hash": diag["cleaned_data_hash"],
                    "modification_pct": metrics["modified_pct_element_1e9"],
                    "cutoff": asr_params.get("cutoff", 20),
                    "use_clean_windows": bool(use_clean_windows),
                },
            },
        )
        logger.info("Saved step QC log: %s", step_qc_path)
        print(f"Saved {subj} ASR cleaned data.\n")


if __name__ == "__main__":
    main()
