# postprocessing/14_preprocessing_qc_summary.py
"""
Preprocessing QC Summary

Collects all preprocessing quality metrics into a single report:
  - Channel interpolation (from average reference step)
  - ASR modification ratio
  - ICLabel component classifications (brain, eye, muscle, etc.)
  - Epoch counts before/after autoreject (P3b and stimulus epochs)
  - Overall rejection rates

Note: 'stimulus' epochs (-0.5 to 1.8s) are used for all oscillatory
analyses: PAC, wPLI, theta power, gamma, and peak frequency.
P3b epochs (-0.2 to 0.8s) are used only for the ERP analysis.

Outputs:
  - preprocessing_qc_summary.csv  (one row per subject x block)
  - Console summary table

This provides a single-glance view of data quality across all
preprocessing steps, rather than having QC scattered across
individual step reports.
"""
import sys
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(Path(__file__).resolve().parents[1]))

try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False

pipeline_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pipeline_dir))

from src.utils_io import load_config

OUTPUT_DIR = pipeline_dir / "outputs" / "features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_qc_json(subj, block, outputs_dir):
    """Load QC JSON data for a subject/block if available."""
    qc_path = outputs_dir / "qc" / subj / f"block{block}" / "qc_data.json"
    if qc_path.exists():
        with open(qc_path, 'r') as f:
            return json.load(f)
    return None


def _extract_step_metrics(qc_data, step_prefix):
    """Extract metrics dict from a QC report for a given step prefix."""
    if qc_data is None:
        return {}
    for step in qc_data.get('steps', []):
        if step.get('step', '').startswith(step_prefix):
            return step.get('metrics', {})
    return {}


def _count_epochs(subj, block, outputs_dir):
    """Count epochs before/after autoreject."""
    if not MNE_AVAILABLE:
        return {}

    results = {}
    for etype in ['p3b', 'pac']:  # pac epochs = stimulus epochs (PAC, wPLI, theta, gamma)
        # Try per-subject layout first, then legacy flat layout
        for base in [outputs_dir / subj, outputs_dir]:
            raw_file = base / "derivatives" / "epochs" / f"{subj}_block{block}_{etype}-epo.fif"
            clean_file = base / "derivatives" / "epochs_clean" / f"{subj}_block{block}_{etype}_clean-epo.fif"

            n_raw = 0
            n_clean = 0

            if raw_file.exists():
                try:
                    ep = mne.read_epochs(str(raw_file), preload=False, verbose=False)
                    n_raw = len(ep)
                except Exception:
                    pass

            if clean_file.exists():
                try:
                    ep = mne.read_epochs(str(clean_file), preload=False, verbose=False)
                    n_clean = len(ep)
                except Exception:
                    pass

            if n_raw > 0 or n_clean > 0:
                n_rejected = n_raw - n_clean
                rejection_pct = (n_rejected / n_raw * 100) if n_raw > 0 else 0
                # Rename 'pac' to 'stim' for clarity in output
                out_label = 'stim' if etype == 'pac' else etype
                results[f'{out_label}_epochs_raw'] = n_raw
                results[f'{out_label}_epochs_clean'] = n_clean
                results[f'{out_label}_rejected'] = n_rejected
                results[f'{out_label}_rejection_pct'] = round(rejection_pct, 1)
                break  # Found data, don't check legacy path

    return results


def main():
    cfg = load_config()
    blocks = cfg.get('blocks', [1, 5])
    outputs_dir = pipeline_dir / "outputs"

    # Discover subjects
    subjects = set()
    for sub_dir in sorted(outputs_dir.glob("sub-*")):
        if sub_dir.is_dir():
            subjects.add(sub_dir.name)

    # Also check derivatives for legacy flat layout
    for deriv_type in ["epochs_clean", "epochs", "asr_cleaned_raw"]:
        deriv_dir = outputs_dir / "derivatives" / deriv_type
        if deriv_dir.exists():
            for f in deriv_dir.glob("sub-*"):
                subj = f.name.split("_")[0]
                if subj.startswith("sub-"):
                    subjects.add(subj)

    subjects = sorted(subjects)
    if not subjects:
        print("No subjects found.")
        return

    print(f"Found {len(subjects)} subjects: {subjects}\n")

    all_rows = []

    for subj in subjects:
        for block in blocks:
            row = {'subject': subj, 'block': block}

            # --- Load QC report data ---
            qc_data = _load_qc_json(subj, block, outputs_dir)

            # --- Channel interpolation (step 02) ---
            ref_metrics = _extract_step_metrics(qc_data, '02_')
            row['n_channels_interpolated'] = ref_metrics.get('n_interpolated',
                                                              ref_metrics.get('n_channels_interpolated', np.nan))

            # --- ASR (step 04) ---
            asr_metrics = _extract_step_metrics(qc_data, '04_asr')
            row['asr_modification_pct'] = asr_metrics.get('modification_pct',
                                                           asr_metrics.get('asr_ratio', np.nan))

            # --- ICA/ICLabel (step 05/06) ---
            ica_metrics = _extract_step_metrics(qc_data, '06_ica')
            if not ica_metrics:
                ica_metrics = _extract_step_metrics(qc_data, '05_ica')
            row['n_ics_total'] = ica_metrics.get('n_components', np.nan)
            row['n_ics_excluded'] = ica_metrics.get('n_excluded',
                                                     ica_metrics.get('ICs rejected', np.nan))
            row['n_brain_ics'] = ica_metrics.get('n_brain', np.nan)
            row['n_eye_ics'] = ica_metrics.get('n_eye', np.nan)
            row['n_muscle_ics'] = ica_metrics.get('n_muscle', np.nan)

            # --- Epoch counts (from files directly) ---
            epoch_counts = _count_epochs(subj, block, outputs_dir)
            row.update(epoch_counts)

            # --- Autoreject metrics from QC report ---
            for etype in ['p3b', 'pac']:
                ar_metrics = _extract_step_metrics(qc_data, f'08_autoreject_{etype}')
                if not ar_metrics:
                    ar_metrics = _extract_step_metrics(qc_data, f'07_autoreject_{etype}')
                out_label = 'stim' if etype == 'pac' else etype
                if ar_metrics:
                    if f'{out_label}_rejection_pct' not in row or np.isnan(row.get(f'{out_label}_rejection_pct', np.nan)):
                        row[f'{out_label}_rejection_pct'] = ar_metrics.get('rejection_pct', np.nan)

            # --- Overall QC status ---
            if qc_data:
                statuses = [s.get('status', 'PASS') for s in qc_data.get('steps', [])]
                if any(s == 'FAIL' for s in statuses):
                    row['qc_overall'] = 'FAIL'
                elif any(s == 'WARNING' for s in statuses):
                    row['qc_overall'] = 'WARNING'
                else:
                    row['qc_overall'] = 'PASS'
            else:
                row['qc_overall'] = 'NO_DATA'

            all_rows.append(row)

            # Print summary
            interp = row.get('n_channels_interpolated', '?')
            asr = row.get('asr_modification_pct', '?')
            ics_ex = row.get('n_ics_excluded', '?')
            p3b_rej = row.get('p3b_rejection_pct', '?')
            stim_rej = row.get('stim_rejection_pct', '?')
            print(f"  {subj} B{block}: interp={interp} | ASR={asr}% | "
                  f"ICs_excl={ics_ex} | P3b_rej={p3b_rej}% | stim_rej={stim_rej}% | "
                  f"QC={row['qc_overall']}")

    if all_rows:
        df = pd.DataFrame(all_rows)

        # Save
        out_path = OUTPUT_DIR / "preprocessing_qc_summary.csv"
        df.to_csv(out_path, index=False)
        print(f"\nSaved QC summary to {out_path}")
        print(f"Shape: {df.shape}")

        # Print summary statistics
        print(f"\n{'='*60}")
        print("  PREPROCESSING QC SUMMARY")
        print(f"{'='*60}")

        for col in ['n_channels_interpolated', 'asr_modification_pct',
                     'n_ics_excluded', 'p3b_rejection_pct', 'stim_rejection_pct']:
            if col in df.columns:
                vals = df[col].dropna()
                if len(vals) > 0:
                    print(f"  {col}: mean={vals.mean():.1f}, "
                          f"range=[{vals.min():.1f}, {vals.max():.1f}]")

        # QC status counts
        if 'qc_overall' in df.columns:
            counts = df['qc_overall'].value_counts()
            print(f"\n  QC Status: {dict(counts)}")
    else:
        print("No data found.")


if __name__ == "__main__":
    main()
