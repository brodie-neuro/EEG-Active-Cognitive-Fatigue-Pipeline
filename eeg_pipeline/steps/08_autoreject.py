# steps/08_autoreject.py
"""
Step 08: Autoreject for final epoch-level quality control.
Automatically finds optimal rejection thresholds and repairs bad epochs.
"""
import sys
from pathlib import Path
import mne

pipeline_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pipeline_dir))
from src.utils_io import load_config, subj_id_from_derivative
from src.utils_config import get_param
from src.utils_report import QCReport, qc_epoch_summary

try:
    from autoreject import AutoReject, get_rejection_threshold
    AUTOREJECT_AVAILABLE = True
except ImportError:
    AUTOREJECT_AVAILABLE = False
    print("Warning: autoreject not available. Install with: pip install autoreject")


def main():
    cfg = load_config()
    
    pipeline_root = Path(__file__).resolve().parents[1]
    input_dir = pipeline_root / "outputs" / "derivatives" / "epochs"
    output_dir = pipeline_root / "outputs" / "derivatives" / "epochs_clean"
    qc_dir = pipeline_root / "outputs" / "qc_figs" / "autoreject"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}. Run Step 07 first.")
    
    # Process both P3b and PAC epoch files
    for epoch_type in ['p3b', 'pac']:
        files = sorted(list(input_dir.glob(f"*_{epoch_type}-epo.fif")))
        
        for f in files:
            subj = subj_id_from_derivative(f)
            print(f"--- Processing {subj} ({epoch_type} epochs, Autoreject) ---")
            
            epochs = mne.read_epochs(f, preload=True)
            n_before = len(epochs)
            
            # Synthetic data bypass -- autoreject can hang or over-reject
            if 'TEST' in subj.upper():
                print("Synthetic data -- skipping autoreject, saving unchanged.")
                epochs.save(output_dir / f"{subj}_{epoch_type}_clean-epo.fif", overwrite=True)
                continue
            
            if not AUTOREJECT_AVAILABLE:
                print("Autoreject not available - saving epochs unchanged.")
                epochs.save(output_dir / f"{subj}_{epoch_type}_clean-epo.fif", overwrite=True)
                continue
            
            if n_before < 10:
                print(f"Only {n_before} epochs - skipping autoreject.")
                epochs.save(output_dir / f"{subj}_{epoch_type}_clean-epo.fif", overwrite=True)
                continue
            
            # Check if we have EEG channels
            if 'eeg' not in epochs.get_channel_types():
                print("No EEG channels found - saving epochs unchanged.")
                epochs.save(output_dir / f"{subj}_{epoch_type}_clean-epo.fif", overwrite=True)
                continue
            
            # Run Autoreject with config parameters
            ar_params = get_param('autoreject')
            print(f"Running Autoreject on {n_before} epochs...")
            ar = AutoReject(
                n_interpolate=ar_params.get('n_interpolate', [1, 4, 8, 16]),
                consensus=ar_params.get('consensus', [0.1, 0.5, 1.0]),
                cv=ar_params.get('cv', 5),
                random_state=ar_params.get('random_state', 42),
                n_jobs=-1, verbose=False
            )
            epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)
            
            n_after = len(epochs_clean)
            n_rejected = n_before - n_after
            print(f"  Rejected {n_rejected}/{n_before} epochs ({100*n_rejected/n_before:.1f}%)")
            
            # Save QC plot
            try:
                fig = reject_log.plot(orientation='horizontal', show=False)
                fig.savefig(qc_dir / f"{subj}_{epoch_type}_reject_log.png", dpi=100)
                print(f"  Saved reject log to {qc_dir}")
            except Exception as e:
                print(f"  Could not save reject log: {e}")
            
            epochs_clean.save(output_dir / f"{subj}_{epoch_type}_clean-epo.fif", overwrite=True)
            
            # QC report
            block_str = ''.join(c for c in f.stem if c.isdigit())
            block_num = int(block_str[-1]) if block_str else 1
            qc = QCReport(subj, block_num)
            
            reject_pct = 100 * n_rejected / n_before if n_before > 0 else 0
            status = qc.assess_metric('Epoch rejection', reject_pct,
                                      'max_epoch_rejection_pct', '<=')
            
            # Save epoch QC figure
            try:
                fig, metrics = qc_epoch_summary(
                    epochs_clean,
                    title=f"{subj} {epoch_type} ({n_after}/{n_before} kept)"
                )
                qc.add_figure(f'08_autoreject_{epoch_type}', fig)
            except Exception as e:
                print(f"  Could not save epoch QC figure: {e}")
                metrics = {}
            
            qc.log_step(f'08_autoreject_{epoch_type}', status=status,
                         metrics={
                             'n_before': n_before,
                             'n_after': n_after,
                             'n_rejected': n_rejected,
                             'rejection_pct': round(reject_pct, 1),
                             **metrics,
                         },
                         params_used=ar_params)
            qc.save_report()
            print(f"  Saved clean epochs for {subj}.\n")


if __name__ == "__main__":
    main()
