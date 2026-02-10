# steps/07_epoch.py
"""
Step 07: Epoch creation for P3b and PAC analysis.
Creates two separate epoch sets with different time-locking.
"""
import sys
from pathlib import Path
import mne
import numpy as np

pipeline_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pipeline_dir))
from src.utils_io import load_config, subj_id_from_derivative
from src.utils_config import get_param
from src.utils_report import QCReport


def main():
    cfg = load_config()
    
    pipeline_root = Path(__file__).resolve().parents[1]
    input_dir = pipeline_root / "outputs" / "derivatives" / "asr_cleaned_raw"
    output_dir = pipeline_root / "outputs" / "derivatives" / "epochs"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}. Run Step 06 first.")
        
    files = sorted(list(input_dir.glob("*_asr-raw.fif")))
    if not files:
        print("No files found to process.")
        return
    
    for f in files:
        subj = subj_id_from_derivative(f)
        print(f"--- Processing {subj} (Epoching) ---")
        
        raw = mne.io.read_raw_fif(f, preload=True)
        events, event_id = mne.events_from_annotations(raw)
        
        if len(events) == 0:
            print(f"No events found for {subj} - skipping.")
            continue
        
        # Extract block number for QC
        block_str = ''.join(c for c in f.stem if c.isdigit())
        block_num = int(block_str[-1]) if block_str else 1
        qc = QCReport(subj, block_num)
        
        # Load epoch parameters from config
        p3b_cfg = get_param('epoching', 'p3b', default={})
        pac_cfg_params = get_param('epoching', 'pac', default={})
        
        # P3b Epochs: Stimulus ONSET locked only (Trigger 1), -200 to +800 ms
        # Exclude stim/offset events (Trigger 2) -- those mark end of stimulus
        print("Creating P3b epochs (stimulus-onset-locked)...")
        stim_events = {k: v for k, v in event_id.items()
                       if 'stim' in k.lower() and 'offset' not in k.lower()}
        if stim_events:
            p3b_tmin = p3b_cfg.get('tmin', -0.2)
            p3b_tmax = p3b_cfg.get('tmax', 0.8)
            p3b_baseline = tuple(p3b_cfg.get('baseline', [-0.2, 0.0]))
            epochs_p3b = mne.Epochs(
                raw, events, stim_events,
                tmin=p3b_tmin, tmax=p3b_tmax,
                baseline=p3b_baseline,
                preload=True,
                reject=None,  # Autoreject handles this in Step 08
                verbose=False
            )
            epochs_p3b.save(output_dir / f"{subj}_p3b-epo.fif", overwrite=True)
            print(f"  P3b epochs: {len(epochs_p3b)} trials")
        else:
            print("  No stimulus events found for P3b epochs.")
        
        # PAC Epochs: Stimulus-onset-locked, shifted into maintenance window
        # Only onset events (same as P3b), then offset by stimulus duration
        pac_offset = pac_cfg_params.get('stim_duration_offset', 0.8)
        pac_tmin = pac_cfg_params.get('tmin', 0.0) + pac_offset
        pac_tmax = pac_cfg_params.get('tmax', 1.0) + pac_offset
        print(f"Creating PAC epochs (stimulus-onset-locked, {pac_tmin}-{pac_tmax}s)...")
        onset_events = {k: v for k, v in event_id.items()
                        if 'stim' in k.lower() and 'offset' not in k.lower()}
        if onset_events:
            epochs_pac = mne.Epochs(
                raw, events, onset_events,
                tmin=pac_tmin, tmax=pac_tmax,
                baseline=None,  # No baseline for PAC
                preload=True,
                reject=None,
                verbose=False
            )
            epochs_pac.save(output_dir / f"{subj}_pac-epo.fif", overwrite=True)
            print(f"  PAC epochs: {len(epochs_pac)} trials")
        else:
            print("  No stimulus events found for PAC epochs.")
        
        # QC report
        n_p3b = len(epochs_p3b) if 'epochs_p3b' in dir() and epochs_p3b is not None else 0
        n_pac = len(epochs_pac) if 'epochs_pac' in dir() and epochs_pac is not None else 0
        qc.log_step('07_epoch', status='PASS',
                     metrics={
                         'n_events_total': len(events),
                         'n_onset_events': len(stim_events),
                         'n_p3b_epochs': n_p3b,
                         'n_pac_epochs': n_pac,
                     })
        qc.save_report()
        print(f"Saved epochs for {subj}.\n")


if __name__ == "__main__":
    main()
