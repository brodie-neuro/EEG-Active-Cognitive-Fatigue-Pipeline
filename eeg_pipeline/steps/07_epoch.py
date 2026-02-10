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
        
        # P3b Epochs: Stimulus-locked, -200 to +800 ms
        print("Creating P3b epochs (stimulus-locked)...")
        stim_events = {k: v for k, v in event_id.items() if 'stim' in k.lower()}
        if stim_events:
            epochs_p3b = mne.Epochs(
                raw, events, stim_events,
                tmin=-0.2, tmax=0.8,
                baseline=(-0.2, 0),
                preload=True,
                reject=None,  # Autoreject handles this in Step 08
                verbose=False
            )
            epochs_p3b.save(output_dir / f"{subj}_p3b-epo.fif", overwrite=True)
            print(f"  P3b epochs: {len(epochs_p3b)} trials")
        else:
            print("  No stimulus events found for P3b epochs.")
        
        # PAC Epochs: Stimulus-locked, maintenance window
        # Config defines pac.tmin/tmax relative to stimulus offset,
        # but we use stimulus-onset locking + offset to capture the
        # maintenance/retention period where theta-gamma PAC occurs.
        pac_cfg = cfg.get('epochs', {}).get('pac', {})
        pac_tmin = pac_cfg.get('tmin', 0.0) + 0.8  # 0.8s = approx stimulus duration
        pac_tmax = pac_cfg.get('tmax', 1.0) + 0.8
        print(f"Creating PAC epochs (stimulus-locked, {pac_tmin}-{pac_tmax}s)...")
        if stim_events:
            epochs_pac = mne.Epochs(
                raw, events, stim_events,
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
        
        print(f"Saved epochs for {subj}.\n")


if __name__ == "__main__":
    main()
