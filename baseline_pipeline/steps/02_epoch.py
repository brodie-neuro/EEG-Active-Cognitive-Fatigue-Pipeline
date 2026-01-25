# baseline_pipeline/steps/02_epoch.py
"""
Baseline epoching: Simple amplitude-based rejection.
No ASR, no Autoreject - just peak-to-peak threshold.
"""
from pathlib import Path
import mne
import numpy as np

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PIPELINE_ROOT / "outputs" / "preprocessed"
OUTPUT_DIR = PIPELINE_ROOT / "outputs" / "epochs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    files = sorted(list(INPUT_DIR.glob("*_baseline-raw.fif")))
    if not files:
        print(f"No preprocessed files found in {INPUT_DIR}")
        return
    
    # Traditional rejection thresholds (fairly standard)
    reject = dict(eeg=150e-6)  # 150 µV peak-to-peak
    
    for f in files:
        subj = f.stem.split("_")[0]
        print(f"--- Baseline epoching: {subj} ---")
        
        raw = mne.io.read_raw_fif(f, preload=True, verbose=False)
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        
        if len(events) == 0:
            print(f"No events found for {subj}")
            continue
        
        # Stimulus-locked epochs (P3b equivalent)
        stim_events = {k: v for k, v in event_id.items() if 'stim' in k.lower()}
        if stim_events:
            epochs_stim = mne.Epochs(
                raw, events, stim_events,
                tmin=-0.2, tmax=0.8,
                baseline=(-0.2, 0),
                reject=reject,
                preload=True,
                verbose=False
            )
            n_rejected = len(events) - len(epochs_stim)
            print(f"Stimulus epochs: {len(epochs_stim)} kept, {n_rejected} rejected")
            epochs_stim.save(OUTPUT_DIR / f"{subj}_stim-epo.fif", overwrite=True)
        
        # Response-locked epochs (PAC equivalent)
        resp_events = {k: v for k, v in event_id.items() if 'resp' in k.lower() and 'miss' not in k.lower()}
        if resp_events:
            epochs_resp = mne.Epochs(
                raw, events, resp_events,
                tmin=0.2, tmax=1.2,
                baseline=None,
                reject=reject,
                preload=True,
                verbose=False
            )
            print(f"Response epochs: {len(epochs_resp)} kept")
            epochs_resp.save(OUTPUT_DIR / f"{subj}_resp-epo.fif", overwrite=True)


if __name__ == "__main__":
    main()
