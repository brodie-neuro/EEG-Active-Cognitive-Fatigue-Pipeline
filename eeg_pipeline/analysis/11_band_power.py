# eeg_pipeline/analysis/11_band_power.py
"""
Step 5 — Frontal Midline Theta (FMθ) Power

Extracts theta power (4-8 Hz) at the CF node during the maintenance
window. Log-transformed. Outputs long format: one row per subject × block.

This feeds into the integrated model (H4, Step 6) as the 'effort' marker.

Reference: post_processing_EEG_plan_v2.docx, Step 5
"""
import sys
from pathlib import Path
import mne
import numpy as np
import pandas as pd

pipeline_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pipeline_dir))

from src.utils_io import load_config
from src.utils_features import (
    load_block_epochs, get_subjects_with_blocks,
    available_channels, get_node_channels
)

OUTPUT_DIR = pipeline_dir / "outputs" / "features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THETA_BAND = (4, 8)


def compute_theta_power(epochs, ch_picks, fmin=4, fmax=8):
    """
    Compute mean theta power (log-transformed) at specified channels.

    Returns
    -------
    float : log10 theta power (µV²)
    """
    avail = available_channels(ch_picks, epochs.ch_names)
    if not avail:
        avail = None

    try:
        psd = epochs.compute_psd(fmin=fmin, fmax=fmax, picks=avail, verbose=False)
        power = psd.get_data().mean()
        power_uv2 = power * 1e12  # Convert to µV²
        return np.log10(power_uv2) if power_uv2 > 0 else np.nan
    except Exception as e:
        print(f"  Theta power failed: {e}")
        return np.nan


def main():
    cfg = load_config()
    blocks = cfg.get('blocks', [1, 5])
    epochs_dir = pipeline_dir / "outputs" / "derivatives" / "epochs_clean"

    cf_channels = get_node_channels('CF', cfg)
    if not cf_channels:
        cf_channels = ['Fz', 'FCz', 'Cz']

    if not epochs_dir.exists():
        print(f"Epochs directory not found: {epochs_dir}")
        return

    subjects = get_subjects_with_blocks(epochs_dir, 'pac', blocks)
    if not subjects:
        legacy = sorted(epochs_dir.glob("*_pac_clean-epo.fif"))
        subjects = sorted(set(f.stem.split("_")[0] for f in legacy))
    if not subjects:
        print("No PAC epoch files found.")
        return

    rows = []

    for subj in subjects:
        print(f"--- FMθ Power: {subj} ---")

        for block in blocks:
            epochs = load_block_epochs(subj, block, 'pac', epochs_dir)
            if epochs is None or 'eeg' not in epochs.get_channel_types():
                continue

            power = compute_theta_power(epochs, cf_channels)
            rows.append({
                'subject': subj,
                'block': block,
                'theta_power_log': power,
            })
            print(f"  Block {block}: log10(θ) = {power:.4f}")

    if rows:
        df = pd.DataFrame(rows)
        output_file = OUTPUT_DIR / "theta_power_features.csv"
        df.to_csv(output_file, index=False)
        print(f"\nSaved FMθ power features (long format) to {output_file}")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
