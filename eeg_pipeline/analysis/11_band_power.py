# eeg_pipeline/analysis/11_band_power.py
"""
Step 5 — Band Power Extraction (Theta + Alpha)

Extracts:
  - Frontal midline theta (FMθ) power at CF node (individualized ITF ± 2 Hz)
  - Posterior alpha power at CP/LP/RP nodes (individualized IAF ± 2 Hz)

Both log-transformed. Outputs long format: one row per subject × block.
Theta feeds into the integrated model as the 'effort' marker.
Alpha provides the individualized slowing analysis.

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
ALPHA_BAND = (8, 13)


def compute_band_power(epochs, ch_picks, fmin=4, fmax=8):
    """
    Compute mean band power (log-transformed) at specified channels.

    Returns
    -------
    float : log10 band power (µV²)
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
        print(f"  Band power failed: {e}")
        return np.nan


def load_individual_peaks(output_dir):
    """Load individual theta (ITF) and alpha (IAF) peak frequencies."""
    itf_map = {}
    feat_file = output_dir / "theta_freq_features.csv"
    if feat_file.exists():
        df = pd.read_csv(feat_file)
        itf_map = {(row['subject'], row['block']): row['f_theta']
                   for _, row in df.iterrows() if not np.isnan(row['f_theta'])}

    iaf_map = {}
    iaf_file = output_dir / "iaf_features.csv"
    if iaf_file.exists():
        df = pd.read_csv(iaf_file)
        iaf_map = {row['subject']: row['iaf']
                   for _, row in df.iterrows()
                   if not np.isnan(row['iaf']) and row['timepoint'] == 'pre'}

    return itf_map, iaf_map


def main():
    cfg = load_config()
    blocks = cfg.get('blocks', [1, 5])
    epochs_dir = pipeline_dir / "outputs" / "derivatives" / "epochs_clean"
    features_dir = pipeline_dir / "outputs" / "features"

    cf_channels = get_node_channels('CF', cfg)
    if not cf_channels:
        cf_channels = ['Fz', 'FCz', 'Cz']

    if not epochs_dir.exists():
        print(f"Epochs directory not found: {epochs_dir}")
        return

    # Load Individual Peak Frequencies (ITF + IAF)
    itf_map, iaf_map = load_individual_peaks(features_dir)
    default_band = (4, 8)
    
    if itf_map:
        print(f"Loaded {len(itf_map)} individual theta peaks for dynamic bands.")
    else:
        print("No individual theta peaks found. Using fixed 4-8 Hz for all.")
    if iaf_map:
        print(f"Loaded {len(iaf_map)} IAF values for theta upper-bound capping.")

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

            # Dynamic Band Definition — cap at IAF-1 Hz
            itf = itf_map.get((subj, block))
            if itf:
                iaf = iaf_map.get(subj, 10.0)  # Default IAF ~10 Hz
                upper_cap = iaf - 1.0
                low = max(1.5, itf - 2.0)
                high = min(upper_cap, itf + 2.0)
                band = (low, high)
                print(f"  Block {block}: band {low:.1f}-{high:.1f} Hz "
                      f"(Peak: {itf:.2f}, IAF cap: {upper_cap:.1f})")
            else:
                band = default_band
                print(f"  Block {block}: fixed band {band[0]}-{band[1]} Hz")

            power = compute_band_power(epochs, cf_channels, fmin=band[0], fmax=band[1])
            rows.append({
                'subject': subj,
                'block': block,
                'theta_power_log': power,
            })
            print(f"           log10(θ) = {power:.4f}")

    # Save theta power
    if rows:
        df = pd.DataFrame(rows)
        output_file = OUTPUT_DIR / "theta_power_features.csv"
        df.to_csv(output_file, index=False)
        print(f"\nSaved FMθ power features (long format) to {output_file}")
        print(df.to_string(index=False))

    # === Posterior Alpha Power (individualized IAF ± 2 Hz) ===
    print("\n" + "="*50)
    print("  Alpha Power Extraction")
    print("="*50)

    # Posterior channels for alpha
    posterior_nodes = ['CP', 'LP', 'RP']
    posterior_chs = []
    for node in posterior_nodes:
        posterior_chs.extend(get_node_channels(node, cfg) or [])
    if not posterior_chs:
        posterior_chs = ['Pz', 'POz', 'Oz', 'O1', 'O2']

    alpha_rows = []

    for subj in subjects:
        print(f"--- Alpha Power: {subj} ---")

        for block in blocks:
            epochs = load_block_epochs(subj, block, 'pac', epochs_dir)
            if epochs is None or 'eeg' not in epochs.get_channel_types():
                continue

            # Individualized alpha band: IAF ± 2 Hz
            iaf = iaf_map.get(subj)
            if iaf:
                # Cap lower bound at ITF + 1 Hz to avoid theta leakage
                itf = itf_map.get((subj, block))
                lower_cap = (itf + 1.0) if itf else 6.0
                alpha_low = max(lower_cap, iaf - 2.0)
                alpha_high = min(20.0, iaf + 2.0)
                alpha_band = (alpha_low, alpha_high)
                print(f"  Block {block}: alpha band {alpha_low:.1f}-{alpha_high:.1f} Hz "
                      f"(IAF: {iaf:.2f})")
            else:
                alpha_band = ALPHA_BAND
                print(f"  Block {block}: fixed alpha band {alpha_band[0]}-{alpha_band[1]} Hz")

            alpha_power = compute_band_power(
                epochs, posterior_chs, fmin=alpha_band[0], fmax=alpha_band[1]
            )
            alpha_rows.append({
                'subject': subj,
                'block': block,
                'alpha_power_log': alpha_power,
            })
            print(f"           log10(α) = {alpha_power:.4f}")

    if alpha_rows:
        df_alpha = pd.DataFrame(alpha_rows)
        output_alpha = OUTPUT_DIR / "alpha_power_features.csv"
        df_alpha.to_csv(output_alpha, index=False)
        print(f"\nSaved alpha power features (long format) to {output_alpha}")
        print(df_alpha.to_string(index=False))


if __name__ == "__main__":
    main()
