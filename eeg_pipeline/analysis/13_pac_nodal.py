# eeg_pipeline/analysis/13_pac_nodal.py
"""
Step 4 — Phase-Amplitude Coupling Analysis (H1, H3)

4a (H1): Between-region PAC — RF theta phase × RP gamma amplitude
4b (H3): Local PAC — within-node, collapsed to 3 regions (F, C, P)
4c: Descriptive 9-node ΔPAC heatmap data

Uses tensorpac for PAC computation with surrogate z-scoring.
Modulation Index (Tort et al., 2010) as the core metric.
Outputs long format: one row per subject × block.

Reference: post_processing_EEG_plan_v2.docx, Step 4
"""
import sys
import copy
from pathlib import Path
import mne
import numpy as np
import pandas as pd
from scipy.stats import trim_mean

pipeline_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pipeline_dir))

from src.utils_io import load_config
from src.utils_features import (
    load_block_epochs, get_subjects_with_blocks,
    available_channels, get_node_channels, get_region_nodes
)

try:
    from tensorpac import Pac
    TENSORPAC_AVAILABLE = True
except ImportError:
    TENSORPAC_AVAILABLE = False
    print("Warning: tensorpac not installed. Install with: pip install tensorpac")

OUTPUT_DIR = pipeline_dir / "outputs" / "features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_node_signal(epochs, node_channels):
    """Extract mean signal across node channels. Returns (n_epochs, n_times) or None."""
    avail = available_channels(node_channels, epochs.ch_names)
    if not avail:
        return None
    data = epochs.copy().pick(avail).get_data()
    return data.mean(axis=1)


def compute_pac_tensorpac(phase_signal, amp_signal, sfreq, cfg):
    """Compute PAC using tensorpac with MI + surrogate z-scoring."""
    pac_cfg = cfg.get('pac', {})
    f_pha = pac_cfg.get('phase_band', [4, 8])
    f_amp = pac_cfg.get('amp_band', [55, 85])
    n_surr = pac_cfg.get('surrogates', 200)

    # idpac=(2, 1, 1): MI (Tort), swap phase/amplitude, z-score
    p = Pac(idpac=(2, 1, 1), f_pha=f_pha, f_amp=f_amp,
            dcomplex='hilbert', verbose=False)

    max_epochs = min(phase_signal.shape[0], 30)
    pha_data = phase_signal[:max_epochs, :]
    amp_data = amp_signal[:max_epochs, :]

    try:
        pac_vals = p.filterfit(sfreq, pha_data, amp_data, n_perm=n_surr)
        return float(np.nanmean(pac_vals))
    except Exception as e:
        print(f"  tensorpac failed: {e}")
        return np.nan


def compute_pac_fallback(phase_signal, amp_signal, sfreq, cfg):
    """Fallback PAC: custom MI + surrogates if tensorpac unavailable."""
    from scipy.signal import hilbert as sp_hilbert

    pac_cfg = cfg.get('pac', {})
    f_pha = pac_cfg.get('phase_band', [4, 8])
    f_amp = pac_cfg.get('amp_band', [55, 85])
    n_surr = pac_cfg.get('surrogates', 200)

    max_epochs = min(phase_signal.shape[0], 20)
    epoch_pacs = []

    for ep in range(max_epochs):
        try:
            theta = mne.filter.filter_data(
                phase_signal[ep], sfreq, f_pha[0], f_pha[1], verbose=False)
            theta_phase = np.angle(sp_hilbert(theta))
            gamma = mne.filter.filter_data(
                amp_signal[ep], sfreq, f_amp[0], f_amp[1], verbose=False)
            gamma_amp = np.abs(sp_hilbert(gamma))

            mi_real = _modulation_index(theta_phase, gamma_amp)

            surr_mis = []
            n_samples = len(theta_phase)
            shift_range = n_samples // 4
            for _ in range(n_surr):
                shift = np.random.randint(shift_range, n_samples - shift_range)
                shifted_amp = np.roll(gamma_amp, shift)
                surr_mis.append(_modulation_index(theta_phase, shifted_amp))

            mean_surr = np.mean(surr_mis)
            std_surr = np.std(surr_mis)
            z = (mi_real - mean_surr) / std_surr if std_surr > 0 else 0
            epoch_pacs.append(z)
        except Exception:
            pass

    return np.mean(epoch_pacs) if epoch_pacs else np.nan


def _modulation_index(theta_phase, gamma_amp, n_bins=18):
    """Tort et al. (2010) Modulation Index."""
    phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    mean_amp = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (theta_phase >= phase_bins[i]) & (theta_phase < phase_bins[i + 1])
        if mask.sum() > 0:
            mean_amp[i] = gamma_amp[mask].mean()
    if mean_amp.sum() == 0:
        return 0
    mean_amp = mean_amp / mean_amp.sum()
    mean_amp = np.clip(mean_amp, 1e-10, None)
    uniform = np.ones(n_bins) / n_bins
    kl_div = np.sum(mean_amp * np.log(mean_amp / uniform))
    return kl_div / np.log(n_bins)


def compute_pac(phase_signal, amp_signal, sfreq, cfg):
    """Dispatch to tensorpac or fallback."""
    if TENSORPAC_AVAILABLE:
        return compute_pac_tensorpac(phase_signal, amp_signal, sfreq, cfg)
    else:
        return compute_pac_fallback(phase_signal, amp_signal, sfreq, cfg)


def compute_local_node_pac(epochs, node_channels, sfreq, cfg):
    """Compute local PAC within a node. Uses 10% trimmed mean across electrodes."""
    avail = available_channels(node_channels, epochs.ch_names)
    if not avail:
        return np.nan

    trim_prop = cfg.get('pac', {}).get('trim', 0.1)
    data = epochs.copy().pick(avail).get_data()

    electrode_pacs = []
    for ch_idx in range(data.shape[1]):
        ch_signal = data[:, ch_idx, :]
        pac_val = compute_pac(ch_signal, ch_signal, sfreq, cfg)
        if not np.isnan(pac_val):
            electrode_pacs.append(pac_val)

    if not electrode_pacs:
        return np.nan
    if len(electrode_pacs) >= 3:
        return trim_mean(electrode_pacs, proportiontocut=trim_prop)
    else:
        return np.mean(electrode_pacs)


def load_individual_peaks(output_dir):
    """Load individual theta (ITF) and alpha (IAF) peak frequencies.
    
    Returns
    -------
    itf_map : dict  {(subject, block): f_theta}
    iaf_map : dict  {(subject, timepoint): iaf}
    """
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

    if not epochs_dir.exists():
        print(f"Epochs directory not found: {epochs_dir}")
        return

    # Load Individual Peak Frequencies (ITF + IAF)
    itf_map, iaf_map = load_individual_peaks(features_dir)
    if itf_map:
        print(f"Loaded {len(itf_map)} individual theta peaks for dynamic bands.")
    else:
        print("No individual theta peaks found. Using config default.")
    if iaf_map:
        print(f"Loaded {len(iaf_map)} IAF values for theta upper-bound capping.")

    subjects = get_subjects_with_blocks(epochs_dir, 'pac', blocks)
    # ... (subject discovery logic same as before)
    if not subjects:
        legacy = sorted(epochs_dir.glob("*_pac_clean-epo.fif"))
        subjects = sorted(set(f.stem.split("_")[0] for f in legacy))
    if not subjects:
        print("No PAC epoch files found.")
        return

    all_nodes = cfg.get('nodes', {})
    regions = cfg.get('regions', {})

    rf_chs = get_node_channels('RF', cfg)
    rp_chs = get_node_channels('RP', cfg)
    lf_chs = get_node_channels('LF', cfg)
    lp_chs = get_node_channels('LP', cfg)

    between_rows = []
    local_rows = []

    for subj in subjects:
        print(f"\n{'='*50}")
        print(f"  PAC Analysis: {subj}")
        print(f"{'='*50}")

        for block in blocks:
            print(f"\n  --- Block {block} ---")
            epochs = load_block_epochs(subj, block, 'pac', epochs_dir)

            if epochs is None or 'eeg' not in epochs.get_channel_types():
                print(f"  No data for block {block}")
                continue

            sfreq = epochs.info['sfreq']
            
            # --- Dynamic Band Definition (Issue 1+5+6 fixes) ---
            # DEEP copy to avoid leaking into cfg across iterations
            pac_cfg = copy.deepcopy(cfg)
            fixed_band = cfg.get('pac', {}).get('phase_band', [4, 8])
            
            # Try to find individual theta peak
            itf = itf_map.get((subj, block))
            if itf:
                # Cap upper bound at IAF - 1 Hz (not fixed 12 Hz)
                # This prevents theta band bleeding into alpha
                iaf = iaf_map.get(subj, 10.0)  # Default IAF ~10 Hz
                upper_cap = iaf - 1.0
                
                low = max(1.5, itf - 2.0)
                high = min(upper_cap, itf + 2.0)
                individual_band = [low, high]
                
                pac_cfg['pac']['phase_band'] = individual_band
                
                print(f"  Individual Theta Band: {low:.1f}-{high:.1f} Hz "
                      f"(Peak: {itf:.2f}, IAF cap: {upper_cap:.1f})")
            else:
                print(f"  Config Theta Band: {fixed_band[0]}-{fixed_band[1]} Hz (no individual peak)")

            # === 4a: Between-region PAC (H1) ===
            rf_signal = get_node_signal(epochs, rf_chs)
            rp_signal = get_node_signal(epochs, rp_chs)

            between_row = {'subject': subj, 'block': block}

            if rf_signal is not None and rp_signal is not None:
                pac_between = compute_pac(rf_signal, rp_signal, sfreq, pac_cfg)
                between_row['pac_between_RF_RP'] = pac_between
                print(f"  Between PAC (RF→RP): {pac_between:.3f}")
            else:
                between_row['pac_between_RF_RP'] = np.nan

            # Exploratory: LF→LP
            lf_signal = get_node_signal(epochs, lf_chs)
            lp_signal = get_node_signal(epochs, lp_chs)
            if lf_signal is not None and lp_signal is not None:
                between_row['pac_between_LF_LP'] = compute_pac(lf_signal, lp_signal, sfreq, pac_cfg)
            else:
                between_row['pac_between_LF_LP'] = np.nan

            between_rows.append(between_row)

            # === 4b: Local PAC per node (H3) ===
            local_row = {'subject': subj, 'block': block}

            for node_name, node_chs in all_nodes.items():
                pac_local = compute_local_node_pac(epochs, node_chs, sfreq, pac_cfg)
                local_row[f'pac_{node_name}'] = pac_local
                status = f"{pac_local:.3f}" if not np.isnan(pac_local) else "N/A"
                print(f"  Local PAC {node_name}: {status}")

            # Regional aggregation (3 regions)
            for reg_name, reg_nodes in regions.items():
                reg_vals = [local_row.get(f'pac_{n}', np.nan) for n in reg_nodes]
                reg_vals = [v for v in reg_vals if not np.isnan(v)]
                local_row[f'pac_{reg_name}'] = np.mean(reg_vals) if reg_vals else np.nan

            local_rows.append(local_row)

    # Save between-region PAC (H1)
    if between_rows:
        df_between = pd.DataFrame(between_rows)
        out_between = OUTPUT_DIR / "pac_between_features.csv"
        df_between.to_csv(out_between, index=False)
        print(f"\nSaved between-region PAC (long format) to {out_between}")
        print(df_between.to_string(index=False))

    # Save local PAC (H3 + descriptive heatmap)
    if local_rows:
        df_local = pd.DataFrame(local_rows)
        out_local = OUTPUT_DIR / "pac_local_features.csv"
        df_local.to_csv(out_local, index=False)
        print(f"\nSaved local PAC (long format) to {out_local}")
        print(df_local.to_string(index=False))


if __name__ == "__main__":
    main()
