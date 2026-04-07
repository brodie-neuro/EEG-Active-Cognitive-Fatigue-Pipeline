# eeg_pipeline/analysis/11_band_power.py
"""
Step 9 -- Band Power Extraction (Theta)

Extracts:
  - Frontal midline theta (FMtheta) power at CF node (individualized ITF +/- 2 Hz)

Log-transformed. Outputs long format: one row per subject x block.
Theta feeds into the integrated model as the 'effort' marker.
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
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

pipeline_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pipeline_dir))

from src.utils_io import load_config, discover_subjects
from src.utils_config import get_param
from src.utils_determinism import file_sha256, save_step_qc
from src.utils_features import (
    load_block_epochs, get_subjects_with_blocks,
    available_channels, get_node_channels, filter_excluded_channels,
)

OUTPUT_DIR = pipeline_dir / "outputs" / "features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = pipeline_dir / "outputs" / "figures" / "band_power"
FIG_DIR.mkdir(parents=True, exist_ok=True)

THETA_BAND = (4, 8)


# -- Plotting constants --
_CLR_B1 = '#1E88E5'
_CLR_B5 = '#E53935'
_CLR_THETA = '#4CAF50'
_CLR_ALPHA = '#673AB7'
_STAT_BOX = dict(boxstyle='round,pad=0.4', facecolor='#ECEFF1', edgecolor='#B0BEC5')


def _style_ax(ax):
    """Apply publication style: remove top/right spines, light grid."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)


def _compute_full_psd(epochs, ch_picks, fmin=1, fmax=30):
    """Compute full-range PSD for plotting. Returns (power_uv2, freqs) or (None, None)."""
    avail = available_channels(ch_picks, epochs.ch_names)
    if not avail:
        avail = None
    sp_cfg = get_param('specparam', default={})
    pad_factor = sp_cfg.get('zero_pad_factor', 4)
    n_samples = len(epochs.times)
    n_fft = max(n_samples, int(n_samples * pad_factor))
    psd = epochs.compute_psd(
        method='welch', fmin=fmin, fmax=fmax, picks=avail, verbose=False,
        n_fft=n_fft, n_per_seg=min(n_samples, n_fft)
    )
    data = psd.get_data()
    power = data.mean(axis=tuple(range(data.ndim - 1)))
    power_uv2 = power * 1e12
    return power_uv2, psd.freqs


def _plot_band_power_diagnostic(subj, block, frontal_psd, frontal_freqs,
                                 posterior_psd, posterior_freqs,
                                 theta_band, alpha_band, itf=None, iaf=None):
    """Generate a 1x2 PSD diagnostic figure for one subject/block."""
    fig, (ax_f, ax_p) = plt.subplots(1, 2, figsize=(16, 6))

    # --- Left: Frontal PSD with theta band ---
    ax_f.semilogy(frontal_freqs, frontal_psd, 'k-', lw=1.2, label='PSD')
    # Fixed theta band (dotted outline)
    f_mask_fixed = (frontal_freqs >= THETA_BAND[0]) & (frontal_freqs <= THETA_BAND[1])
    ax_f.fill_between(frontal_freqs, frontal_psd,
                      where=f_mask_fixed, alpha=0.15, color=_CLR_THETA,
                      linestyle=':', label=f'Fixed θ ({THETA_BAND[0]}-{THETA_BAND[1]} Hz)')
    # Individualized band (solid green fill)
    if theta_band != THETA_BAND:
        f_mask_ind = (frontal_freqs >= theta_band[0]) & (frontal_freqs <= theta_band[1])
        ax_f.fill_between(frontal_freqs, frontal_psd,
                          where=f_mask_ind, alpha=0.35, color=_CLR_THETA,
                          label=f'Indiv. θ ({theta_band[0]:.1f}-{theta_band[1]:.1f} Hz)')
    # ITF marker
    if itf is not None and not np.isnan(itf):
        # Find nearest PSD value at ITF
        idx = np.argmin(np.abs(frontal_freqs - itf))
        ax_f.plot(itf, frontal_psd[idx], 'D', color='red', ms=8, zorder=5, label=f'ITF={itf:.1f} Hz')
        ax_f.axvline(itf, color='red', ls='--', lw=0.8, alpha=0.6)
    ax_f.set_xlabel('Frequency (Hz)')
    ax_f.set_ylabel('Power (µV²)')
    ax_f.set_title('Frontal PSD (CF cluster)')
    ax_f.legend(fontsize=8, loc='upper right')
    _style_ax(ax_f)

    # --- Right: Posterior PSD with alpha band ---
    ax_p.semilogy(posterior_freqs, posterior_psd, 'k-', lw=1.2, label='PSD')
    # Fixed alpha band (dotted outline)
    p_mask_fixed = (posterior_freqs >= ALPHA_BAND[0]) & (posterior_freqs <= ALPHA_BAND[1])
    ax_p.fill_between(posterior_freqs, posterior_psd,
                      where=p_mask_fixed, alpha=0.15, color=_CLR_ALPHA,
                      linestyle=':', label=f'Fixed α ({ALPHA_BAND[0]}-{ALPHA_BAND[1]} Hz)')
    # Individualized band (solid purple fill)
    if alpha_band != ALPHA_BAND:
        p_mask_ind = (posterior_freqs >= alpha_band[0]) & (posterior_freqs <= alpha_band[1])
        ax_p.fill_between(posterior_freqs, posterior_psd,
                          where=p_mask_ind, alpha=0.35, color=_CLR_ALPHA,
                          label=f'Indiv. α ({alpha_band[0]:.1f}-{alpha_band[1]:.1f} Hz)')
    # IAF marker
    if iaf is not None and not np.isnan(iaf):
        idx = np.argmin(np.abs(posterior_freqs - iaf))
        ax_p.plot(iaf, posterior_psd[idx], 'D', color='red', ms=8, zorder=5, label=f'IAF={iaf:.1f} Hz')
        ax_p.axvline(iaf, color='red', ls='--', lw=0.8, alpha=0.6)
    ax_p.set_xlabel('Frequency (Hz)')
    ax_p.set_ylabel('Power (µV²)')
    ax_p.set_title('Posterior PSD (CP/LP/RP)')
    ax_p.legend(fontsize=8, loc='upper right')
    _style_ax(ax_p)

    fig.suptitle(f'Band Power Diagnostic — {subj}, Block {block}',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    out = FIG_DIR / f"band_power_{subj}_block{block}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved figure: {out}")


def compute_band_power(epochs, ch_picks, fmin=4, fmax=8):
    """
    Compute mean band power (log-transformed) at specified channels.

    Zero-pads short epochs (e.g. 1s -> 4s) for 0.25 Hz frequency resolution,
    critical for accurate band power in narrow bands like theta (4-8 Hz).

    Returns
    -------
    float : log10 band power (uV^2)
    """
    avail = available_channels(ch_picks, epochs.ch_names)
    if not avail:
        avail = None

    sp_cfg = get_param('specparam', default={})
    pad_factor = sp_cfg.get('zero_pad_factor', 4)
    n_samples = len(epochs.times)
    n_fft = max(n_samples, int(n_samples * pad_factor))

    psd = epochs.compute_psd(
        method='welch', fmin=fmin, fmax=fmax, picks=avail, verbose=False,
        n_fft=n_fft, n_per_seg=min(n_samples, n_fft)
    )
    power = psd.get_data().mean()
    power_uv2 = power * 1e12  # Convert to uV^2
    if power_uv2 <= 0:
        raise RuntimeError(
            f"Non-positive band power encountered for range {fmin}-{fmax} Hz."
        )
    return np.log10(power_uv2)


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
    parser = argparse.ArgumentParser(description='Band power extraction')
    parser.add_argument('--no-plots', action='store_true', help='Skip diagnostic figures')
    args, _ = parser.parse_known_args()
    do_plots = not args.no_plots

    cfg = load_config()
    blocks = cfg.get('blocks', [1, 5])
    epochs_dir = pipeline_dir / "outputs" / "derivatives" / "epochs_clean"
    features_dir = pipeline_dir / "outputs" / "features"
    theta_qc_records = []


    # Prefer explicit FMT cluster from parameters.json.
    cf_channels = get_param('band_power', 'theta_cf_channels', default=None)
    if cf_channels:
        print(f"Using FMT channels from parameters.band_power.theta_cf_channels: {cf_channels}")
    else:
        cf_channels = get_node_channels('CF', cfg)
    if not cf_channels:
        raise ValueError(
            "CF/FMT channels not configured in parameters.band_power.theta_cf_channels "
            "or study.yml nodes.CF."
        )
    cf_channels, _excl = filter_excluded_channels(cf_channels)
    if _excl:
        print(f"  Excluded from CF node: {', '.join(_excl)}")
    if not cf_channels:
        raise ValueError("CF/FMT channels are empty after exclusions — cannot compute band power. Check node config and excluded channels.")

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
        subjects = discover_subjects(
            epochs_dir=epochs_dir,
            blocks=blocks,
            epoch_type='pac',
            require_all_blocks=False,
        )
    if not subjects:
        print("No PAC epoch files found.")
        return

    rows = []
    frontal_psds = {}  # (subj, block) -> (psd, freqs, band, itf)

    for subj in subjects:
        print(f"--- FMtheta Power: {subj} ---")

        for block in blocks:
            epochs = load_block_epochs(subj, block, 'pac', epochs_dir)
            if epochs is None or 'eeg' not in epochs.get_channel_types():
                continue

            # Dynamic Band Definition -- cap at IAF-1 Hz
            itf = itf_map.get((subj, block))
            if itf:
                iaf = iaf_map.get(subj, 10.0)  # Default IAF ~10 Hz
                upper_cap = iaf - 1.0
                low = max(1.5, itf - 2.0)
                high = min(upper_cap, itf + 2.0)
                band = (low, high)
                if low >= high:
                    raise ValueError(
                        f"Block {block}: individualized theta band inverted "
                        f"({low:.1f}>={high:.1f}) for {subj}."
                    )
                else:
                    print(f"  Block {block}: band {low:.1f}-{high:.1f} Hz "
                          f"(Peak: {itf:.2f}, IAF cap: {upper_cap:.1f})")
            else:
                band = default_band
                print(f"  Block {block}: fixed band {band[0]}-{band[1]} Hz")

            power = compute_band_power(epochs, cf_channels, fmin=band[0], fmax=band[1])
            input_file = str(getattr(epochs, "filenames", [""])[0])
            rows.append({
                'subject': subj,
                'block': block,
                'theta_power_log': power,
            })
            theta_qc_records.append({
                "subject": subj,
                "block": block,
                "input_file": input_file,
                "channels_used": list(cf_channels),
                "band": [float(band[0]), float(band[1])],
                "power": float(power),
            })
            print(f"           log10(theta) = {power:.4f}")

            # Store frontal PSD for plotting
            if do_plots:
                f_psd, f_freqs = _compute_full_psd(epochs, cf_channels)
                frontal_psds[(subj, block)] = (f_psd, f_freqs, band, itf)

    # Save theta power
    if rows:
        df = pd.DataFrame(rows)
        output_file = OUTPUT_DIR / "theta_power_features.csv"
        df.to_csv(output_file, index=False)
        print(f"\nSaved FMtheta power features (long format) to {output_file}")
        print(df.to_string(index=False))

    if rows:
        for record in theta_qc_records:
            save_step_qc(
                "09_band_power",
                record["subject"],
                record["block"],
                {
                    "status": "PASS",
                    "input_file": record["input_file"],
                    "input_hash": file_sha256(record["input_file"]) if record["input_file"] else "UNKNOWN",
                    "output_file": str(output_file),
                    "output_hash": file_sha256(output_file),
                    "parameters_used": {"band_type": "theta"},
                    "step_specific": {
                        "band_type": "theta",
                        "channels_used": record["channels_used"],
                        "bands_computed": record["band"],
                        "power_log": record["power"],
                    },
                },
            )


if __name__ == "__main__":
    main()
