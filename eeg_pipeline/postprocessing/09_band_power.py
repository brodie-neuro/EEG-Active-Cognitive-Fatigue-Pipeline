# eeg_pipeline/postprocessing/09_band_power.py
"""
Step 09: Band Power Extraction (Theta)

Extracts confirmatory frontal midline theta (FM-theta) power at the CF node
using a fixed 4-8 Hz band.

Descriptive theta peak summaries live elsewhere and are not used to define
the theta-power band in this step.
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
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pipeline_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pipeline_dir))

from src.utils_io import load_config, discover_subjects
from src.utils_config import get_param
from src.utils_determinism import file_sha256, save_step_qc
from src.utils_features import (
    load_block_epochs,
    get_subjects_with_blocks,
    available_channels,
    get_node_channels,
    filter_excluded_channels,
)

OUTPUT_DIR = pipeline_dir / "outputs" / "features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = pipeline_dir / "outputs" / "figures" / "band_power"
FIG_DIR.mkdir(parents=True, exist_ok=True)

THETA_BAND = (4.0, 8.0)
_CLR_THETA = "#4CAF50"


def _style_ax(ax):
    """Apply a compact publication-style axis format."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)


def _compute_full_psd(epochs, ch_picks, fmin=1, fmax=30):
    """Compute full-range PSD for plotting."""
    avail = available_channels(ch_picks, epochs.ch_names)
    if not avail:
        avail = None
    sp_cfg = get_param("specparam", default={}) or {}
    pad_factor = sp_cfg.get("zero_pad_factor", 4)
    n_samples = len(epochs.times)
    n_fft = max(n_samples, int(n_samples * pad_factor))
    psd = epochs.compute_psd(
        method="welch",
        fmin=fmin,
        fmax=fmax,
        picks=avail,
        verbose=False,
        n_fft=n_fft,
        n_per_seg=min(n_samples, n_fft),
    )
    data = psd.get_data()
    power = data.mean(axis=tuple(range(data.ndim - 1)))
    power_uv2 = power * 1e12
    return power_uv2, psd.freqs


def _plot_band_power_diagnostic(subj, block, frontal_psd, frontal_freqs):
    """Generate a frontal theta PSD diagnostic figure for one subject/block."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.semilogy(frontal_freqs, frontal_psd, "k-", lw=1.2, label="PSD")
    fixed_mask = (frontal_freqs >= THETA_BAND[0]) & (frontal_freqs <= THETA_BAND[1])
    ax.fill_between(
        frontal_freqs,
        frontal_psd,
        where=fixed_mask,
        alpha=0.25,
        color=_CLR_THETA,
        label=f"Fixed theta ({THETA_BAND[0]}-{THETA_BAND[1]} Hz)",
    )

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (uV^2)")
    ax.set_title("Frontal PSD (CF cluster)")
    ax.legend(fontsize=8, loc="upper right")
    _style_ax(ax)

    fig.suptitle(f"Band Power Diagnostic - {subj}, Block {block}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = FIG_DIR / f"band_power_{subj}_block{block}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved figure: {out}")


def compute_band_power(epochs, ch_picks, fmin=4.0, fmax=8.0):
    """Compute log10 mean band power (uV^2) for the requested channels."""
    avail = available_channels(ch_picks, epochs.ch_names)
    if not avail:
        avail = None

    sp_cfg = get_param("specparam", default={}) or {}
    pad_factor = sp_cfg.get("zero_pad_factor", 4)
    n_samples = len(epochs.times)
    n_fft = max(n_samples, int(n_samples * pad_factor))

    psd = epochs.compute_psd(
        method="welch",
        fmin=fmin,
        fmax=fmax,
        picks=avail,
        verbose=False,
        n_fft=n_fft,
        n_per_seg=min(n_samples, n_fft),
    )
    power = psd.get_data().mean()
    power_uv2 = power * 1e12
    if power_uv2 <= 0:
        raise RuntimeError(
            f"Non-positive band power encountered for range {fmin}-{fmax} Hz."
        )
    return float(np.log10(power_uv2))


def main():
    parser = argparse.ArgumentParser(description="Band power extraction")
    parser.add_argument("--no-plots", action="store_true", help="Skip diagnostic figures")
    args, _ = parser.parse_known_args()
    do_plots = not args.no_plots

    cfg = load_config()
    blocks = cfg.get("blocks", [1, 5])
    epochs_dir = pipeline_dir / "outputs" / "derivatives" / "epochs_clean"
    theta_qc_records = []

    cf_channels = get_param("band_power", "theta_cf_channels", default=None)
    if cf_channels:
        print(f"Using FMT channels from parameters.band_power.theta_cf_channels: {cf_channels}")
    else:
        cf_channels = get_node_channels("CF", cfg)
    if not cf_channels:
        raise ValueError(
            "CF/FMT channels not configured in parameters.band_power.theta_cf_channels "
            "or study.yml nodes.CF."
        )
    cf_channels, excluded = filter_excluded_channels(cf_channels)
    if excluded:
        print(f"  Excluded from CF node: {', '.join(excluded)}")
    if not cf_channels:
        raise ValueError(
            "CF/FMT channels are empty after exclusions - cannot compute band power. "
            "Check node config and excluded channels."
        )

    band_cfg = get_param("band_power", default={}) or {}
    theta_band = tuple(float(x) for x in band_cfg.get("theta_default", THETA_BAND))
    if len(theta_band) != 2 or theta_band[0] >= theta_band[1]:
        raise ValueError(f"Invalid band_power.theta_default={theta_band}.")

    print(f"Using fixed theta band {theta_band[0]}-{theta_band[1]} Hz for FM-theta power.")

    subjects = get_subjects_with_blocks(epochs_dir, "pac", blocks)
    if not subjects:
        subjects = discover_subjects(
            epochs_dir=epochs_dir,
            blocks=blocks,
            epoch_type="pac",
            require_all_blocks=False,
        )
    if not subjects:
        print("No PAC epoch files found.")
        return

    rows = []
    output_file = OUTPUT_DIR / "theta_power_features.csv"

    for subj in subjects:
        print(f"--- FMtheta Power: {subj} ---")

        for block in blocks:
            epochs = load_block_epochs(subj, block, "pac", epochs_dir)
            if epochs is None or "eeg" not in epochs.get_channel_types():
                continue

            print(f"  Block {block}: fixed band {theta_band[0]}-{theta_band[1]} Hz")
            power = compute_band_power(epochs, cf_channels, fmin=theta_band[0], fmax=theta_band[1])
            input_file = str(getattr(epochs, "filenames", [""])[0])
            rows.append({
                "subject": subj,
                "block": block,
                "theta_power_log": power,
            })
            theta_qc_records.append({
                "subject": subj,
                "block": block,
                "input_file": input_file,
                "channels_used": list(cf_channels),
                "band": [float(theta_band[0]), float(theta_band[1])],
                "power": float(power),
            })
            print(f"           log10(theta) = {power:.4f}")

            if do_plots:
                frontal_psd, frontal_freqs = _compute_full_psd(epochs, cf_channels)
                _plot_band_power_diagnostic(subj, block, frontal_psd, frontal_freqs)

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
        print(f"\nSaved FMtheta power features (long format) to {output_file}")
        print(df.to_string(index=False))

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
