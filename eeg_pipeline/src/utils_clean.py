# src/utils_clean.py
import mne
import numpy as np
import matplotlib.pyplot as plt
from pyprep.find_noisy_channels import NoisyChannels
from pathlib import Path


def run_robust_reference(raw, montage_name="standard_1020"):
    """
    Applies PyPREP robust referencing.
    1. Identifies bad channels.
    2. Interpolates them.
    3. Re-references to the average of the GOOD channels.
    """
    print("--- Starting Robust Referencing (PyPREP) ---")

    # PyPREP requires a montage to be set
    # (We assume raw already has it, but good to be safe)
    if raw.get_montage() is None:
        montage = mne.channels.make_standard_montage(montage_name)
        raw.set_montage(montage, match_case=False)

    # 1. Setup PyPREP
    # We must pass the raw data and the sample rate
    nd = NoisyChannels(raw, random_state=42)

    # 2. Find all types of bad channels
    nd.find_all_bads(ransac=True)

    # Get the list of bads
    bads = nd.get_bads()
    print(f"Found {len(bads)} bad channels: {bads}")

    # 3. Interpolate the bads
    # PyPREP identifies them, but we use MNE to fix them
    raw.info['bads'] = bads
    raw_clean = raw.copy()
    raw_clean.interpolate_bads(reset_bads=True)

    # 4. Re-reference to Average
    # Now that bads are fixed, we can safely average all channels
    raw_clean.set_eeg_reference(ref_channels="average", projection=False)

    return raw_clean, bads


def plot_before_after(raw_orig, raw_clean, bads, out_path):
    """Generates a comparison plot to verify the cleaning."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Helper to plot PSD
    def plot_psd_on_ax(data, ax, title):
        psd, freqs = mne.time_frequency.psd_array_welch(
            data.get_data(picks='eeg'),
            sfreq=data.info['sfreq'],
            fmin=1, fmax=100, n_fft=2048
        )
        # Plot mean spectrum
        ax.plot(freqs, 10 * np.log10(psd.mean(0)), color='k', lw=1)
        ax.set_title(title)
        ax.set_ylabel("Power (dB)")
        ax.grid(True)

    plot_psd_on_ax(raw_orig, axes[0], "Original Data (Before)")
    plot_psd_on_ax(raw_clean, axes[1], f"Cleaned Data (After PyPREP)\nInterp: {len(bads)}")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)