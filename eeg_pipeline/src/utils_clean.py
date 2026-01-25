# src/utils_clean.py
import mne
import numpy as np
import matplotlib.pyplot as plt
from pyprep.find_noisy_channels import NoisyChannels
from pathlib import Path


def run_robust_reference(raw):
    """
    Run PyPREP's NoisyChannels detection and RANSAC referencing.
    Includes a safety fallback for synthetic data.
    """
    import pyprep
    import numpy as np

    # 1. Setup PyPREP
    nd = pyprep.NoisyChannels(raw, random_state=42)

    # 2. Try running full RANSAC
    try:
        print("Running PyPREP/RANSAC (this takes ~30-60s)...")
        nd.find_all_bads(ransac=True)
    except IndexError:
        # GPT's Catch: If RANSAC crashes due to "float indices" (often caused by synthetic data)
        print("\n!!! RANSAC CRASHED (Known Issue with Synthetic Data) !!!")
        print("Falling back to standard artifact detection (ransac=False)...")
        nd.find_all_bads(ransac=False)

    # 3. Get the bad channels
    bads = nd.get_bads()

    # 4. Interpolate bads and re-reference
    print(f"Found {len(bads)} bad channels: {bads}")
    raw.info['bads'] = bads
    raw_clean = raw.copy()
    raw_clean.interpolate_bads(reset_bads=True)
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