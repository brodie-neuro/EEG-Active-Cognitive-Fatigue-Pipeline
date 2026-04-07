# src/utils_qc.py
import sys
import numpy as np
import mne
import re


def basic_filters(raw, hp, lp, notch):
    raw = raw.copy()
    # Windows environments can fail with process-based parallel filtering.
    # LOCKED to 1 for deterministic reproducibility.
    default_n_jobs = 1
    try:
        raw.filter(hp, lp, fir_design="firwin", n_jobs=default_n_jobs)
    except Exception:
        raw.filter(hp, lp, fir_design="firwin", n_jobs=1)

    if notch:
        try:
            raw.notch_filter(freqs=[notch], n_jobs=default_n_jobs)
        except Exception:
            raw.notch_filter(freqs=[notch], n_jobs=1)

    return raw


def find_bad_channels(raw, flat_thresh_uv=0.01, noisy_thresh_std=5.0):
    data = raw.get_data(picks="eeg")

    ptp = np.ptp(data, axis=1)

    eeg_names = raw.copy().pick("eeg").ch_names
    flat = [ch for ch, amp in zip(eeg_names, ptp) if amp < flat_thresh_uv * 1e-6]

    psd, freqs = mne.time_frequency.psd_array_welch(
        data, sfreq=raw.info["sfreq"], fmin=30, fmax=90, n_fft=2048
    )
    bandpow = psd.mean(axis=1)
    z = (bandpow - bandpow.mean()) / bandpow.std()
    noisy = [ch for ch, zi in zip(eeg_names, z) if zi > noisy_thresh_std]

    return list(set(flat + noisy))


def is_synthetic_subject(subj: str) -> bool:
    """Return True only for explicitly synthetic/simulated subjects."""
    if not subj:
        return False
    s = str(subj).lower()
    # Avoid false positives for real subjects containing 'test' in the name.
    tokens = ("fake", "synthetic", "simulated", "sim")
    return any(tok in s for tok in tokens)
