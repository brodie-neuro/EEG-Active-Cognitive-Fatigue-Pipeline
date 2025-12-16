# src/utils_qc.py
import numpy as np
import mne

def basic_filters(raw, hp, lp, notch):
    raw = raw.copy()
    raw.filter(hp, lp, fir_design="firwin", n_jobs="auto")
    if notch:
        raw.notch_filter(freqs=[notch], n_jobs="auto")
    return raw

def find_bad_channels(raw, flat_thresh_uv=0.01, noisy_thresh_std=5.0):
    data = raw.get_data(picks="eeg")
    ptp = data.ptp(axis=1)
    eeg_names = raw.copy().pick("eeg").ch_names
    flat = [ch for ch, amp in zip(eeg_names, ptp) if amp < flat_thresh_uv * 1e-6]

    psd, freqs = mne.time_frequency.psd_array_welch(
        data, sfreq=raw.info["sfreq"], fmin=30, fmax=90, n_fft=2048
    )
    bandpow = psd.mean(axis=1)
    z = (bandpow - bandpow.mean()) / bandpow.std()
    noisy = [ch for ch, zi in zip(eeg_names, z) if zi > noisy_thresh_std]

    return list(set(flat + noisy))
