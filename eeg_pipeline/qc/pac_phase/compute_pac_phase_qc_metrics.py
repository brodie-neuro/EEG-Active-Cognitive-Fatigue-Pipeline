"""Metric computation for PAC phase spectral QC."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.signal import welch


@dataclass
class SpectrumQC:
    freqs: np.ndarray
    psd: np.ndarray
    log_psd: np.ndarray
    aperiodic_log: np.ndarray
    residual_log: np.ndarray
    aperiodic_slope: float
    aperiodic_intercept: float
    aperiodic_exponent: float
    aperiodic_method: str
    residual_noise_mad: float


def crop_and_concatenate_trials(
    roi_signal: np.ndarray,
    times: np.ndarray,
    window: tuple[float, float],
) -> np.ndarray:
    """Crop each trial to the PAC analysis window and concatenate trials."""
    tmin, tmax = window
    mask = (times >= tmin) & (times <= tmax)
    if int(mask.sum()) < 10:
        raise ValueError(
            f"PAC phase QC window {tmin}-{tmax}s has fewer than 10 samples."
        )
    cropped = roi_signal[:, mask]
    return np.asarray(cropped, dtype=float).reshape(-1)


def compute_welch_psd(
    signal_1d: np.ndarray,
    sfreq: float,
    psd_cfg: dict,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Compute a transparent Welch PSD for a concatenated signal."""
    signal_1d = np.asarray(signal_1d, dtype=float)
    signal_1d = signal_1d[np.isfinite(signal_1d)]
    if signal_1d.size < 32:
        raise ValueError("Not enough finite samples for PAC phase QC PSD.")

    nperseg = int(round(float(psd_cfg.get("nperseg_seconds", 2.0)) * sfreq))
    nperseg = max(16, min(nperseg, signal_1d.size))
    noverlap = int(round(nperseg * float(psd_cfg.get("noverlap_fraction", 0.5))))
    noverlap = max(0, min(noverlap, nperseg - 1))
    nfft_factor = max(1.0, float(psd_cfg.get("nfft_factor", psd_cfg.get("zero_pad_factor", 1.0))))
    nfft = max(nperseg, int(round(nperseg * nfft_factor)))

    freqs, psd = welch(
        signal_1d,
        fs=sfreq,
        window=psd_cfg.get("window", "hann"),
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=psd_cfg.get("detrend", "constant"),
        scaling="density",
    )
    fmin = float(psd_cfg.get("fmin", 1.0))
    fmax = float(psd_cfg.get("fmax", 40.0))
    keep = (freqs >= fmin) & (freqs <= fmax) & np.isfinite(psd) & (psd > 0)
    if int(keep.sum()) < 8:
        raise ValueError("Welch PSD has too few valid frequency bins for QC.")

    settings = {
        "method": "welch",
        "nperseg": int(nperseg),
        "noverlap": int(noverlap),
        "nfft": int(nfft),
        "nfft_factor": float(nfft_factor),
        "true_resolution_hz": float(sfreq / nperseg),
        "bin_spacing_hz": float(sfreq / nfft),
        "fmin": fmin,
        "fmax": fmax,
    }
    return freqs[keep], psd[keep], settings


def _range_mask(freqs: np.ndarray, ranges: list[list[float]] | tuple[tuple[float, float], ...]) -> np.ndarray:
    mask = np.zeros(freqs.shape, dtype=bool)
    for lo, hi in ranges or []:
        mask |= (freqs >= float(lo)) & (freqs <= float(hi))
    return mask


def _mad(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)) * 1.4826)
    return mad


def _specparam_fit(
    freqs: np.ndarray,
    psd: np.ndarray,
    cfg: dict,
) -> tuple[float, float, str]:
    """Fit the aperiodic component with required specparam."""
    from specparam import SpectralModel

    peak_width_limits = cfg.get("specparam_peak_width_limits", [1.0, 12.0])
    model = SpectralModel(
        aperiodic_mode="fixed",
        max_n_peaks=int(cfg.get("specparam_max_n_peaks", 6)),
        peak_width_limits=tuple(float(v) for v in peak_width_limits),
        verbose=False,
    )
    fit_range = [float(v) for v in cfg.get("fit_range", [2.0, 20.0])]
    model.fit(freqs, psd, fit_range)
    params = np.asarray(model.get_params("aperiodic"), dtype=float)
    if params.size < 2 or not np.all(np.isfinite(params[:2])):
        raise ValueError("specparam returned invalid aperiodic parameters.")
    intercept = float(params[0])
    exponent = float(params[1])
    slope = -exponent
    return slope, intercept, "specparam"


def fit_aperiodic_and_residual(
    freqs: np.ndarray,
    psd: np.ndarray,
    bands: dict[str, tuple[float, float]],
    aperiodic_cfg: dict,
) -> SpectrumQC:
    """Fit aperiodic background and return residual spectrum in log10 units."""
    eps = np.finfo(float).tiny
    log_psd = np.log10(np.maximum(psd, eps))

    method = str(aperiodic_cfg.get("method", "specparam")).lower()
    if method != "specparam":
        raise ValueError(
            "PAC phase QC requires aperiodic.method = 'specparam'. "
            "Fallback aperiodic fitting is intentionally disabled."
        )
    slope, intercept, fit_method = _specparam_fit(freqs, psd, aperiodic_cfg)

    aperiodic_log = intercept + slope * np.log10(freqs)
    residual_log = log_psd - aperiodic_log

    fit_lo, fit_hi = aperiodic_cfg.get("fit_range", [2.0, 20.0])
    noise_mask = (freqs >= float(fit_lo)) & (freqs <= float(fit_hi))
    noise_mask &= ~_range_mask(freqs, [[lo, hi] for lo, hi in bands.values()])
    noise = _mad(residual_log[noise_mask])
    if noise <= 0:
        noise = _mad(residual_log)
    if noise <= 0:
        noise = float(aperiodic_cfg.get("noise_floor", 1e-6))

    exponent = -float(slope)
    out = SpectrumQC(
        freqs=freqs,
        psd=psd,
        log_psd=log_psd,
        aperiodic_log=aperiodic_log,
        residual_log=residual_log,
        aperiodic_slope=float(slope),
        aperiodic_intercept=float(intercept),
        aperiodic_exponent=exponent,
        aperiodic_method=fit_method,
        residual_noise_mad=float(noise),
    )
    return out


def classify_band(
    positive_fraction: float,
    positive_area: float,
    peak_noise_z: float,
    thresholds: dict,
) -> str:
    """Classify residual support using simple configurable rules."""
    clear = (
        positive_fraction >= float(thresholds.get("clear_min_positive_fraction", 0.35))
        and positive_area >= float(thresholds.get("clear_min_positive_area", 0.025))
        and peak_noise_z >= float(thresholds.get("clear_min_peak_noise_z", 1.25))
    )
    if clear:
        return "clear"

    weak = (
        positive_fraction >= float(thresholds.get("weak_min_positive_fraction", 0.20))
        and positive_area >= float(thresholds.get("weak_min_positive_area", 0.01))
        and peak_noise_z >= float(thresholds.get("weak_min_peak_noise_z", 0.75))
    )
    if weak:
        return "weak"
    return "indeterminate"


def compute_band_metrics(
    spectrum: SpectrumQC,
    band_name: str,
    band: tuple[float, float],
    thresholds: dict,
) -> dict:
    """Compute centre-of-mass and residual support metrics for one band."""
    lo, hi = band
    band_mask = (spectrum.freqs >= lo) & (spectrum.freqs <= hi)
    n_bins = int(band_mask.sum())
    base = {
        "band": band_name,
        "band_low_hz": float(lo),
        "band_high_hz": float(hi),
        "n_frequency_bins": n_bins,
    }
    if n_bins < 2:
        base.update(
            {
                "positive_residual_exists": False,
                "positive_residual_fraction": 0.0,
                "centre_of_mass_hz": np.nan,
                "peak_frequency_hz": np.nan,
                "peak_residual_log10": np.nan,
                "peak_residual_noise_z": np.nan,
                "positive_residual_area": 0.0,
                "mean_positive_residual_log10": np.nan,
                "qc_status": "indeterminate",
            }
        )
        return base

    freqs = spectrum.freqs[band_mask]
    residual = spectrum.residual_log[band_mask]
    positive = np.clip(residual, 0.0, None)
    positive_sum = float(np.sum(positive))
    positive_fraction = float(np.mean(positive > 0))
    positive_area = float(np.trapezoid(positive, freqs)) if freqs.size > 1 else positive_sum
    peak_idx = int(np.argmax(positive))
    peak_resid = float(positive[peak_idx])
    peak_noise_z = peak_resid / max(spectrum.residual_noise_mad, float(thresholds.get("noise_floor", 1e-6)))

    status = classify_band(
        positive_fraction=positive_fraction,
        positive_area=positive_area,
        peak_noise_z=peak_noise_z,
        thresholds=thresholds,
    )
    valid = status != "indeterminate" and positive_sum > 0 and math.isfinite(positive_sum)

    if valid:
        com = float(np.sum(freqs * positive) / positive_sum)
        peak_freq = float(freqs[peak_idx])
        mean_positive = float(np.mean(positive[positive > 0])) if np.any(positive > 0) else np.nan
    else:
        com = np.nan
        peak_freq = np.nan
        peak_resid = np.nan
        peak_noise_z = np.nan
        mean_positive = np.nan

    base.update(
        {
            "positive_residual_exists": bool(valid),
            "positive_residual_fraction": positive_fraction,
            "centre_of_mass_hz": com,
            "peak_frequency_hz": peak_freq,
            "peak_residual_log10": peak_resid,
            "peak_residual_noise_z": peak_noise_z,
            "positive_residual_area": positive_area if valid else 0.0,
            "mean_positive_residual_log10": mean_positive,
            "qc_status": status,
        }
    )
    return base
