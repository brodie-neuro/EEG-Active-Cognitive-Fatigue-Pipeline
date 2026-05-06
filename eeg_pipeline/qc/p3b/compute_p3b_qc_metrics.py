"""Metric computation for P3b QC."""

from __future__ import annotations

import numpy as np


def select_target_epochs(epochs):
    """Use target trials when the event labels are available."""
    target_keys = [key for key in epochs.event_id.keys() if "stim/target" in key.lower()]
    if target_keys:
        return epochs[target_keys]
    return epochs


def fractional_area_latency(erp_win: np.ndarray, times_win: np.ndarray, fraction: float = 0.5) -> float:
    """Return fractional area latency in seconds."""
    pos_erp = np.clip(erp_win, 0, None)
    total = float(pos_erp.sum())
    if total <= 0:
        return float(times_win[len(times_win) // 2])
    cumulative = np.cumsum(pos_erp)
    idx = int(np.searchsorted(cumulative, fraction * total))
    idx = min(idx, len(times_win) - 1)
    return float(times_win[idx])


def compute_p3b_block_metrics(
    epochs,
    roi_channels: list[str],
    p3b_window: tuple[float, float],
    baseline_window: tuple[float, float],
    thresholds: dict,
) -> tuple[dict, dict]:
    """Compute P3b QC metrics and plot payload for one subject/block."""
    available = [ch for ch in roi_channels if ch in epochs.ch_names]
    missing = [ch for ch in roi_channels if ch not in epochs.ch_names]
    if not available:
        raise ValueError(f"No P3b ROI channels available from {roi_channels}")

    data = epochs.copy().pick(available).get_data() * 1e6
    roi_trials = data.mean(axis=1)
    roi_erp = roi_trials.mean(axis=0)
    times = epochs.times
    p3b_mask = (times >= p3b_window[0]) & (times <= p3b_window[1])
    base_mask = (times >= baseline_window[0]) & (times <= baseline_window[1])
    if int(p3b_mask.sum()) < 2:
        raise ValueError("P3b QC window has fewer than 2 samples.")

    win_erp = roi_erp[p3b_mask]
    win_times = times[p3b_mask]
    p3b_mean = float(win_erp.mean())
    p3b_peak_idx = int(np.argmax(win_erp))
    p3b_peak = float(win_erp[p3b_peak_idx])
    p3b_peak_latency = float(win_times[p3b_peak_idx] * 1000.0)
    p3b_min_idx = int(np.argmin(win_erp))
    p3b_min = float(win_erp[p3b_min_idx])
    p3b_min_latency = float(win_times[p3b_min_idx] * 1000.0)
    fal = float(fractional_area_latency(win_erp, win_times) * 1000.0)

    baseline_mean = float(roi_trials[:, base_mask].mean()) if int(base_mask.sum()) else np.nan
    baseline_sd = float(roi_trials[:, base_mask].std(ddof=1)) if int(base_mask.sum()) else np.nan

    channel_means = {}
    for idx, ch in enumerate(available):
        channel_means[ch] = float(data[:, idx, :].mean(axis=0)[p3b_mask].mean())

    clear_min = float(thresholds.get("clear_min_mean_uV", 0.5))
    weak_min = float(thresholds.get("weak_min_mean_uV", 0.0))
    review_missing = bool(thresholds.get("review_if_missing_roi_channel", True))
    if missing and review_missing:
        status = "review"
    elif p3b_mean >= clear_min:
        status = "clear"
    elif p3b_mean >= weak_min:
        status = "weak"
    else:
        status = "review"

    metrics = {
        "n_epochs": int(len(epochs)),
        "roi_channels": ",".join(available),
        "missing_roi_channels": ",".join(missing),
        "p3b_mean_uV": p3b_mean,
        "p3b_fractional_area_latency_ms": fal,
        "p3b_peak_uV": p3b_peak,
        "p3b_peak_latency_ms": p3b_peak_latency,
        "p3b_min_uV": p3b_min,
        "p3b_min_latency_ms": p3b_min_latency,
        "baseline_mean_uV": baseline_mean,
        "baseline_sd_uV": baseline_sd,
        "roi_positive_in_window": bool(p3b_mean > 0),
        "qc_status": status,
    }
    for ch in roi_channels:
        metrics[f"{ch}_mean_uV"] = channel_means.get(ch, np.nan)

    payload = {
        "times": times,
        "roi_trials": roi_trials,
        "roi_erp": roi_erp,
        "available_channels": available,
        "missing_channels": missing,
        "channel_data_uV": data,
        "p3b_mask": p3b_mask,
    }
    return metrics, payload
