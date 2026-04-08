# eeg_pipeline/analysis/19_theta_stim_specparam.py
"""
Stimulus-period theta extraction pipeline.

Pipeline:
  1. Welch (zero-padded) PSD on averaged epochs (zero-padded frequency grid)
  2. SpecParam aperiodic removal (aperiodic fit only, no Gaussian peaks)
  3. Peak detection: Centre-of-Mass (CoM)
  4. Bootstrap stability (block-level): resample epochs, average PSD,
     fit SpecParam, compute CoM — repeat 1000x for 95% CI

Creates onset-locked epochs (0-1.0s) from ICA-cleaned continuous data,
independent of the P3b/ERP pipeline. Trial selection matched to autoreject.

Runs on one primary stimulus window:
  - 0-0.6s (stimulus / pre-motor; aligned with PAC and wPLI windows)

Reports both evoked+induced and induced-only (ERP subtracted) spectra.

Notes:
  - Zero-padding (factor 4) smooths the frequency grid but does NOT improve
    true spectral resolution, which is set by the window length.
  - True resolution for the 0-0.6s window: ~1.67 Hz.

References:
  - SpecParam aperiodic: Donoghue et al., 2020, Nature Neuroscience
  - Centre-of-mass peak: robust to boundary effects, uses full bump shape
"""
import os
import sys
from pathlib import Path

pipeline_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pipeline_dir))

from src.utils_determinism import file_sha256, save_step_qc, set_determinism_env, set_random_seeds

set_determinism_env()
set_random_seeds()

import mne
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils_io import load_config, discover_subjects, iter_derivative_files
from src.utils_features import (
    load_block_epochs, get_subjects_with_blocks,
    available_channels, get_node_channels, filter_excluded_channels,
)
from src.utils_config import get_param

from specparam import SpectralModel

OUTPUT_DIR = pipeline_dir / "outputs" / "features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = pipeline_dir / "outputs" / "analysis_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_TAG = os.environ.get("EEG_OUTPUT_TAG", "").strip()

# --- Constants ---
THETA_BAND = (4.0, 8.0)
PAD_FACTOR = 4
N_BOOTSTRAP = 1000
CI_LEVEL = 0.95
RNG_SEED = 42
WINDOWS = [(0.0, 0.6, "0-0.6s (stimulus)")]


def _env_theta_band(default_band: tuple[float, float]) -> tuple[float, float]:
    """Optional runtime override via EEG_THETA_BAND='lo,hi'."""
    raw = os.environ.get("EEG_THETA_BAND", "").strip()
    if not raw:
        return default_band
    parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(
            f"[theta] invalid EEG_THETA_BAND='{raw}' (expected 'lo,hi')."
        )
    try:
        lo = float(parts[0])
        hi = float(parts[1])
    except ValueError:
        raise ValueError(
            f"[theta] invalid EEG_THETA_BAND='{raw}' (non-numeric)."
        )
    if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
        raise ValueError(
            f"[theta] invalid EEG_THETA_BAND='{raw}' (lo>=hi or non-finite)."
        )
    print(f"[theta] overriding theta band via EEG_THETA_BAND: {lo}-{hi} Hz")
    return (lo, hi)


THETA_BAND = _env_theta_band(THETA_BAND)


def _tag_path(base: str) -> Path:
    s, ext = Path(base).stem, Path(base).suffix
    if OUTPUT_TAG:
        s = f"{s}_{OUTPUT_TAG}"
    return FIG_DIR / f"{s}{ext}"


# ---------------------------------------------------------------------------
# Epoch creation (independent of P3b pipeline)
# ---------------------------------------------------------------------------

def _make_theta_epochs(subj, block, raw_dir, epochs_dir, tmin=0.0, tmax=1.0):
    """Create onset-locked epochs from ICA-cleaned continuous data.

    Epochs span 0 to 1.0s post-stimulus-onset. Trial selection is matched
    to autoreject-cleaned P3b epochs (which trials survived), but the
    epochs themselves are created independently.

    This keeps theta analysis separate from the ERP (P3b) pipeline.
    """
    # Find ICA-cleaned continuous file (new per-subject layout + legacy flat)
    raw_candidates = [
        p for p in iter_derivative_files(
            "ica_cleaned_raw", "*_ica-raw.fif", subject=subj
        )
        if f"_block{block}_" in p.name
    ]
    if not raw_candidates:
        raw_candidates = [
            p for p in iter_derivative_files(
                "ica_cleaned_raw", "*_ica*.fif", subject=subj
            )
            if f"_block{block}_" in p.name
        ]
    raw_file = raw_candidates[0] if raw_candidates else None
    if raw_file is None:
        raise FileNotFoundError(f"No ICA-cleaned raw file for {subj} block {block}.")

    raw = mne.io.read_raw_fif(raw_file, preload=True, verbose=False)
    events_raw, event_id = mne.events_from_annotations(raw, verbose=False)

    # Find onset event codes: match what P3b epochs used (if available)
    clean_p3b = load_block_epochs(subj, block, "p3b", epochs_dir)
    if clean_p3b is not None:
        onset_codes = set(clean_p3b.events[:, 2])
        clean_samples = set(clean_p3b.events[:, 0])
    else:
        onset_codes = {1}
        clean_samples = None

    onset_dict = {k: v for k, v in event_id.items() if v in onset_codes}
    if not onset_dict:
        raise RuntimeError(
            f"No onset events found for {subj} block {block}. Raw codes: {set(event_id.values())}"
        )

    # Create epochs: 0 to 1.0s onset-locked
    epochs = mne.Epochs(
        raw, events_raw, onset_dict,
        tmin=tmin, tmax=tmax,
        baseline=None, preload=True, reject=None, verbose=False,
    )

    # Match trial selection to autoreject-cleaned set
    if clean_samples is not None:
        epoch_samples = epochs.events[:, 0]
        keep = np.array([s in clean_samples for s in epoch_samples])
        if keep.sum() == 0:
            print(f"    WARNING: no matching clean trial samples")
        else:
            drop_idx = np.where(~keep)[0].tolist()
            if drop_idx:
                epochs.drop(drop_idx, reason="not in autoreject clean set")

    n_kept = len(epochs)
    print(f"    Created {n_kept} onset-locked epochs "
          f"[{tmin}-{tmax}s] from {raw_file.name}")
    return epochs


# ---------------------------------------------------------------------------
# Step 1: Welch (zero-padded) PSD
# ---------------------------------------------------------------------------

def _welch_psd(epochs, picks, fmin=1, fmax=40, pad_factor=PAD_FACTOR):
    """Compute zero-padded Welch PSD on averaged epochs.

    Zero-padding (pad_factor) interpolates the frequency grid for smoother
    CoM estimation. True spectral resolution is set by the window length
    (~1.67 Hz for the 0.6s window). The finer grid does NOT add
    spectral information but allows CoM to operate on a denser
    frequency axis.

    Returns (power, freqs) where power is in V^2/Hz.
    """
    avail = available_channels(picks, epochs.ch_names)
    if not avail:
        avail = [epochs.ch_names[0]]

    n_samples = len(epochs.times)
    n_fft = max(n_samples, int(n_samples * pad_factor))
    n_per_seg = min(n_samples, n_fft)

    psd_obj = epochs.compute_psd(
        method="welch", fmin=fmin, fmax=fmax, picks=avail,
        n_fft=n_fft, n_per_seg=n_per_seg, verbose=False,
    )
    data = psd_obj.get_data()
    power = data.mean(axis=tuple(range(data.ndim - 1)))
    return power, psd_obj.freqs


def _welch_psd_per_epoch(epochs, picks, fmin=1, fmax=40,
                          pad_factor=PAD_FACTOR):
    """Compute zero-padded Welch PSD per epoch (for bootstrap resampling).

    Returns (power_per_epoch, freqs) where power_per_epoch is
    shape (n_epochs, n_freqs) averaged across channels.
    """
    avail = available_channels(picks, epochs.ch_names)
    if not avail:
        avail = [epochs.ch_names[0]]

    n_samples = len(epochs.times)
    n_fft = max(n_samples, int(n_samples * pad_factor))
    n_per_seg = min(n_samples, n_fft)

    psd_obj = epochs.compute_psd(
        method="welch", fmin=fmin, fmax=fmax, picks=avail,
        n_fft=n_fft, n_per_seg=n_per_seg, verbose=False,
    )
    data = psd_obj.get_data()  # (n_epochs, n_channels, n_freqs)
    if data.ndim == 3:
        power_per_epoch = data.mean(axis=1)  # (n_epochs, n_freqs)
    else:
        power_per_epoch = data
    return power_per_epoch, psd_obj.freqs


# ---------------------------------------------------------------------------
# Step 2: SpecParam aperiodic removal
# ---------------------------------------------------------------------------

def _specparam_aperiodic(power, freqs, fit_range=None):
    """Fit SpecParam aperiodic component ONLY. No Gaussian peaks used.

    Returns (residual, exponent, aperiodic_fit, r_squared, method).
    Residual is in log10 space: log10(power) - aperiodic_fit.
    """
    sp_cfg = get_param("specparam", default={}) or {}
    if fit_range is None:
        fit_range = sp_cfg.get("freq_range", [3, 30])
    if len(fit_range) != 2 or fit_range[0] >= fit_range[1]:
        raise ValueError(f"Invalid specparam.freq_range={fit_range}.")
    sm = SpectralModel(
        peak_width_limits=sp_cfg.get("peak_width_limits", [1, 6]),
        max_n_peaks=sp_cfg.get("max_n_peaks", 6),
        min_peak_height=sp_cfg.get("min_peak_height", 0.05),
        aperiodic_mode=sp_cfg.get("aperiodic_mode", "fixed"),
        verbose=False,
    )

    sm.fit(freqs, power, list(fit_range))

    ap_params = sm.results.params.aperiodic.params
    offset = float(ap_params[0])
    exponent = float(ap_params[1]) if len(ap_params) > 1 else np.nan

    try:
        r_sq = float(sm.results.metrics.results["gof_rsquared"])
    except Exception:
        r_sq = np.nan

    log_freqs = np.log10(np.clip(freqs, 1e-10, None))
    aperiodic_fit = offset - exponent * log_freqs
    log_power = np.log10(np.clip(power, 1e-30, None))
    residual = log_power - aperiodic_fit

    return residual, exponent, aperiodic_fit, r_sq, "specparam"


# ---------------------------------------------------------------------------
# Step 3: Peak detection
# ---------------------------------------------------------------------------

def _centre_of_mass(freqs, residual, band=THETA_BAND):
    """CoM frequency using positive residual as weights."""
    mask = (freqs >= band[0]) & (freqs <= band[1])
    f_band = freqs[mask]
    r_band = residual[mask]
    weights = np.clip(r_band, 0, None)
    if weights.sum() <= 0:
        return float(np.mean(f_band)), False
    return float(np.average(f_band, weights=weights)), True





# ---------------------------------------------------------------------------
# Step 4: Bootstrap (block-level)
# ---------------------------------------------------------------------------

def _bootstrap_block(power_per_epoch, freqs, n_boot=N_BOOTSTRAP, ci=CI_LEVEL,
                     rng=None):
    """Bootstrap by resampling epochs, averaging PSD, then SpecParam + CoM.

    This preserves the SNR benefit of averaging while assessing stability.
    Returns dict with observed values, bootstrap distributions, and CIs.
    """
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)

    n_epochs = power_per_epoch.shape[0]
    boot_com = np.full(n_boot, np.nan)

    for b in range(n_boot):
        idx = rng.choice(n_epochs, size=n_epochs, replace=True)
        avg_psd = power_per_epoch[idx].mean(axis=0)
        resid, _, _, _, _ = _specparam_aperiodic(avg_psd, freqs)
        if resid is not None and np.any(np.isfinite(resid)):
            com, _ = _centre_of_mass(freqs, resid)
            boot_com[b] = com

    alpha = (1 - ci) / 2
    return {
        "com_boot": boot_com,
        "com_ci_lo": float(np.nanpercentile(boot_com, alpha * 100)),
        "com_ci_hi": float(np.nanpercentile(boot_com, (1 - alpha) * 100)),
    }


def _bootstrap_difference(psd_per_epoch_a, psd_per_epoch_b, freqs,
                           n_boot=N_BOOTSTRAP, ci=CI_LEVEL, rng=None):
    """Bootstrap CI for block difference in CoM."""
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)

    n_a = psd_per_epoch_a.shape[0]
    n_b = psd_per_epoch_b.shape[0]
    boot_com_diff = np.full(n_boot, np.nan)

    for b in range(n_boot):
        idx_a = rng.choice(n_a, size=n_a, replace=True)
        idx_b = rng.choice(n_b, size=n_b, replace=True)
        avg_a = psd_per_epoch_a[idx_a].mean(axis=0)
        avg_b = psd_per_epoch_b[idx_b].mean(axis=0)

        res_a, _, _, _, _ = _specparam_aperiodic(avg_a, freqs)
        res_b, _, _, _, _ = _specparam_aperiodic(avg_b, freqs)

        if (res_a is not None and res_b is not None and
                np.any(np.isfinite(res_a)) and np.any(np.isfinite(res_b))):
            com_a, _ = _centre_of_mass(freqs, res_a)
            com_b, _ = _centre_of_mass(freqs, res_b)
            boot_com_diff[b] = com_b - com_a

    alpha = (1 - ci) / 2
    return {
        "com_diff_boot": boot_com_diff,
        "com_diff_ci_lo": float(np.nanpercentile(boot_com_diff, alpha * 100)),
        "com_diff_ci_hi": float(np.nanpercentile(boot_com_diff, (1 - alpha) * 100)),
    }


# ---------------------------------------------------------------------------
# ERP subtraction for induced-only analysis
# ---------------------------------------------------------------------------

def _subtract_erp(epochs):
    """Subtract the ERP (evoked average) from each epoch to isolate induced.

    Only subtracts from EEG channels; non-EEG channels are left unchanged.
    """
    induced = epochs.copy()
    eeg_picks = mne.pick_types(epochs.info, eeg=True)
    eeg_data = induced.get_data()[:, eeg_picks, :]  # (n_epochs, n_eeg, n_times)
    erp_eeg = eeg_data.mean(axis=0, keepdims=True)  # (1, n_eeg, n_times)
    all_data = induced.get_data()
    all_data[:, eeg_picks, :] = eeg_data - erp_eeg
    induced._data = all_data
    return induced


# ---------------------------------------------------------------------------
# Run pipeline for one window
# ---------------------------------------------------------------------------

def _run_window(epochs, cf_chs, tmin, tmax, window_label, rng):
    """Run the full pipeline for one time window. Returns results dict."""
    epochs_win = epochs.copy().crop(tmin=tmin, tmax=tmax)
    n_trials = len(epochs_win)
    win_dur = tmax - tmin

    print(f"\n  Window {window_label}: {n_trials} trials, "
          f"{len(epochs_win.times)} samples, "
          f"true resolution ~{1/win_dur:.2f} Hz")

    results = {"window": window_label, "tmin": tmin, "tmax": tmax,
               "n_trials": n_trials, "win_dur": win_dur}

    # Also prepare induced-only epochs
    epochs_induced = _subtract_erp(epochs_win)

    for signal_type, ep in [("evoked+induced", epochs_win),
                             ("induced_only", epochs_induced)]:

        tag = signal_type.replace("+", "_")
        print(f"\n    --- {signal_type} ---")

        # Step 1: Welch (zero-padded) PSD
        power, freqs = _welch_psd(ep, cf_chs, fmin=1, fmax=40)
        power_per_epoch, _ = _welch_psd_per_epoch(ep, cf_chs, fmin=1, fmax=40)
        freq_grid = float(freqs[1] - freqs[0]) if len(freqs) > 1 else np.nan
        print(f"    Freq grid spacing: {freq_grid:.3f} Hz "
              f"(pad_factor={PAD_FACTOR}, true res ~{1/win_dur:.2f} Hz)")

        # Step 2: SpecParam aperiodic removal
        resid, ap_exp, ap_fit, r_sq, method = _specparam_aperiodic(power, freqs)
        print(f"    Aperiodic: {method} (exp={ap_exp:.3f}, R2={r_sq:.3f})")

        # Step 3: Peak detection (Centre-of-Mass)
        com_freq, com_valid = _centre_of_mass(freqs, resid)
        com_label = "positive-residual" if com_valid else "flat-residual"

        theta_mask = (freqs >= THETA_BAND[0]) & (freqs <= THETA_BAND[1])
        resid_mean = float(np.mean(resid[theta_mask]))
        resid_peak = float(np.max(resid[theta_mask]))

        print(f"    CoM: {com_freq:.2f} Hz ({com_label})")

        # Step 4: Bootstrap
        print(f"    Bootstrapping (n={N_BOOTSTRAP})...")
        boot = _bootstrap_block(power_per_epoch, freqs, rng=rng)
        print(f"    Bootstrap CoM: [{boot['com_ci_lo']:.2f}, {boot['com_ci_hi']:.2f}]")

        results[f"{tag}_power"] = power
        results[f"{tag}_freqs"] = freqs
        results[f"{tag}_resid"] = resid
        results[f"{tag}_ap_fit"] = ap_fit
        results[f"{tag}_ap_exp"] = ap_exp
        results[f"{tag}_r_sq"] = r_sq
        results[f"{tag}_method"] = method
        results[f"{tag}_com"] = com_freq
        results[f"{tag}_com_label"] = com_label
        results[f"{tag}_resid_mean"] = resid_mean
        results[f"{tag}_resid_peak"] = resid_peak
        results[f"{tag}_boot"] = boot
        results[f"{tag}_psd_per_epoch"] = power_per_epoch

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_pipeline(subj, all_results, blocks):
    """Master figure: 2 rows (windows) x 4 cols (PSD, aperiodic, residual, bootstrap)."""
    sorted_blocks = sorted(blocks)
    colors = {1: "#1E88E5", 5: "#E53935"}
    blabels = {1: "Block 1", 5: "Block 5"}

    n_windows = len(WINDOWS)
    fig, axes = plt.subplots(n_windows, 4, figsize=(22, 6 * n_windows))
    if n_windows == 1:
        axes = axes[np.newaxis, :]

    for row, (tmin, tmax, wlabel) in enumerate(WINDOWS):
        wkey = f"{tmin}-{tmax}"

        # --- Col 0: Raw PSD (evoked+induced) ---
        ax = axes[row, 0]
        for block in sorted_blocks:
            r = all_results.get((block, wkey))
            if r is None:
                continue
            c = colors.get(block, "#455A64")
            power = r["evoked_induced_power"]
            freqs = r["evoked_induced_freqs"]
            ax.semilogy(freqs, power * 1e12, color=c, lw=1.5,
                        label=blabels.get(block, f"B{block}"))
        ax.axvspan(THETA_BAND[0], THETA_BAND[1], color="#E3F2FD", alpha=0.4,
                   label="Theta 4-8 Hz")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power (uV2/Hz)")
        ax.set_title(f"PSD ({wlabel})", fontweight="bold")
        ax.set_xlim(1, 30)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # --- Col 1: Aperiodic fit ---
        ax = axes[row, 1]
        for block in sorted_blocks:
            r = all_results.get((block, wkey))
            if r is None:
                continue
            c = colors.get(block, "#455A64")
            freqs = r["evoked_induced_freqs"]
            log_p = np.log10(np.clip(r["evoked_induced_power"], 1e-30, None))
            ax.plot(freqs, log_p, color=c, lw=1.5,
                    label=f'{blabels.get(block, f"B{block}")} power')
            ap = r["evoked_induced_ap_fit"]
            if ap is not None:
                ax.plot(freqs, ap, color=c, lw=2, ls="--", alpha=0.7,
                        label=f'{blabels.get(block, f"B{block}")} aperiodic '
                              f'(exp={r["evoked_induced_ap_exp"]:.2f})')
        ax.axvspan(THETA_BAND[0], THETA_BAND[1], color="#E3F2FD", alpha=0.4)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("log10(Power)")
        ax.set_title(f"SpecParam Aperiodic ({wlabel})", fontweight="bold")
        ax.set_xlim(1, 30)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

        # --- Col 2: Residual + peaks (both evoked+induced AND induced-only) ---
        ax = axes[row, 2]
        for block in sorted_blocks:
            r = all_results.get((block, wkey))
            if r is None:
                continue
            c = colors.get(block, "#455A64")
            freqs = r["evoked_induced_freqs"]

            # Evoked+induced (solid)
            ax.plot(freqs, r["evoked_induced_resid"], color=c, lw=1.5,
                    label=f'{blabels.get(block, f"B{block}")} evoked+ind')
            if np.isfinite(r["evoked_induced_com"]):
                ax.axvline(r["evoked_induced_com"], color=c, ls="-", lw=2,
                           alpha=0.7)

            # Induced-only (dashed)
            ax.plot(freqs, r["induced_only_resid"], color=c, lw=1.2, ls="--",
                    alpha=0.6,
                    label=f'{blabels.get(block, f"B{block}")} induced')
            if np.isfinite(r["induced_only_com"]):
                ax.axvline(r["induced_only_com"], color=c, ls=":", lw=1.5,
                           alpha=0.5)

        ax.axvspan(THETA_BAND[0], THETA_BAND[1], color="#E3F2FD", alpha=0.4)
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Residual (aperiodic removed)")
        ax.set_title(f"Periodic Residual ({wlabel})", fontweight="bold")
        ax.set_xlim(2, 20)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

        # --- Col 3: Bootstrap distributions ---
        ax = axes[row, 3]
        for block in sorted_blocks:
            r = all_results.get((block, wkey))
            if r is None:
                continue
            c = colors.get(block, "#455A64")
            boot = r["evoked_induced_boot"]
            valid_boot = boot["com_boot"][np.isfinite(boot["com_boot"])]
            if len(valid_boot) > 0:
                ax.hist(valid_boot, bins=50, alpha=0.5, color=c,
                        edgecolor="white",
                        label=f'{blabels.get(block, f"B{block}")} '
                              f'{np.nanmean(valid_boot):.2f} '
                              f'[{boot["com_ci_lo"]:.2f}, '
                              f'{boot["com_ci_hi"]:.2f}]')
                ax.axvline(np.nanmean(valid_boot), color=c, lw=2)
                ax.axvline(boot["com_ci_lo"], color=c, lw=1, ls="--",
                           alpha=0.7)
                ax.axvline(boot["com_ci_hi"], color=c, lw=1, ls="--",
                           alpha=0.7)
        ax.set_xlabel("Bootstrap CoM (Hz)")
        ax.set_ylabel("Count")
        ax.set_title(f"Bootstrap 95% CI ({wlabel})", fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Stimulus Theta Pipeline -- {subj}\n"
        f"Welch (zero-padded) | SpecParam aperiodic | CoM | "
        f"4-8 Hz | pad={PAD_FACTOR} | n_boot={N_BOOTSTRAP}",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = _tag_path(f"theta_pipeline_{subj}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {out}")


def _plot_window_comparison(subj, all_results, blocks, diff_results):
    """Comparison figure: summary table + block difference per window."""
    sorted_blocks = sorted(blocks)
    n_windows = len(WINDOWS)
    win_colors = ["#7B1FA2", "#FF6F00", "#00897B"]

    fig, axes = plt.subplots(1, 1 + n_windows, figsize=(7 + 6 * n_windows, 7))

    # --- Panel A: Summary table ---
    ax = axes[0]
    ax.axis("off")

    col_labels = [""]
    for block in sorted_blocks:
        for _, _, wlabel in WINDOWS:
            short = wlabel.split("(")[1].rstrip(")") if "(" in wlabel else wlabel
            col_labels.append(f"B{block}\n{short}")

    table_data = []

    # CoM
    row = ["CoM (Hz)"]
    for block in sorted_blocks:
        for tmin, tmax, _ in WINDOWS:
            r = all_results.get((block, f"{tmin}-{tmax}"))
            row.append(f"{r['evoked_induced_com']:.2f}" if r else "-")
    table_data.append(row)

    # CoM CI
    row = ["CoM 95% CI"]
    for block in sorted_blocks:
        for tmin, tmax, _ in WINDOWS:
            r = all_results.get((block, f"{tmin}-{tmax}"))
            if r:
                b = r["evoked_induced_boot"]
                row.append(f"[{b['com_ci_lo']:.1f},{b['com_ci_hi']:.1f}]")
            else:
                row.append("-")
    table_data.append(row)

    # Induced CoM
    row = ["Induced CoM"]
    for block in sorted_blocks:
        for tmin, tmax, _ in WINDOWS:
            r = all_results.get((block, f"{tmin}-{tmax}"))
            row.append(f"{r['induced_only_com']:.2f}" if r else "-")
    table_data.append(row)

    # Resid peak
    row = ["Resid peak"]
    for block in sorted_blocks:
        for tmin, tmax, _ in WINDOWS:
            r = all_results.get((block, f"{tmin}-{tmax}"))
            row.append(f"{r['evoked_induced_resid_peak']:.3f}" if r else "-")
    table_data.append(row)

    # Aperiodic exp
    row = ["Aperiodic exp"]
    for block in sorted_blocks:
        for tmin, tmax, _ in WINDOWS:
            r = all_results.get((block, f"{tmin}-{tmax}"))
            row.append(f"{r['evoked_induced_ap_exp']:.2f}" if r else "-")
    table_data.append(row)

    # R2
    row = ["R2"]
    for block in sorted_blocks:
        for tmin, tmax, _ in WINDOWS:
            r = all_results.get((block, f"{tmin}-{tmax}"))
            row.append(f"{r['evoked_induced_r_sq']:.3f}" if r else "-")
    table_data.append(row)

    tbl = ax.table(cellText=table_data, colLabels=col_labels,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.5)
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#E3F2FD")
        tbl[0, j].set_text_props(fontweight="bold")
    ax.set_title("A. Summary Comparison", fontweight="bold", pad=20)

    # --- Panels B, C, D: Block difference bootstrap per window ---
    for i, (tmin, tmax, wlabel) in enumerate(WINDOWS):
        ax = axes[1 + i]
        wkey = f"{tmin}-{tmax}"
        c = win_colors[i % len(win_colors)]
        short = wlabel.split("(")[1].rstrip(")") if "(" in wlabel else wlabel

        if wkey in diff_results:
            d = diff_results[wkey]
            valid = d["com_diff_boot"][np.isfinite(d["com_diff_boot"])]
            if len(valid) > 0:
                obs = float(np.nanmean(valid))
                ax.hist(valid, bins=50, alpha=0.6, color=c,
                        edgecolor="white",
                        label=f'CoM diff: {obs:.2f} '
                              f'[{d["com_diff_ci_lo"]:.2f}, '
                              f'{d["com_diff_ci_hi"]:.2f}]')
                ax.axvline(obs, color=c, lw=2)
                ax.axvline(d["com_diff_ci_lo"], color=c, lw=1, ls="--")
                ax.axvline(d["com_diff_ci_hi"], color=c, lw=1, ls="--")
            ax.axvline(0, color="gray", lw=1, ls="--")
            sig = (d["com_diff_ci_lo"] > 0) or (d["com_diff_ci_hi"] < 0)
            letter = chr(ord("B") + i)
            ax.set_title(f"{letter}. {short} "
                         f"({'CI excludes 0' if sig else 'CI includes 0'})",
                         fontweight="bold")
        else:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                    transform=ax.transAxes)
        ax.set_xlabel("CoM diff B5-B1 (Hz)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Window Comparison -- {subj} | "
        + " vs ".join(w[2] for w in WINDOWS),
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = _tag_path(f"theta_window_comparison_{subj}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def _plot_window_overlay(subj, all_results, blocks):
    """2x2 publication figure: periodic residual + aperiodic fit per block.

    Top row:    Periodic residual (theta peak comparison across windows)
    Bottom row: Raw PSD + aperiodic fit (1/f slope comparison)

    Together these provide two independent lines of evidence for the
    stimulus window (0-0.6 s) as the correct analysis epoch.
    """
    sorted_blocks = sorted(blocks)
    n_blocks = len(sorted_blocks)
    block_colors = {1: "#1E88E5", 5: "#E53935"}
    block_labels_map = {1: "Block 1", 5: "Block 5"}

    # Windows to overlay
    stim_key = "0.0-0.6"
    resp_key = "0.6-1.0"
    maint_key = "0.8-1.8"
    window_styles = {
        stim_key:  {"ls": "-",  "lw": 2.2, "label": "Stimulus (0-0.6 s)"},
        resp_key:  {"ls": "--", "lw": 1.8, "label": "Response control (0.6-1.0 s)"},
        maint_key: {"ls": ":",  "lw": 2.2, "label": "Maintenance (0.8-1.8 s)"},
    }
    all_wkeys = [stim_key, resp_key, maint_key]

    # --- First pass: shared y-limits for periodic row ---
    global_ymin, global_ymax = 0.0, 0.0
    for block in sorted_blocks:
        for wkey in all_wkeys:
            r = all_results.get((block, wkey))
            if r is None:
                continue
            freqs_r = r["evoked_induced_freqs"]
            resid_r = r["evoked_induced_resid"]
            mask = (freqs_r >= 3) & (freqs_r <= 12)
            global_ymin = min(global_ymin, np.min(resid_r[mask]))
            global_ymax = max(global_ymax, np.max(resid_r[mask]))
    pad = max(abs(global_ymin), abs(global_ymax)) * 0.15
    shared_ylo = global_ymin - pad
    shared_yhi = global_ymax + pad

    # --- First pass: shared y-limits for aperiodic row ---
    ap_ymin, ap_ymax = np.inf, -np.inf
    for block in sorted_blocks:
        for wkey in all_wkeys:
            r = all_results.get((block, wkey))
            if r is None:
                continue
            freqs_r = r["evoked_induced_freqs"]
            log_p = np.log10(np.clip(r["evoked_induced_power"], 1e-30, None))
            mask = (freqs_r >= 3) & (freqs_r <= 12)
            ap_ymin = min(ap_ymin, np.min(log_p[mask]))
            ap_ymax = max(ap_ymax, np.max(log_p[mask]))
    ap_pad = (ap_ymax - ap_ymin) * 0.08
    ap_ylo = ap_ymin - ap_pad
    ap_yhi = ap_ymax + ap_pad

    fig, axes = plt.subplots(2, n_blocks, figsize=(8 * n_blocks, 11),
                             squeeze=False)

    for col, block in enumerate(sorted_blocks):
        c = block_colors.get(block, "#455A64")
        blabel = block_labels_map.get(block, f"B{block}")

        # ===== TOP ROW: Periodic residual =====
        ax = axes[0, col]
        periodic_lines = []

        for wkey, style in window_styles.items():
            r = all_results.get((block, wkey))
            if r is None:
                continue

            freqs = r["evoked_induced_freqs"]
            resid = r["evoked_induced_resid"]
            com   = r["evoked_induced_com"]
            resid_peak = r["evoked_induced_resid_peak"]

            ax.plot(freqs, resid, color=c, ls=style["ls"], lw=style["lw"],
                    alpha=0.9 if style["ls"] == "-" else 0.6,
                    label=style["label"])

            # CoM diamond on the curve
            if np.isfinite(com):
                com_y = float(np.interp(com, freqs, resid))
                ax.plot(com, com_y, "D", color=c, ms=7, zorder=5,
                        markeredgecolor="white", markeredgewidth=1.0)

            # Positive area in theta band
            theta_m = (freqs >= THETA_BAND[0]) & (freqs <= THETA_BAND[1])
            if np.any(theta_m):
                pos_r = np.clip(resid[theta_m], 0, None)
                df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
                pos_area = float(np.sum(pos_r) * df)
            else:
                pos_area = 0.0

            short = style["label"].split("(")[0].strip()
            periodic_lines.append(
                f"{short}: pk={resid_peak:.3f}  "
                f"area+={pos_area:.3f}  CoM={com:.1f}Hz"
            )

        # Theta band shading
        ax.axvspan(THETA_BAND[0], THETA_BAND[1], color="#E3F2FD", alpha=0.4)
        ax.axvline(THETA_BAND[0], color="#90CAF9", lw=0.8, alpha=0.6)
        ax.axvline(THETA_BAND[1], color="#90CAF9", lw=0.8, alpha=0.6)
        ax.text(6.0, shared_yhi * 0.92, "Theta (4-8 Hz)",
                ha="center", fontsize=8, color="#1565C0", alpha=0.7,
                fontstyle="italic")
        ax.axvspan(THETA_BAND[1], 12, color="#E0E0E0", alpha=0.15)
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.set_xlim(3, 12)
        ax.set_ylim(shared_ylo, shared_yhi)
        ax.set_xlabel("Frequency (Hz)", fontsize=10)
        ax.set_ylabel("Periodic Residual", fontsize=10)
        ax.set_title(f"{blabel} -- Periodic (aperiodic removed)",
                     fontweight="bold", fontsize=11)
        ax.legend(fontsize=7.5, loc="upper right")
        ax.grid(True, alpha=0.25)

        if periodic_lines:
            ax.text(0.03, 0.03, "\n".join(periodic_lines),
                    transform=ax.transAxes, fontsize=6.5,
                    verticalalignment="bottom", fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                              alpha=0.85, edgecolor="#BDBDBD"))

        # ===== BOTTOM ROW: Aperiodic (raw PSD + fit) =====
        ax = axes[1, col]
        aperiodic_lines = []

        for wkey, style in window_styles.items():
            r = all_results.get((block, wkey))
            if r is None:
                continue

            freqs = r["evoked_induced_freqs"]
            log_p = np.log10(np.clip(r["evoked_induced_power"], 1e-30, None))
            ap_fit = r["evoked_induced_ap_fit"]
            ap_exp = r["evoked_induced_ap_exp"]

            # Raw PSD (faint)
            ax.plot(freqs, log_p, color=c, ls=style["ls"], lw=1.0,
                    alpha=0.35)

            # Aperiodic fit (bold)
            if ap_fit is not None:
                ax.plot(freqs, ap_fit, color=c, ls=style["ls"], lw=2.0,
                        alpha=0.85,
                        label=f'{style["label"]}  exp={ap_exp:.2f}')

            short = style["label"].split("(")[0].strip()
            aperiodic_lines.append(f"{short}: exp={ap_exp:.2f}")

        ax.axvspan(THETA_BAND[0], THETA_BAND[1], color="#E3F2FD", alpha=0.3)
        ax.axvspan(THETA_BAND[1], 12, color="#E0E0E0", alpha=0.15)
        ax.set_xlim(3, 12)
        ax.set_ylim(ap_ylo, ap_yhi)
        ax.set_xlabel("Frequency (Hz)", fontsize=10)
        ax.set_ylabel("log10(Power)", fontsize=10)
        ax.set_title(f"{blabel} -- Aperiodic (1/f slope)",
                     fontweight="bold", fontsize=11)
        ax.legend(fontsize=7.5, loc="upper right")
        ax.grid(True, alpha=0.25)

        if aperiodic_lines:
            ax.text(0.03, 0.03, "\n".join(aperiodic_lines),
                    transform=ax.transAxes, fontsize=7,
                    verticalalignment="bottom", fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                              alpha=0.85, edgecolor="#BDBDBD"))

    fig.suptitle(
        f"Analysis Window Justification -- {subj}\n"
        f"Top: periodic theta exists only in stimulus window  |  "
        f"Bottom: aperiodic exponent steepens in later windows",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = _tag_path(f"theta_window_overlay_{subj}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = load_config()
    blocks = cfg.get("blocks", [1, 5])
    epochs_dir = pipeline_dir / "outputs" / "derivatives" / "epochs_clean"
    raw_dir = pipeline_dir / "outputs" / "derivatives" / "ica_cleaned_raw"

    cf_env = os.environ.get("EEG_THETA_CF_CHANNELS", "").strip()
    if cf_env:
        cf_chs = [c.strip() for c in cf_env.replace(";", ",").split(",") if c.strip()]
        print(f"[theta] overriding CF channels via EEG_THETA_CF_CHANNELS: {cf_chs}")
    else:
        cf_chs = get_node_channels("CF", cfg)
    if not cf_chs:
        raise ValueError("[theta] CF node channels missing from config.")
    cf_chs, _excl = filter_excluded_channels(cf_chs)
    if _excl:
        print(f"  Excluded from CF node: {', '.join(_excl)}")
    if not cf_chs:
        raise ValueError("[theta] CF node channels empty after exclusions — check node config and excluded channels")

    # Standardized subject discovery from clean epochs
    subjects = discover_subjects(
        epochs_dir=epochs_dir,
        blocks=blocks,
        epoch_type="p3b",
        require_all_blocks=False,
    )
    if not subjects:
        print("No subjects found.")
        return

    rng = np.random.default_rng(RNG_SEED)
    csv_rows = []
    qc_records = []

    for subj in subjects:
        print(f"\n{'='*60}")
        print(f"  Theta Pipeline: {subj}")
        print(f"{'='*60}")

        all_results = {}  # keyed by (block, window_key)
        psd_per_epoch_store = {}  # for block-difference bootstrap

        for block in blocks:
            # Create onset-locked epochs (0-1.0s) from ICA-cleaned continuous
            # Independent of P3b pipeline; uses autoreject trial selection only
            epochs = _make_theta_epochs(subj, block, raw_dir, epochs_dir,
                                        tmin=0.0, tmax=1.0)
            if "eeg" not in epochs.get_channel_types():
                raise ValueError(f"Block {block}: theta epochs contain no EEG channels.")

            # Separate extended epochs for maintenance window (0.8-1.8s)
            epochs_long = _make_theta_epochs(subj, block, raw_dir, epochs_dir,
                                             tmin=0.0, tmax=1.8)

            for tmin, tmax, wlabel in WINDOWS:
                print(f"\n  === Block {block}, {wlabel} ===")
                # Use extended epochs only for maintenance window
                if tmax > 1.0 and epochs_long is not None:
                    ep = epochs_long
                else:
                    ep = epochs
                res = _run_window(ep, cf_chs, tmin, tmax, wlabel, rng)
                wkey = f"{tmin}-{tmax}"
                all_results[(block, wkey)] = res
                psd_per_epoch_store[(block, wkey)] = res["evoked_induced_psd_per_epoch"]

                # CSV row
                for sig_type in ["evoked_induced", "induced_only"]:
                    tag = sig_type
                    boot = res.get(f"{tag}_boot", {})
                    row = {
                        "subject": subj,
                        "block": block,
                        "window": wlabel,
                        "window_s": f"{tmin}-{tmax}",
                        "signal": sig_type.replace("_", "+") if "+" not in sig_type else sig_type,
                        "theta_com_hz": res[f"{tag}_com"],
                        "theta_com_label": res[f"{tag}_com_label"],
                        "aperiodic_exp": res[f"{tag}_ap_exp"],
                        "aperiodic_method": res[f"{tag}_method"],
                        "aperiodic_r2": res[f"{tag}_r_sq"],
                        "theta_resid_mean": res[f"{tag}_resid_mean"],
                        "theta_resid_peak": res[f"{tag}_resid_peak"],
                        "boot_com_ci_lo": boot.get("com_ci_lo", np.nan),
                        "boot_com_ci_hi": boot.get("com_ci_hi", np.nan),
                        "theta_band": f"{THETA_BAND[0]}-{THETA_BAND[1]}",
                        "n_trials": res["n_trials"],
                        "pad_factor": PAD_FACTOR,
                        "n_bootstrap": N_BOOTSTRAP,
                    }
                    csv_rows.append(row)
                    qc_records.append({
                        "subject": subj,
                        "block": block,
                        "input_file": str(getattr(ep, "filenames", [""])[0]),
                        "window": wlabel,
                        "signal": row["signal"],
                        "theta_com_hz": row["theta_com_hz"],
                        "aperiodic_exp": row["aperiodic_exp"],
                        "method": row["aperiodic_method"],
                        "channels_used": list(cf_chs),
                    })

        # Block-difference bootstrap for each window
        diff_results = {}
        if len(blocks) >= 2:
            b1, b2 = sorted(blocks)[0], sorted(blocks)[-1]
            for tmin, tmax, wlabel in WINDOWS:
                wkey = f"{tmin}-{tmax}"
                psd_a = psd_per_epoch_store.get((b1, wkey))
                psd_b = psd_per_epoch_store.get((b2, wkey))
                if psd_a is not None and psd_b is not None:
                    freqs = all_results[(b1, wkey)]["evoked_induced_freqs"]
                    print(f"\n  Block diff bootstrap ({wlabel})...")
                    d = _bootstrap_difference(psd_a, psd_b, freqs, rng=rng)
                    diff_results[wkey] = d
                    obs = float(np.nanmean(d["com_diff_boot"]))
                    sig = (d["com_diff_ci_lo"] > 0) or (d["com_diff_ci_hi"] < 0)
                    print(f"    CoM diff: {obs:+.2f} "
                          f"[{d['com_diff_ci_lo']:+.2f}, "
                          f"{d['com_diff_ci_hi']:+.2f}] "
                          f"{'*CI excludes 0*' if sig else 'CI includes 0'}")

        # Plots
        if all_results:
            _plot_pipeline(subj, all_results, blocks)
            _plot_window_comparison(subj, all_results, blocks, diff_results)
            _plot_window_overlay(subj, all_results, blocks)

    if csv_rows:
        df = pd.DataFrame(csv_rows)
        out = OUTPUT_DIR / "theta_stim_features.csv"
        df.to_csv(out, index=False)
        print(f"\nSaved to {out}")
        print(df.to_string(index=False))
        for record in qc_records:
            save_step_qc(
                "15_theta",
                record["subject"],
                record["block"],
                {
                    "status": "PASS",
                    "input_file": record["input_file"],
                    "input_hash": file_sha256(record["input_file"]) if record["input_file"] else "UNKNOWN",
                    "output_file": str(out),
                    "output_hash": file_sha256(out),
                    "parameters_used": {
                        "theta_band": list(THETA_BAND),
                        "window": record["window"],
                    },
                    "step_specific": {
                        "com_freq": record["theta_com_hz"],
                        "aperiodic_exp": record["aperiodic_exp"],
                        "method": record["method"],
                        "channels_used": record["channels_used"],
                        "signal": record["signal"],
                    },
                },
            )


if __name__ == "__main__":
    main()
