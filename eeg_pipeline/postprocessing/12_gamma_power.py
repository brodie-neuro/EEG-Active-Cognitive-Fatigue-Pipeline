# eeg_pipeline/analysis/19_gamma_power.py
"""
Gamma power extraction pipeline with EMG contamination diagnostics.

Mirrors the theta pipeline (script 19) for the gamma band, providing:
  1. Per-block gamma features CSV (envelope + descriptive PSD)
  2. Pipeline diagnostic figure (violin plots, raw PSD, EMG proxy)
  3. Topography figure with contamination checks
  4. Contrast figure (window/block differences, gamma-EMG scatter)

Two gamma measures:
  - Gamma A (Envelope, primary): 4th-order Butterworth IIR bandpass 55-85 Hz,
    zero-phase (filtfilt via MNE), Hilbert transform, mean envelope per trial.
    PAC-compatible. Filtered on full epoch then cropped to avoid edge artefacts.
  - Gamma B (PSD, descriptive): Welch PSD (zero-padded), log10 mean power in
    55-85 Hz. No SpecParam aperiodic correction (unstable at high frequencies
    due to muscle/line noise). Kept as descriptive sanity check only.

EMG proxy (primary safeguard):
  - Same two methods on 70-110 Hz band.
  - Spatial correlation between gamma and EMG topographies flags
    potential muscle contamination:
      r > 0.7: high contamination risk
      0.4-0.7: moderate contamination risk
  - Trial-level correlation within ROI also reported.
  - Gamma/EMG ratio reported as robustness scalar.

Edge-safe filtering:
  Envelope is computed on the full 0-1.0s epoch, then cropped to the analysis
  window (0-0.6s or 0.6-1.0s) for averaging. This avoids IIR edge transients
  contaminating the gamma envelope estimate.

Windows (from theta pipeline):
  - 0-0.6s (stimulus, clean cognitive window)
  

References:
  - Gamma envelope: matches PAC script 13 filter parameters
  - EMG contamination: Muthukumaraswamy, 2013, NeuroImage
"""
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
from scipy.signal import hilbert as sp_hilbert

pipeline_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pipeline_dir))

from src.utils_io import load_config, discover_subjects, iter_derivative_files
from src.utils_features import (
    load_block_epochs, available_channels,
    get_node_channels, filter_excluded_channels,
)
from src.utils_config import get_param

OUTPUT_DIR = pipeline_dir / "outputs" / "features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = pipeline_dir / "outputs" / "analysis_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_TAG = os.environ.get("EEG_OUTPUT_TAG", "").strip()

# --- Constants ---
GAMMA_BAND = (55.0, 85.0)
EMG_BAND = (70.0, 110.0)
PAD_FACTOR = 4
N_BOOTSTRAP = 1000
CI_LEVEL = 0.95
RNG_SEED = 42
BUTTER_ORDER = 4  # matches PAC script 13
WINDOWS = [
    (0.0, 0.6, "0-0.6s (stimulus)"),
]
# Rim channels: temporal/fronto-temporal where muscle artefact is strongest
RIM_CHANNELS = ["T7", "T8", "FT7", "FT8", "TP7", "TP8"]
MIDLINE_CHANNELS = ["Fz", "FCz", "Cz", "Pz"]


def _tag_path(base: str) -> Path:
    s, ext = Path(base).stem, Path(base).suffix
    if OUTPUT_TAG:
        s = f"{s}_{OUTPUT_TAG}"
    return FIG_DIR / f"{s}{ext}"


# ---------------------------------------------------------------------------
# Epoch creation (same pattern as script 19)
# ---------------------------------------------------------------------------

def _make_gamma_epochs(subj, block, raw_dir, epochs_dir, tmin=0.0, tmax=1.0):
    """Create onset-locked epochs from ICA-cleaned continuous data.

    Trial selection matched to autoreject-cleaned P3b epochs.
    """
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
        print(f"    No ICA-cleaned raw file for {subj} block {block}")
        return None

    raw = mne.io.read_raw_fif(raw_file, preload=True, verbose=False)
    events_raw, event_id = mne.events_from_annotations(raw, verbose=False)

    clean_p3b = load_block_epochs(subj, block, "p3b", epochs_dir)
    if clean_p3b is not None:
        onset_codes = set(clean_p3b.events[:, 2])
        clean_samples = set(clean_p3b.events[:, 0])
    else:
        onset_codes = {1}
        clean_samples = None

    onset_dict = {k: v for k, v in event_id.items() if v in onset_codes}
    if not onset_dict:
        print(f"    No onset events found. Raw codes: {set(event_id.values())}")
        return None

    epochs = mne.Epochs(
        raw, events_raw, onset_dict,
        tmin=tmin, tmax=tmax,
        baseline=None, preload=True, reject=None, verbose=False,
    )

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
# Channel handling (same pattern as script 21)
# ---------------------------------------------------------------------------

def _get_good_eeg_picks(epochs_win):
    """Return channel names for good EEG channels, excluding EOG, bads,
    and pre-specified analysis exclusions."""
    eog_chs = ["LHEOG", "RHEOG", "BVEOG", "TVEOG"]
    all_eeg = mne.pick_types(epochs_win.info, eeg=True, exclude=eog_chs)
    all_names = [epochs_win.ch_names[i] for i in all_eeg]
    bads = set(epochs_win.info["bads"])
    good_names = [ch for ch in all_names if ch not in bads]
    good_names, excluded = filter_excluded_channels(good_names)
    if excluded:
        print(f"    Excluded from analysis: {', '.join(excluded)}")
    return good_names


# ---------------------------------------------------------------------------
# Gamma A: Envelope (PAC-compatible, edge-safe)
# ---------------------------------------------------------------------------

def _bandpass_envelope_edge_safe(epochs_data, sfreq, band, tmin_crop,
                                  tmax_crop, times,
                                  order=BUTTER_ORDER):
    """Bandpass filter on full epoch, then crop to window for envelope.

    This avoids IIR edge transients contaminating the gamma estimate.
    Filter is applied to the full epoch (0-1.0s), then only the analysis
    window (e.g. 0-0.6s) is used for the mean envelope.

    Parameters
    ----------
    epochs_data : ndarray, shape (n_trials, n_times)
        Full-epoch data (0-1.0s).
    sfreq : float
        Sampling frequency.
    band : tuple
        (low, high) frequency in Hz.
    tmin_crop, tmax_crop : float
        Analysis window boundaries (seconds from epoch start).
    times : ndarray
        Time vector for the epoch.

    Returns
    -------
    envelopes : ndarray, shape (n_trials,)
        Mean envelope amplitude per trial in the cropped window.
    """
    n_trials = epochs_data.shape[0]
    envelopes = np.full(n_trials, np.nan)

    # Determine crop indices
    crop_mask = (times >= tmin_crop) & (times <= tmax_crop)
    if crop_mask.sum() == 0:
        return envelopes

    for i in range(n_trials):
        try:
            # Filter on the FULL epoch to avoid edge transients
            filtered = mne.filter.filter_data(
                epochs_data[i].astype(np.float64), sfreq,
                l_freq=band[0], h_freq=band[1],
                method="iir",
                iir_params=dict(order=order, ftype="butter"),
                verbose=False,
            )
            analytic = sp_hilbert(filtered)
            envelope = np.abs(analytic)
            # Crop to analysis window AFTER filtering
            envelopes[i] = float(np.mean(envelope[crop_mask]))
        except Exception:
            pass
    return envelopes


def _compute_envelope_features(epochs_full, ch_names, band, tmin_crop,
                                tmax_crop, rng, n_boot=N_BOOTSTRAP):
    """Compute envelope amplitude with edge-safe filtering and bootstrap CI.

    Filters on the full epoch (0-1.0s), then crops to the analysis window
    for averaging.

    Returns dict with mean, ci_lo, ci_hi, per_trial array.
    """
    avail = available_channels(ch_names, epochs_full.ch_names)
    if not avail:
        return None

    sfreq = epochs_full.info["sfreq"]
    if band[1] >= sfreq / 2:
        print(f"    WARNING: band upper {band[1]} Hz >= Nyquist "
              f"{sfreq/2} Hz, skipping envelope")
        return None

    data = epochs_full.copy().pick(avail).get_data()  # (n_ep, n_ch, n_times)
    roi_data = data.mean(axis=1)  # (n_ep, n_times)
    times = epochs_full.times

    trial_envs = _bandpass_envelope_edge_safe(
        roi_data, sfreq, band, tmin_crop, tmax_crop, times)

    valid = trial_envs[np.isfinite(trial_envs)]
    if len(valid) == 0:
        return None

    # Convert to uV (data is in V)
    trial_envs_uv = trial_envs * 1e6
    valid_uv = valid * 1e6
    obs_mean = float(np.mean(valid_uv))

    # Bootstrap
    boot_means = np.full(n_boot, np.nan)
    n_valid = len(valid_uv)
    for b in range(n_boot):
        idx = rng.choice(n_valid, size=n_valid, replace=True)
        boot_means[b] = float(np.mean(valid_uv[idx]))

    alpha = (1 - CI_LEVEL) / 2
    return {
        "mean": obs_mean,
        "ci_lo": float(np.nanpercentile(boot_means, alpha * 100)),
        "ci_hi": float(np.nanpercentile(boot_means, (1 - alpha) * 100)),
        "per_trial": trial_envs_uv,
        "boot_means": boot_means,
    }


# ---------------------------------------------------------------------------
# Gamma B: PSD-based (descriptive log bandpower, no SpecParam)
# ---------------------------------------------------------------------------

def _welch_psd_per_epoch_gamma(epochs_win, ch_names, fmin=40, fmax=120,
                                pad_factor=PAD_FACTOR):
    """Welch PSD per epoch for gamma range, averaged across channels."""
    avail = available_channels(ch_names, epochs_win.ch_names)
    if not avail:
        return None, None

    sfreq = epochs_win.info["sfreq"]
    actual_fmax = min(fmax, sfreq / 2 - 1)
    if actual_fmax <= fmin:
        return None, None

    n_samples = len(epochs_win.times)
    n_fft = max(n_samples, int(n_samples * pad_factor))
    n_per_seg = min(n_samples, n_fft)

    psd_obj = epochs_win.compute_psd(
        method="welch", fmin=fmin, fmax=actual_fmax, picks=avail,
        n_fft=n_fft, n_per_seg=n_per_seg, verbose=False,
    )
    data = psd_obj.get_data()  # (n_epochs, n_channels, n_freqs)
    if data.ndim == 3:
        power_per_epoch = data.mean(axis=1)  # (n_epochs, n_freqs)
    else:
        power_per_epoch = data
    return power_per_epoch, psd_obj.freqs


def _log_bandpower(freqs, power, band):
    """Compute log10 mean power in a frequency band (descriptive, no 1/f)."""
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if mask.sum() == 0:
        return np.nan
    return float(np.log10(np.mean(power[mask]) * 1e12 + 1e-30))


def _compute_psd_features(epochs_win, ch_names, gamma_band, emg_band,
                           rng, n_boot=N_BOOTSTRAP):
    """Compute descriptive PSD-based gamma and EMG with bootstrap CIs.

    Uses log10 mean bandpower (no SpecParam aperiodic correction).
    """
    psd_per_epoch, freqs = _welch_psd_per_epoch_gamma(
        epochs_win, ch_names, fmin=40, fmax=120)
    if psd_per_epoch is None or freqs is None:
        return None

    n_epochs = psd_per_epoch.shape[0]
    avg_psd = psd_per_epoch.mean(axis=0)

    # Observed values (descriptive log bandpower)
    gamma_psd = _log_bandpower(freqs, avg_psd, gamma_band)
    emg_psd = _log_bandpower(freqs, avg_psd, emg_band)

    # Bootstrap
    boot_gamma = np.full(n_boot, np.nan)
    boot_emg = np.full(n_boot, np.nan)
    for b in range(n_boot):
        idx = rng.choice(n_epochs, size=n_epochs, replace=True)
        avg_b = psd_per_epoch[idx].mean(axis=0)
        boot_gamma[b] = _log_bandpower(freqs, avg_b, gamma_band)
        boot_emg[b] = _log_bandpower(freqs, avg_b, emg_band)

    alpha = (1 - CI_LEVEL) / 2

    return {
        "gamma_psd_mean": gamma_psd,
        "gamma_psd_ci_lo": float(np.nanpercentile(boot_gamma, alpha * 100)),
        "gamma_psd_ci_hi": float(np.nanpercentile(boot_gamma,
                                                    (1 - alpha) * 100)),
        "emg_psd_mean": emg_psd,
        "emg_psd_ci_lo": float(np.nanpercentile(boot_emg, alpha * 100)),
        "emg_psd_ci_hi": float(np.nanpercentile(boot_emg,
                                                  (1 - alpha) * 100)),
        # For plotting
        "avg_psd": avg_psd,
        "freqs": freqs,
    }


# ---------------------------------------------------------------------------
# Per-channel computation (for topographies), edge-safe
# ---------------------------------------------------------------------------

def _compute_per_channel(epochs, tmin, tmax):
    """Compute gamma envelope and EMG per channel for topography.

    Uses edge-safe filtering: filter on full epoch, crop for envelope.
    Returns dict with per-channel arrays and MNE info.
    """
    ch_names = _get_good_eeg_picks(epochs)
    if not ch_names:
        return None

    sfreq = epochs.info["sfreq"]
    times = epochs.times
    n_chs = len(ch_names)

    gamma_env = np.full(n_chs, np.nan)
    emg_env = np.full(n_chs, np.nan)

    data = epochs.copy().pick(ch_names).get_data()  # (n_ep, n_ch, n_t)

    for i in range(n_chs):
        ch_data = data[:, i, :]  # (n_ep, n_times)

        # Gamma envelope (filter full epoch, crop to window)
        if GAMMA_BAND[1] < sfreq / 2:
            envs = _bandpass_envelope_edge_safe(
                ch_data, sfreq, GAMMA_BAND, tmin, tmax, times)
            valid = envs[np.isfinite(envs)]
            if len(valid) > 0:
                gamma_env[i] = float(np.mean(valid)) * 1e6  # uV

        # EMG envelope (filter full epoch, crop to window)
        if EMG_BAND[1] < sfreq / 2:
            envs = _bandpass_envelope_edge_safe(
                ch_data, sfreq, EMG_BAND, tmin, tmax, times)
            valid = envs[np.isfinite(envs)]
            if len(valid) > 0:
                emg_env[i] = float(np.mean(valid)) * 1e6  # uV

    # Build info from full epochs (not cropped) to preserve montage
    good_idx = mne.pick_channels(epochs.ch_names, ch_names)
    info_eeg = mne.pick_info(epochs.info, good_idx)

    return {
        "gamma_env": gamma_env,
        "emg_env": emg_env,
        "ch_names": ch_names,
        "info": info_eeg,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _winsorise(values, lo_pct=1, hi_pct=99):
    """Clip values to percentile range, ignoring NaN."""
    valid = values[np.isfinite(values)]
    if len(valid) < 3:
        return values
    lo = np.percentile(valid, lo_pct)
    hi = np.percentile(valid, hi_pct)
    out = values.copy()
    out[np.isfinite(out)] = np.clip(out[np.isfinite(out)], lo, hi)
    return out


def _print_top5(ch_names, values, label):
    """Print the top 5 channels by absolute value."""
    valid_idx = [i for i in range(len(values)) if np.isfinite(values[i])]
    if not valid_idx:
        return
    sorted_idx = sorted(valid_idx, key=lambda i: abs(values[i]),
                         reverse=True)
    top5 = sorted_idx[:5]
    parts = [f"{ch_names[i]}={values[i]:+.4f}" for i in top5]
    print(f"    Top-5 {label}: {', '.join(parts)}")


def _compute_joint_vlims(block_data, metric_key, symmetric=False):
    """Compute joint vlim across all conditions for a metric."""
    all_vals = []
    for d in block_data.values():
        v = d[metric_key]
        valid = v[np.isfinite(v)]
        if len(valid) > 0:
            lo = np.percentile(valid, 1)
            hi = np.percentile(valid, 99)
            all_vals.extend([lo, hi])
    if not all_vals:
        return (0, 1)
    if symmetric:
        absmax = max(abs(min(all_vals)), abs(max(all_vals)))
        if absmax == 0:
            absmax = 0.1
        return (-absmax, absmax)
    vmin, vmax = min(all_vals), max(all_vals)
    if vmin == vmax:
        vmin -= 0.1
        vmax += 0.1
    return (vmin, vmax)


# ---------------------------------------------------------------------------
# Run pipeline for one block x window
# ---------------------------------------------------------------------------

def _run_window(epochs, cf_chs, tmin, tmax, window_label, rng):
    """Run full gamma pipeline for one time window. Returns results dict.

    Epochs are the full 0-1.0s epochs; cropping happens inside envelope
    computation (edge-safe) and for PSD.
    """
    n_trials = len(epochs)
    sfreq = epochs.info["sfreq"]

    print(f"\n  Window {window_label}: {n_trials} trials, sfreq={sfreq}")

    results = {
        "window": window_label, "tmin": tmin, "tmax": tmax,
        "n_trials": n_trials,
    }

    # --- Gamma envelope (55-85 Hz), edge-safe ---
    env_gamma = _compute_envelope_features(
        epochs, cf_chs, GAMMA_BAND, tmin, tmax, rng)
    if env_gamma is not None:
        results["gamma_env_mean"] = env_gamma["mean"]
        results["gamma_env_ci_lo"] = env_gamma["ci_lo"]
        results["gamma_env_ci_hi"] = env_gamma["ci_hi"]
        results["_env_gamma"] = env_gamma
        print(f"    Gamma envelope (55-85 Hz): {env_gamma['mean']:.4f} uV "
              f"[{env_gamma['ci_lo']:.4f}, {env_gamma['ci_hi']:.4f}]")
    else:
        results["gamma_env_mean"] = np.nan
        results["gamma_env_ci_lo"] = np.nan
        results["gamma_env_ci_hi"] = np.nan

    # --- EMG envelope (70-110 Hz), edge-safe ---
    env_emg = _compute_envelope_features(
        epochs, cf_chs, EMG_BAND, tmin, tmax, rng)
    if env_emg is not None:
        results["emg_env_mean"] = env_emg["mean"]
        results["emg_env_ci_lo"] = env_emg["ci_lo"]
        results["emg_env_ci_hi"] = env_emg["ci_hi"]
        results["_env_emg"] = env_emg
        print(f"    EMG envelope (70-110 Hz):  {env_emg['mean']:.4f} uV "
              f"[{env_emg['ci_lo']:.4f}, {env_emg['ci_hi']:.4f}]")
    else:
        results["emg_env_mean"] = np.nan
        results["emg_env_ci_lo"] = np.nan
        results["emg_env_ci_hi"] = np.nan

    # --- Gamma/EMG ratio (robustness scalar) ---
    g_mean = results.get("gamma_env_mean", np.nan)
    e_mean = results.get("emg_env_mean", np.nan)
    if np.isfinite(g_mean) and np.isfinite(e_mean) and e_mean > 0:
        results["gamma_emg_ratio"] = float(np.log10(g_mean / e_mean))
        print(f"    Gamma/EMG log-ratio:       "
              f"{results['gamma_emg_ratio']:.4f}")
    else:
        results["gamma_emg_ratio"] = np.nan

    # --- PSD-based gamma + EMG (descriptive, no SpecParam) ---
    epochs_win = epochs.copy().crop(tmin=tmin, tmax=tmax)
    psd_feat = _compute_psd_features(
        epochs_win, cf_chs, GAMMA_BAND, EMG_BAND, rng)
    if psd_feat is not None:
        results["gamma_psd_mean"] = psd_feat["gamma_psd_mean"]
        results["gamma_psd_ci_lo"] = psd_feat["gamma_psd_ci_lo"]
        results["gamma_psd_ci_hi"] = psd_feat["gamma_psd_ci_hi"]
        results["emg_psd_mean"] = psd_feat["emg_psd_mean"]
        results["emg_psd_ci_lo"] = psd_feat["emg_psd_ci_lo"]
        results["emg_psd_ci_hi"] = psd_feat["emg_psd_ci_hi"]
        results["_psd_feat"] = psd_feat
        print(f"    Gamma PSD (log10 uV2/Hz):  {psd_feat['gamma_psd_mean']:.4f} "
              f"[{psd_feat['gamma_psd_ci_lo']:.4f}, "
              f"{psd_feat['gamma_psd_ci_hi']:.4f}]")
        print(f"    EMG PSD (log10 uV2/Hz):    {psd_feat['emg_psd_mean']:.4f} "
              f"[{psd_feat['emg_psd_ci_lo']:.4f}, "
              f"{psd_feat['emg_psd_ci_hi']:.4f}]")
    else:
        for k in ["gamma_psd_mean", "gamma_psd_ci_lo", "gamma_psd_ci_hi",
                   "emg_psd_mean", "emg_psd_ci_lo", "emg_psd_ci_hi"]:
            results[k] = np.nan

    return results


# ---------------------------------------------------------------------------
# Figure 1: Pipeline diagnostic
# ---------------------------------------------------------------------------

def _plot_pipeline(subj, all_results, blocks):
    """3 rows (gamma_env violin, gamma_psd, emg_env violin) x N cols."""
    sorted_blocks = sorted(blocks)
    conditions = [(b, w) for b in sorted_blocks for w in WINDOWS]
    n_cols = len(conditions)
    colors = {1: "#1E88E5", 5: "#E53935"}

    fig, axes = plt.subplots(3, n_cols, figsize=(5 * n_cols, 12))
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    # Collect y-limits for joint scaling per row
    row_ylims = [[], [], []]

    for col_idx, (block, (tmin, tmax, wlabel)) in enumerate(conditions):
        wkey = f"{tmin}-{tmax}"
        r = all_results.get((block, wkey))
        if r is None:
            continue

        short = wlabel.split("(")[1].rstrip(")") if "(" in wlabel else wlabel
        c = colors.get(block, "#455A64")

        # --- Row 0: Gamma envelope violin + boxplot ---
        ax = axes[0, col_idx]
        env_g = r.get("_env_gamma")
        if env_g is not None:
            valid = env_g["per_trial"][np.isfinite(env_g["per_trial"])]
            if len(valid) > 0:
                vp = ax.violinplot(valid, positions=[0], showmedians=True,
                                   showextrema=False)
                for body in vp["bodies"]:
                    body.set_facecolor(c)
                    body.set_alpha(0.4)
                vp["cmedians"].set_color(c)
                bp = ax.boxplot(valid, positions=[0], widths=0.15,
                                patch_artist=True, showfliers=True,
                                flierprops=dict(marker=".", ms=3, alpha=0.4))
                bp["boxes"][0].set_facecolor(c)
                bp["boxes"][0].set_alpha(0.3)
                ax.axhline(env_g["mean"], color=c, lw=1.5, ls="--",
                           label=f'mean={env_g["mean"]:.3f} uV')
                ax.axhspan(env_g["ci_lo"], env_g["ci_hi"],
                           color=c, alpha=0.1, label="95% CI")
                row_ylims[0].extend([valid.min(), valid.max()])
        ax.set_xticks([])
        ax.set_ylabel("Envelope (uV)")
        ax.set_title(f"B{block} {short}\nGamma env 55-85 Hz",
                     fontsize=9, fontweight="bold")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3, axis="y")

        # --- Row 1: Raw PSD (descriptive, no aperiodic) ---
        ax = axes[1, col_idx]
        psd_f = r.get("_psd_feat")
        if psd_f is not None:
            freqs = psd_f["freqs"]
            log_p = np.log10(np.clip(psd_f["avg_psd"], 1e-30, None) * 1e12)
            ax.plot(freqs, log_p, color=c, lw=1.5, label="PSD")
            ax.axvspan(GAMMA_BAND[0], GAMMA_BAND[1], color="#E8F5E9",
                       alpha=0.4, label="Gamma 55-85")
            ax.axvspan(EMG_BAND[0], EMG_BAND[1], color="#FFF3E0",
                       alpha=0.3, label="EMG 70-110")
            row_ylims[1].extend([log_p.min(), log_p.max()])

            # Annotate bandpower values
            gp = r.get("gamma_psd_mean", np.nan)
            ep = r.get("emg_psd_mean", np.nan)
            ratio = r.get("gamma_emg_ratio", np.nan)
            txt = f"gamma={gp:.3f}\nEMG={ep:.3f}"
            if np.isfinite(ratio):
                txt += f"\nlog(G/E)={ratio:.3f}"
            ax.text(0.97, 0.95, txt, transform=ax.transAxes, fontsize=7,
                    va="top", ha="right", fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              alpha=0.8))
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("log10(Power uV2/Hz)")
        ax.set_title(f"B{block} {short}\nPSD (descriptive)",
                     fontsize=9, fontweight="bold")
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

        # --- Row 2: EMG envelope violin + boxplot ---
        ax = axes[2, col_idx]
        env_e = r.get("_env_emg")
        if env_e is not None:
            valid = env_e["per_trial"][np.isfinite(env_e["per_trial"])]
            if len(valid) > 0:
                vp = ax.violinplot(valid, positions=[0], showmedians=True,
                                   showextrema=False)
                for body in vp["bodies"]:
                    body.set_facecolor("#FF6F00")
                    body.set_alpha(0.4)
                vp["cmedians"].set_color("#FF6F00")
                bp = ax.boxplot(valid, positions=[0], widths=0.15,
                                patch_artist=True, showfliers=True,
                                flierprops=dict(marker=".", ms=3, alpha=0.4))
                bp["boxes"][0].set_facecolor("#FF6F00")
                bp["boxes"][0].set_alpha(0.3)
                ax.axhline(env_e["mean"], color="#FF6F00", lw=1.5, ls="--",
                           label=f'mean={env_e["mean"]:.3f} uV')
                ax.axhspan(env_e["ci_lo"], env_e["ci_hi"],
                           color="#FF6F00", alpha=0.1, label="95% CI")
                row_ylims[2].extend([valid.min(), valid.max()])
        ax.set_xticks([])
        ax.set_ylabel("Envelope (uV)")
        ax.set_title(f"B{block} {short}\nEMG proxy 70-110 Hz",
                     fontsize=9, fontweight="bold")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3, axis="y")

    # Apply joint y-scaling per row
    for row in range(3):
        if row_ylims[row]:
            ylo = min(row_ylims[row])
            yhi = max(row_ylims[row])
            pad = (yhi - ylo) * 0.05 if yhi > ylo else 0.1
            for col_idx in range(n_cols):
                axes[row, col_idx].set_ylim(ylo - pad, yhi + pad)

    fig.suptitle(
        f"Gamma Pipeline Diagnostic -- {subj}\n"
        f"Envelope (Hilbert, edge-safe) | PSD (descriptive log bandpower) | "
        f"55-85 Hz gamma, 70-110 Hz EMG | n_boot={N_BOOTSTRAP}",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = _tag_path(f"gamma_pipeline_{subj}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 2: Topographies
# ---------------------------------------------------------------------------

def _plot_topography(subj, topo_data, blocks):
    """3 rows x N cols: gamma env, EMG env, log(gamma/EMG) ratio."""
    sorted_blocks = sorted(blocks)
    conditions = [(b, w) for b in sorted_blocks for w in WINDOWS]
    n_cols = len(conditions)

    metrics = [
        ("gamma_env", "Gamma Envelope\n55-85 Hz (uV)", "viridis"),
        ("emg_env", "EMG Proxy\n70-110 Hz (uV)", "inferno"),
    ]

    # Compute joint vlims
    joint_vlims = {}
    for mkey, _, _ in metrics:
        joint_vlims[mkey] = _compute_joint_vlims(topo_data, mkey)

    # Gamma/EMG log ratio per condition (contamination diagnostic)
    ratio_vals = {}
    for key, d in topo_data.items():
        g = d["gamma_env"].copy()
        e = d["emg_env"].copy()
        valid = np.isfinite(g) & np.isfinite(e) & (e > 0) & (g > 0)
        ratio = np.full_like(g, np.nan)
        ratio[valid] = np.log10(g[valid] / e[valid])
        ratio_vals[key] = ratio

    # Joint vlim for ratio row (symmetric around 0)
    all_ratio = []
    for rv in ratio_vals.values():
        v = rv[np.isfinite(rv)]
        if len(v) > 0:
            all_ratio.extend([np.percentile(v, 1), np.percentile(v, 99)])
    if all_ratio:
        absmax = max(abs(min(all_ratio)), abs(max(all_ratio)))
        if absmax == 0:
            absmax = 0.1
        ratio_vlim = (-absmax, absmax)
    else:
        ratio_vlim = (-0.1, 0.1)

    fig, axes = plt.subplots(3, n_cols,
                              figsize=(4.5 * n_cols, 12))
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for col_idx, (block, (tmin, tmax, wlabel)) in enumerate(conditions):
        key = (block, f"{tmin}-{tmax}")
        d = topo_data.get(key)
        if d is None:
            continue

        short = wlabel.split("(")[1].rstrip(")") if "(" in wlabel else wlabel

        # Row 0 & 1: gamma and EMG envelopes
        for row_idx, (mkey, mlabel, cmap) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            values = _winsorise(d[mkey].copy())
            vmin, vmax = joint_vlims[mkey]

            im, _ = mne.viz.plot_topomap(
                values, d["info"], axes=ax, show=False,
                cmap=cmap, vlim=(vmin, vmax),
                sensors=True, contours=4,
            )
            ax.set_title(f"B{block} {short}", fontsize=10,
                         fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(mlabel, fontsize=8, fontweight="bold",
                              rotation=0, ha="right", va="center",
                              labelpad=70)
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=7)

        # Row 2: log(Gamma/EMG) ratio (contamination diagnostic)
        ax = axes[2, col_idx]
        ratio = _winsorise(ratio_vals[key].copy())
        im, _ = mne.viz.plot_topomap(
            ratio, d["info"], axes=ax, show=False,
            cmap="RdBu_r", vlim=ratio_vlim,
            sensors=True, contours=4,
        )
        ax.set_title(f"B{block} {short}", fontsize=10,
                     fontweight="bold")
        if col_idx == 0:
            ax.set_ylabel("log10(G/EMG)\n(diagnostic only)",
                          fontsize=8, fontweight="bold", rotation=0,
                          ha="right", va="center", labelpad=70)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        f"Gamma / EMG Topography -- {subj}\n"
        f"Row 1: gamma envelope | Row 2: EMG proxy | "
        f"Row 3: log10(gamma/EMG) ratio (diagnostic only) | "
        f"Joint scaling per row | Winsorised 1-99th pct",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout(rect=[0.1, 0, 1, 0.91])
    out = _tag_path(f"gamma_topography_{subj}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 3: Contrasts
# ---------------------------------------------------------------------------

def _plot_contrasts(subj, topo_data, blocks):
    """Window contrast, block contrast topographies, and gamma-vs-EMG scatter."""
    sorted_blocks = sorted(blocks)
    n_scatter_cols = len(sorted_blocks) * len(WINDOWS)
    n_topo_cols = max(len(sorted_blocks), len(WINDOWS))
    n_cols = max(n_topo_cols, n_scatter_cols)

    fig, axes = plt.subplots(3, n_cols, figsize=(5 * n_cols, 13))
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    # --- Row 0: Window contrast (pre-motor minus motor) per block ---
    all_window_diffs = []
    if len(WINDOWS) >= 2:
        for bi, block in enumerate(sorted_blocks):
            key_pre = (block, f"{WINDOWS[0][0]}-{WINDOWS[0][1]}")
            key_mot = (block, f"{WINDOWS[1][0]}-{WINDOWS[1][1]}")
            d_pre = topo_data.get(key_pre)
            d_mot = topo_data.get(key_mot)
            if d_pre is not None and d_mot is not None:
                common = [ch for ch in d_pre["ch_names"]
                          if ch in d_mot["ch_names"]]
                i1 = [d_pre["ch_names"].index(ch) for ch in common]
                i2 = [d_mot["ch_names"].index(ch) for ch in common]
                diff = _winsorise(d_pre["gamma_env"][i1] - d_mot["gamma_env"][i2])
                all_window_diffs.append(diff)

    if all_window_diffs:
        all_flat = np.concatenate([d[np.isfinite(d)]
                                    for d in all_window_diffs])
        absmax = max(abs(np.min(all_flat)), abs(np.max(all_flat)))
        if absmax == 0:
            absmax = 0.1
    else:
        absmax = 0.1

    if len(WINDOWS) >= 2:
        for bi, block in enumerate(sorted_blocks):
            if bi >= n_cols:
                break
            ax = axes[0, bi]
            key_pre = (block, f"{WINDOWS[0][0]}-{WINDOWS[0][1]}")
            key_mot = (block, f"{WINDOWS[1][0]}-{WINDOWS[1][1]}")
            d_pre = topo_data.get(key_pre)
            d_mot = topo_data.get(key_mot)

            if d_pre is None or d_mot is None:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes)
                continue

            common = [ch for ch in d_pre["ch_names"]
                      if ch in d_mot["ch_names"]]
            i1 = [d_pre["ch_names"].index(ch) for ch in common]
            i2 = [d_mot["ch_names"].index(ch) for ch in common]
            pick_idx = mne.pick_channels(d_pre["info"].ch_names, common)
            info_common = mne.pick_info(d_pre["info"], pick_idx)
            diff = _winsorise(d_pre["gamma_env"][i1] - d_mot["gamma_env"][i2])

            im, _ = mne.viz.plot_topomap(
                diff, info_common, axes=ax, show=False,
                cmap="RdBu_r", vlim=(-absmax, absmax),
                sensors=True, contours=4,
            )
            ax.set_title(f"B{block}: pre-motor\nminus motor",
                         fontsize=9, fontweight="bold")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ci in range(len(sorted_blocks), n_cols):
        axes[0, ci].axis("off")

    # --- Row 1: Block contrast (B5 minus B1) per window ---
    if len(sorted_blocks) >= 2:
        b1, b2 = sorted_blocks[0], sorted_blocks[-1]
        all_block_diffs = []
        for wi, (tmin, tmax, wlabel) in enumerate(WINDOWS):
            d1 = topo_data.get((b1, f"{tmin}-{tmax}"))
            d2 = topo_data.get((b2, f"{tmin}-{tmax}"))
            if d1 is not None and d2 is not None:
                common = [ch for ch in d1["ch_names"]
                          if ch in d2["ch_names"]]
                i1 = [d1["ch_names"].index(ch) for ch in common]
                i2 = [d2["ch_names"].index(ch) for ch in common]
                diff = _winsorise(d2["gamma_env"][i2] - d1["gamma_env"][i1])
                all_block_diffs.append(diff)

        if all_block_diffs:
            all_flat = np.concatenate([d[np.isfinite(d)]
                                        for d in all_block_diffs])
            babs = max(abs(np.min(all_flat)), abs(np.max(all_flat)))
            if babs == 0:
                babs = 0.1
        else:
            babs = 0.1

        for wi, (tmin, tmax, wlabel) in enumerate(WINDOWS):
            if wi >= n_cols:
                break
            ax = axes[1, wi]
            d1 = topo_data.get((b1, f"{tmin}-{tmax}"))
            d2 = topo_data.get((b2, f"{tmin}-{tmax}"))
            if d1 is None or d2 is None:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes)
                continue

            common = [ch for ch in d1["ch_names"]
                      if ch in d2["ch_names"]]
            i1 = [d1["ch_names"].index(ch) for ch in common]
            i2 = [d2["ch_names"].index(ch) for ch in common]
            pick_idx = mne.pick_channels(d1["info"].ch_names, common)
            info_common = mne.pick_info(d1["info"], pick_idx)
            diff = _winsorise(d2["gamma_env"][i2] - d1["gamma_env"][i1])

            short = wlabel.split("(")[1].rstrip(")") if "(" in wlabel \
                else wlabel
            im, _ = mne.viz.plot_topomap(
                diff, info_common, axes=ax, show=False,
                cmap="RdBu_r", vlim=(-babs, babs),
                sensors=True, contours=4,
            )
            ax.set_title(f"B{b2}-B{b1}\n{short}",
                         fontsize=9, fontweight="bold")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ci in range(len(WINDOWS), n_cols):
        axes[1, ci].axis("off")

    # --- Row 2: Gamma vs EMG scatter per condition ---
    for col_idx, (block, (tmin, tmax, wlabel)) in enumerate(
            [(b, w) for b in sorted_blocks for w in WINDOWS]):
        if col_idx >= n_cols:
            break
        ax = axes[2, col_idx]
        key = (block, f"{tmin}-{tmax}")
        d = topo_data.get(key)
        if d is None:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        g = d["gamma_env"]
        e = d["emg_env"]
        valid = np.isfinite(g) & np.isfinite(e)
        if valid.sum() < 3:
            ax.text(0.5, 0.5, "Insufficient data", ha="center",
                    va="center", transform=ax.transAxes)
            continue

        gv, ev = g[valid], e[valid]
        ch_valid = [d["ch_names"][i] for i in range(len(d["ch_names"]))
                    if valid[i]]

        rim_mask = np.array([ch in RIM_CHANNELS for ch in ch_valid])
        mid_mask = np.array([ch in MIDLINE_CHANNELS for ch in ch_valid])
        other_mask = ~rim_mask & ~mid_mask

        if other_mask.any():
            ax.scatter(ev[other_mask], gv[other_mask], s=15, alpha=0.5,
                       color="#757575", label="other")
        if mid_mask.any():
            ax.scatter(ev[mid_mask], gv[mid_mask], s=30, alpha=0.8,
                       color="#1E88E5", marker="D", label="midline")
        if rim_mask.any():
            ax.scatter(ev[rim_mask], gv[rim_mask], s=30, alpha=0.8,
                       color="#E53935", marker="^", label="rim")

        r = np.corrcoef(gv, ev)[0, 1]
        short = wlabel.split("(")[1].rstrip(")") if "(" in wlabel \
            else wlabel
        ax.set_title(f"B{block} {short}\nr={r:.3f}",
                     fontsize=9, fontweight="bold")
        ax.set_xlabel("EMG envelope (uV)")
        ax.set_ylabel("Gamma envelope (uV)")
        ax.legend(fontsize=6, loc="upper left")
        ax.grid(True, alpha=0.3)

        # Identity line
        lim_lo = min(gv.min(), ev.min())
        lim_hi = max(gv.max(), ev.max())
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=0.8,
                alpha=0.4)

    n_actual = len(sorted_blocks) * len(WINDOWS)
    for ci in range(n_actual, n_cols):
        axes[2, ci].axis("off")

    fig.suptitle(
        f"Gamma Contrasts -- {subj}\n"
        f"Row 1: window (pre-motor minus motor) | "
        f"Row 2: block (B5 minus B1) | "
        f"Row 3: gamma vs EMG scatter per condition",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = _tag_path(f"gamma_contrasts_{subj}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Contamination diagnostics (console output)
# ---------------------------------------------------------------------------

def _contamination_diagnostics(topo_data, all_results, blocks):
    """Print spatial + trial-level correlation and rim/midline diagnostics."""
    sorted_blocks = sorted(blocks)
    print(f"\n  {'='*50}")
    print(f"  EMG CONTAMINATION DIAGNOSTICS")
    print(f"  {'='*50}")

    for block in sorted_blocks:
        for tmin, tmax, wlabel in WINDOWS:
            key = (block, f"{tmin}-{tmax}")
            d = topo_data.get(key)
            r_window = all_results.get((block, f"{tmin}-{tmax}"))
            if d is None:
                continue

            short = wlabel.split("(")[1].rstrip(")") if "(" in wlabel \
                else wlabel
            g = d["gamma_env"]
            e = d["emg_env"]
            valid = np.isfinite(g) & np.isfinite(e)

            if valid.sum() < 3:
                continue

            gv, ev = g[valid], e[valid]
            ch_valid = [d["ch_names"][i]
                        for i in range(len(d["ch_names"])) if valid[i]]

            # 1) Spatial correlation (across channels)
            r_spatial = np.corrcoef(gv, ev)[0, 1]
            if r_spatial > 0.7:
                level = "HIGH"
                flag = " *** HIGH EMG CONTAMINATION RISK ***"
            elif r_spatial > 0.4:
                level = "MODERATE"
                flag = " * moderate contamination risk *"
            else:
                level = "LOW"
                flag = ""
            print(f"\n  B{block} {short}: gamma-EMG spatial r = "
                  f"{r_spatial:.3f} ({level}){flag}")

            # 2) Trial-level correlation within ROI
            env_g = r_window.get("_env_gamma") if r_window else None
            env_e = r_window.get("_env_emg") if r_window else None
            if env_g is not None and env_e is not None:
                gt = env_g["per_trial"]
                et = env_e["per_trial"]
                tv = np.isfinite(gt) & np.isfinite(et)
                if tv.sum() >= 5:
                    r_trial = np.corrcoef(gt[tv], et[tv])[0, 1]
                    if r_trial > 0.7:
                        t_flag = " *** HIGH ***"
                    elif r_trial > 0.4:
                        t_flag = " * moderate *"
                    else:
                        t_flag = ""
                    print(f"    Trial-level gamma-EMG r (ROI): "
                          f"{r_trial:.3f}{t_flag}")

            # 3) Rim vs midline ratio
            rim_idx = [i for i, ch in enumerate(ch_valid)
                       if ch in RIM_CHANNELS]
            mid_idx = [i for i, ch in enumerate(ch_valid)
                       if ch in MIDLINE_CHANNELS]

            if rim_idx and mid_idx:
                rim_gamma = np.mean(gv[rim_idx])
                mid_gamma = np.mean(gv[mid_idx])
                rim_emg = np.mean(ev[rim_idx])
                mid_emg = np.mean(ev[mid_idx])

                g_ratio = rim_gamma / mid_gamma if mid_gamma > 0 else np.nan
                e_ratio = rim_emg / mid_emg if mid_emg > 0 else np.nan

                print(f"    Gamma rim/midline ratio: {g_ratio:.2f} "
                      f"(rim={rim_gamma:.4f}, mid={mid_gamma:.4f})")
                print(f"    EMG   rim/midline ratio: {e_ratio:.2f} "
                      f"(rim={rim_emg:.4f}, mid={mid_emg:.4f})")

                if g_ratio > 2.0:
                    print(f"    *** Rim-dominant gamma — likely muscle ***")

            # 4) Order-of-magnitude check
            median_gamma = np.median(gv)
            if r_window:
                g_mean = r_window.get("gamma_env_mean", np.nan)
                if np.isfinite(g_mean):
                    print(f"    Gamma (ROI mean): {g_mean:.4f} uV, "
                          f"(all-ch median): {median_gamma:.4f} uV")

    # Block difference: do gamma and EMG move together?
    if len(sorted_blocks) >= 2:
        b1, b2 = sorted_blocks[0], sorted_blocks[-1]
        print(f"\n  Block contrast (B{b2} - B{b1}):")
        for tmin, tmax, wlabel in WINDOWS:
            d1 = topo_data.get((b1, f"{tmin}-{tmax}"))
            d2 = topo_data.get((b2, f"{tmin}-{tmax}"))
            if d1 is None or d2 is None:
                continue

            short = wlabel.split("(")[1].rstrip(")") if "(" in wlabel \
                else wlabel
            common = [ch for ch in d1["ch_names"]
                      if ch in d2["ch_names"]]
            i1 = [d1["ch_names"].index(ch) for ch in common]
            i2 = [d2["ch_names"].index(ch) for ch in common]

            g1 = d1["gamma_env"][i1]
            g2 = d2["gamma_env"][i2]
            e1 = d1["emg_env"][i1]
            e2 = d2["emg_env"][i2]

            valid = (np.isfinite(g1) & np.isfinite(g2)
                     & np.isfinite(e1) & np.isfinite(e2))
            if valid.sum() < 3:
                continue

            dg = np.mean((g2 - g1)[valid])
            de = np.mean((e2 - e1)[valid])
            r_diff = np.corrcoef((g2 - g1)[valid], (e2 - e1)[valid])[0, 1]

            print(f"    {short}: delta_gamma={dg:+.4f}, "
                  f"delta_EMG={de:+.4f}, r(delta)={r_diff:.3f}")
            if abs(r_diff) > 0.7:
                print(f"    *** gamma and EMG change together — "
                      f"interpret gamma with caution ***")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = load_config()
    blocks = cfg.get("blocks", [1, 5])
    epochs_dir = pipeline_dir / "outputs" / "derivatives" / "epochs_clean"
    raw_dir = pipeline_dir / "outputs" / "derivatives" / "ica_cleaned_raw"

    # CF node channels (Fz, FCz, Cz)
    cf_chs = get_node_channels("CF", cfg) or ["Fz", "FCz", "Cz"]
    cf_chs, _excl = filter_excluded_channels(cf_chs)
    if _excl:
        print(f"  Excluded from CF node: {', '.join(_excl)}")

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

    for subj in subjects:
        print(f"\n{'='*60}")
        print(f"  Gamma Power Pipeline: {subj}")
        print(f"{'='*60}")

        all_results = {}
        topo_data = {}

        for block in blocks:
            epochs = _make_gamma_epochs(subj, block, raw_dir, epochs_dir,
                                         tmin=0.0, tmax=1.0)
            if epochs is None or "eeg" not in epochs.get_channel_types():
                print(f"  Block {block}: could not create epochs")
                continue

            print(f"\n  Block {block}: {len(epochs)} epochs, "
                  f"sfreq={epochs.info['sfreq']}")

            for tmin, tmax, wlabel in WINDOWS:
                print(f"\n  === Block {block}, {wlabel} ===")

                # ROI-level features (full epochs passed for edge-safe filtering)
                res = _run_window(epochs, cf_chs, tmin, tmax, wlabel, rng)
                wkey = f"{tmin}-{tmax}"
                all_results[(block, wkey)] = res

                # Per-channel topography (edge-safe)
                td = _compute_per_channel(epochs, tmin, tmax)
                if td is not None:
                    topo_data[(block, wkey)] = td
                    _print_top5(td["ch_names"], td["gamma_env"],
                                "gamma env (uV)")
                    _print_top5(td["ch_names"], td["emg_env"],
                                "EMG env (uV)")

                # CSV row
                csv_rows.append({
                    "subject": subj,
                    "block": block,
                    "window": wlabel,
                    "window_s": f"{tmin}-{tmax}",
                    "n_trials": res["n_trials"],
                    "gamma_env_mean": res.get("gamma_env_mean", np.nan),
                    "gamma_env_ci_lo": res.get("gamma_env_ci_lo", np.nan),
                    "gamma_env_ci_hi": res.get("gamma_env_ci_hi", np.nan),
                    "gamma_psd_mean": res.get("gamma_psd_mean", np.nan),
                    "gamma_psd_ci_lo": res.get("gamma_psd_ci_lo", np.nan),
                    "gamma_psd_ci_hi": res.get("gamma_psd_ci_hi", np.nan),
                    "emg_env_mean": res.get("emg_env_mean", np.nan),
                    "emg_env_ci_lo": res.get("emg_env_ci_lo", np.nan),
                    "emg_env_ci_hi": res.get("emg_env_ci_hi", np.nan),
                    "emg_psd_mean": res.get("emg_psd_mean", np.nan),
                    "emg_psd_ci_lo": res.get("emg_psd_ci_lo", np.nan),
                    "emg_psd_ci_hi": res.get("emg_psd_ci_hi", np.nan),
                    "gamma_emg_ratio": res.get("gamma_emg_ratio", np.nan),
                    "gamma_band": f"{GAMMA_BAND[0]}-{GAMMA_BAND[1]}",
                    "emg_band": f"{EMG_BAND[0]}-{EMG_BAND[1]}",
                    "pad_factor": PAD_FACTOR,
                    "n_bootstrap": N_BOOTSTRAP,
                })

        # Figures
        if all_results:
            _plot_pipeline(subj, all_results, blocks)
        if topo_data:
            _plot_topography(subj, topo_data, blocks)
            _plot_contrasts(subj, topo_data, blocks)
            _contamination_diagnostics(topo_data, all_results, blocks)

    # Save CSV
    if csv_rows:
        df = pd.DataFrame(csv_rows)
        out = OUTPUT_DIR / "gamma_stim_features.csv"
        df.to_csv(out, index=False)
        print(f"\nSaved to {out}")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
