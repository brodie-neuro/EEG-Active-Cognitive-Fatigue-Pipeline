# eeg_pipeline/analysis/14_theta_wpli.py
"""
H2: Frontal-parietal theta connectivity breakdown.

Framework
---------
We extract TWO mechanistic features from the same data:
  - H1: PAC (theta phase → gamma amplitude)           [script 11]
  - H2: theta phase synchrony (wPLI)                  [this script]
Both are tested as standalone hypotheses for fatigue-related change.

OPERATIONAL DESIGN
------------------
Measure: theta wPLI (4-8 Hz), debiased (Vinck 2011)
  Theta is the coordinating control rhythm in Holroyd's framework.
  Scalp EEG has robust theta phase signal; phase coupling is most
  reliable in this band. No EMG concern at 4-8 Hz.

Edges (theta):
  C_broad_F-C_broad_P (primary) — bilateral frontal→parietal
  Fz_desc-Pz_desc (descriptive) — single-channel subset check

Note: gamma wPLI was piloted but dropped — values near zero everywhere
(~0.001-0.01), confirming the old AEC gamma 'synchrony' was volume
conduction artefact, not true phase coupling. Theta is the signal.

Sensitivity check: imaginary coherence (iCoh, Nolte 2004)
  Reported alongside wPLI. If both agree in direction, the effect
  is robust to choice of phase-coupling metric.

Why wPLI over AEC
-----------------
1. wPLI discards zero-lag contributions (volume conduction immune).
2. Debiased variant reduces positive sample-size bias.
3. AEC is pragmatic but not the closest match to Holroyd's
   'temporal synchrony' concept — phase metrics are.

Windows
-------
- 0.0-0.6 s: stimulus/cognitive (primary)


Statistics
----------
- Paired test on wPLI: paired t (or Wilcoxon if non-normal).
- Bootstrap 95% CI on mean delta wPLI (1000 resamples).
- Permutation test (within-subject sign flip, 5000 permutations).
- No correction for preregistered single primary
  (C_broad_F-C_broad_P theta, stim window).

Outputs
-------
- features/theta_wpli_subject.csv
- features/theta_wpli_group.csv
- analysis_figures/theta_wpli_{subj}.png
- group_figures/theta_wpli_group.png

References
----------
- Vinck et al. (2011), NeuroImage — debiased wPLI
- Nolte (2004), Clin Neurophysiol — imaginary coherence
- Holroyd & Verguts (2021), Trends Cogn Sci — reward/control gamma
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
from scipy.stats import pearsonr, ttest_rel, wilcoxon, shapiro

pipeline_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pipeline_dir))

from src.utils_io import load_config
from src.utils_features import (
    load_block_epochs, available_channels,
    get_node_channels, save_feature_tables,
)

FEAT_DIR = pipeline_dir / "outputs" / "features"
FEAT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = pipeline_dir / "outputs" / "analysis_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
GRP_DIR = pipeline_dir / "outputs" / "group_figures"
GRP_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_TAG = os.environ.get("EEG_OUTPUT_TAG", "").strip()

# ── Constants ──────────────────────────────────────────────────────────
THETA_BAND = (4.0, 8.0)
BUTTER_ORDER = 4
WINDOWS = [
    (0.0, 0.6, "stim"),
]

# Revised nodes (2026-03-04): C_broad bilateral
SYNC_EDGES = [("C_broad_F", "C_broad_P")]   # primary: bilateral frontal->parietal

RIM_CHANNELS = ["T7", "T8", "FT7", "FT8", "TP7", "TP8",
                "FT9", "FT10", "TP9", "TP10", "P7", "P8"]

N_BOOTSTRAP = 1000
N_PERM = 5000
CI_LEVEL = 0.95
RNG_SEED = 42

EMG_RISK_THRESHOLDS = {"high": 0.7, "moderate": 0.4}


def _tag_path(base: str, directory: Path) -> Path:
    s, ext = Path(base).stem, Path(base).suffix
    if OUTPUT_TAG:
        s = f"{s}_{OUTPUT_TAG}"
    return directory / f"{s}{ext}"

def _filter_rim_channels(ch_names):
    """Exclude rim channels from the EMG-proxy helper path."""
    kept = [ch for ch in ch_names if ch not in RIM_CHANNELS]
    removed = [ch for ch in ch_names if ch in RIM_CHANNELS]
    return kept, removed


# ── Epoch creation ────────────────────────────────────────────────────

def _make_onset_epochs(subj, block, raw_dir, epochs_dir):
    """Obsolete helper kept only as historical context.

    11_theta_wpli.py uses the stored clean PAC epochs directly in main().
    This helper is intentionally not part of the live analysis path.
    """
    raise RuntimeError(
        "_make_onset_epochs() is obsolete. 11_theta_wpli.py uses clean PAC epochs directly."
    )

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
        print(f"    No ICA-cleaned raw for {subj} block {block}")
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
        return None

    epochs = mne.Epochs(
        raw, events_raw, onset_dict,
        tmin=0.0, tmax=1.0,
        baseline=None, preload=True, reject=None, verbose=False,
    )

    if clean_samples is not None:
        epoch_samples = epochs.events[:, 0]
        keep = np.array([s in clean_samples for s in epoch_samples])
        if keep.sum() > 0:
            drop_idx = np.where(~keep)[0].tolist()
            if drop_idx:
                epochs.drop(drop_idx, reason="not in autoreject clean set")

    return epochs


# ── wPLI computation ──────────────────────────────────────────────────

def _bandpass_analytic(data_1d, sfreq, band, order=BUTTER_ORDER):
    """Bandpass filter and return analytic signal (complex).

    Uses IIR Butterworth (zero-phase filtfilt via MNE) on the full epoch,
    then Hilbert to get the analytic signal.

    Parameters
    ----------
    data_1d : ndarray (n_times,)
    sfreq : float
    band : tuple (lo, hi) in Hz

    Returns
    -------
    analytic : ndarray (n_times,) complex, or None on failure
    """
    try:
        filtered = mne.filter.filter_data(
            data_1d.astype(np.float64), sfreq,
            l_freq=band[0], h_freq=band[1],
            method="iir",
            iir_params=dict(order=order, ftype="butter"),
            verbose=False,
        )
        return sp_hilbert(filtered)
    except Exception:
        return None


def _node_analytic_timeseries(epochs, ch_names, band, sfreq):
    """Extract analytic signal per trial for a node.

    Averages across node channels first, then filters + Hilbert.
    Returns complex ndarray (n_trials, n_times) or None.
    """
    avail = available_channels(ch_names, epochs.ch_names)
    if not avail:
        return None

    if band[1] >= sfreq / 2:
        return None

    data = epochs.copy().pick(avail).get_data()  # (n_trials, n_ch, n_times)
    roi_data = data.mean(axis=1)  # (n_trials, n_times)

    n_trials = roi_data.shape[0]
    n_times = roi_data.shape[1]
    analytic = np.full((n_trials, n_times), np.nan, dtype=complex)

    for i in range(n_trials):
        result = _bandpass_analytic(roi_data[i], sfreq, band)
        if result is not None:
            analytic[i] = result

    valid_mask = np.isfinite(analytic[:, 0].real)
    if valid_mask.sum() == 0:
        return None
    return analytic


def _crop_to_window(signal, times, tmin, tmax):
    """Crop signal to a time window. Returns (n_trials, n_win)."""
    mask = (times >= tmin) & (times <= tmax)
    return signal[:, mask]


def _debiased_wpli(cross_spectra_imag_sign, cross_spectra_imag_sq):
    """Compute debiased wPLI from pre-computed cross-spectral quantities.

    Vinck et al. (2011) debiased wPLI:
        wPLI_debiased = (sum(Im(Sxy))^2 - sum(Im(Sxy)^2)) /
                        (sum(|Im(Sxy)|)^2 - sum(Im(Sxy)^2))

    This formulation allows computation from time-domain cross-spectra
    computed sample-by-sample across concatenated trials, which is
    equivalent to the frequency-domain approach for narrowband signals.
    """
    n = len(cross_spectra_imag_sign)
    if n < 10:
        return np.nan

    # sum of signed imaginary parts
    sum_imag = np.sum(cross_spectra_imag_sign)
    # sum of absolute imaginary parts
    sum_abs_imag = np.sum(np.abs(cross_spectra_imag_sign))
    # sum of squared imaginary parts
    sum_imag_sq = np.sum(cross_spectra_imag_sq)

    numerator = sum_imag ** 2 - sum_imag_sq
    denominator = sum_abs_imag ** 2 - sum_imag_sq

    if denominator <= 0 or not np.isfinite(denominator):
        return np.nan

    return float(numerator / denominator)


def _imaginary_coherence(cross):
    """Imaginary coherence (Nolte 2004).

    iCoh = |mean(Im(Sxy))| / sqrt(mean(|Sxy|^2))

    Insensitive to volume conduction (zero-lag = real-valued, disappears
    in the imaginary part). Unlike wPLI, this retains amplitude weighting.
    """
    n = len(cross)
    if n < 10:
        return np.nan
    imag_mean = np.mean(cross.imag)
    power_mean = np.mean(np.abs(cross) ** 2)
    if power_mean <= 0 or not np.isfinite(power_mean):
        return np.nan
    return float(abs(imag_mean) / np.sqrt(power_mean))


def _compute_connectivity(analytic_a, analytic_b, times, tmin, tmax):
    """Compute debiased wPLI AND imaginary coherence between two nodes.

    1. Crop both analytic signals to the window.
    2. Compute sample-by-sample cross-spectrum: Sxy = a * conj(b).
    3. Concatenate across trials.
    4. Compute debiased wPLI from the imaginary part of Sxy.
    5. Compute imaginary coherence (Nolte 2004) as sensitivity check.

    Parameters
    ----------
    analytic_a, analytic_b : ndarray (n_trials, n_times) complex
    times : ndarray
    tmin, tmax : float

    Returns
    -------
    wpli : float  (debiased wPLI value)
    icoh : float  (imaginary coherence value)
    n_samples : int (total samples used)
    """
    a_win = _crop_to_window(analytic_a, times, tmin, tmax)
    b_win = _crop_to_window(analytic_b, times, tmin, tmax)

    # Only use trials where both nodes have valid data
    valid = (np.isfinite(a_win[:, 0].real) &
             np.isfinite(b_win[:, 0].real))
    if valid.sum() < 3:
        return np.nan, np.nan, 0

    a_valid = a_win[valid]
    b_valid = b_win[valid]

    # Cross-spectrum: Sxy = a * conj(b), for each sample
    cross = a_valid * np.conj(b_valid)
    cross_flat = cross.ravel()  # concatenate trials
    imag = cross_flat.imag

    n_samples = len(imag)
    if n_samples < 10:
        return np.nan, np.nan, 0

    # Debiased wPLI
    wpli = _debiased_wpli(imag, imag ** 2)

    # Imaginary coherence (sensitivity check)
    icoh = _imaginary_coherence(cross_flat)

    return wpli, icoh, n_samples


# ── Signal quality diagnostics (kept for legacy checks) ────────────

def _classify_risk(r):
    r_abs = abs(r) if np.isfinite(r) else 0.0
    if r_abs > EMG_RISK_THRESHOLDS["high"]:
        return "HIGH"
    elif r_abs > EMG_RISK_THRESHOLDS["moderate"]:
        return "moderate"
    return "low"


def _proxy_available(sfreq):
    """Check if proxy band (90-120 Hz) is available given Nyquist."""
    return sfreq / 2.0 >= 122


def _node_envelope_timeseries(epochs, ch_names, band, sfreq,
                              notch_freq=None):
    """Hilbert envelope time series for EMG proxy checks."""
    avail = available_channels(ch_names, epochs.ch_names)
    avail, _ = _filter_rim_channels(avail)
    if not avail:
        return None
    if band[1] >= sfreq / 2:
        return None

    data = epochs.copy().pick(avail).get_data().mean(axis=1)
    n_trials = data.shape[0]
    envelopes = np.full_like(data, np.nan)

    for i in range(n_trials):
        try:
            sig = data[i].astype(np.float64)
            if notch_freq is not None:
                sig = mne.filter.notch_filter(
                    sig, sfreq, freqs=notch_freq,
                    verbose=False, method="iir")
            filtered = mne.filter.filter_data(
                sig, sfreq, l_freq=band[0], h_freq=band[1],
                method="iir",
                iir_params=dict(order=BUTTER_ORDER, ftype="butter"),
                verbose=False)
            envelopes[i] = np.abs(sp_hilbert(filtered))
        except Exception:
            pass

    if np.isfinite(envelopes[:, 0]).sum() == 0:
        return None
    return envelopes


def _aec_concat(env_a, env_b, times, tmin, tmax):
    """AEC-concat for EMG proxy checks only."""
    mask = (times >= tmin) & (times <= tmax)
    a = env_a[:, mask]
    b = env_b[:, mask]
    valid = np.isfinite(a[:, 0]) & np.isfinite(b[:, 0])
    if valid.sum() < 3:
        return np.nan
    av, bv = a[valid].ravel(), b[valid].ravel()
    if len(av) < 10:
        return np.nan
    r, _ = pearsonr(av, bv)
    return float(r)


# ── Statistics ─────────────────────────────────────────────────────────

def _paired_test(b1_vals, b5_vals):
    """Paired test: t-test if normal, Wilcoxon otherwise."""
    deltas = b5_vals - b1_vals
    valid = np.isfinite(deltas)
    d = deltas[valid]
    n = len(d)
    if n < 3:
        return {"test": "insufficient_data", "p": np.nan, "stat": np.nan,
                "n": n}

    if n >= 8:
        _, p_norm = shapiro(d)
    else:
        p_norm = 0.0

    if p_norm > 0.05:
        stat, p = ttest_rel(b1_vals[valid], b5_vals[valid])
        method = "paired_t"
    else:
        stat, p = wilcoxon(d)
        method = "wilcoxon"

    return {"test": method, "p": float(p), "stat": float(stat), "n": n}


def _bootstrap_ci(deltas, n_boot=N_BOOTSTRAP, ci=CI_LEVEL, rng=None):
    """Bootstrap 95% CI on mean delta."""
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    valid = deltas[np.isfinite(deltas)]
    if len(valid) < 2:
        return np.nan, np.nan, np.nan
    boots = np.array([np.mean(rng.choice(valid, len(valid), replace=True))
                      for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    return (float(np.mean(valid)),
            float(np.percentile(boots, alpha * 100)),
            float(np.percentile(boots, (1 - alpha) * 100)))


def _permutation_test(deltas, n_perm=N_PERM, rng=None):
    """Within-subject sign-flip permutation test on mean delta."""
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    valid = deltas[np.isfinite(deltas)]
    n = len(valid)
    if n < 3:
        return np.nan
    obs = np.mean(valid)
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=n)
        if abs(np.mean(valid * signs)) >= abs(obs):
            count += 1
    return float(count / n_perm)


# ── Figures ────────────────────────────────────────────────────────────

def _plot_subject_diagnostic(subj, results_per_block, out_path):
    """Per-subject diagnostic: 1 row (stim) x 2 cols (bars, detail)."""
    n_rows = len(WINDOWS)
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    block_colors = {1: "#1E88E5", 5: "#E53935"}
    edge_labels = [f"{n1}-{n2}" for n1, n2 in SYNC_EDGES]
    blocks_sorted = sorted(results_per_block.keys())

    for row_idx, (tmin, tmax, wlabel) in enumerate(WINDOWS):
        primary = wlabel == "stim"
        role = "PRIMARY_WINDOW" if primary else "secondary_window"

        # Col 0: wPLI bar chart — all edges, both blocks
        ax = axes[row_idx, 0]
        x = np.arange(len(edge_labels))
        width = 0.8 / len(blocks_sorted)
        for bi, block in enumerate(blocks_sorted):
            bdata = results_per_block.get(block, {})
            wpli_vals = [bdata.get(f"wpli_{edge_label}_{wlabel}", np.nan)
                         for edge_label in edge_labels]
            c = block_colors.get(block, "#455A64")
            offset = -0.4 + (bi + 0.5) * width
            ax.bar(x + offset, wpli_vals, width, label=f"B{block}",
                   color=c, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(edge_labels, fontsize=11)
        ax.set_ylabel("wPLI (debiased)")
        ax.set_title(f"Theta wPLI | {wlabel} ({tmin}-{tmax} s) [{role}]",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        ax.axhline(0, color="gray", ls="--", lw=0.8)

        # Col 1: Numeric detail
        ax = axes[row_idx, 1]
        ax.axis("off")
        lines = [f"Theta wPLI + iCoh | {wlabel} ({tmin}-{tmax} s) | {role}",
                 ""]
        for edge_label in edge_labels:
            primary_edge = (wlabel == "stim" and edge_label == "C_broad_F-C_broad_P")
            tag = "  ★ PRIMARY" if primary_edge else ""
            lines.append(f"  {edge_label}{tag}")
            for block in blocks_sorted:
                bdata = results_per_block.get(block, {})
                wpli = bdata.get(f"wpli_{edge_label}_{wlabel}", np.nan)
                icoh = bdata.get(f"icoh_{edge_label}_{wlabel}", np.nan)
                n_s = bdata.get(f"n_{edge_label}_{wlabel}", 0)
                lines.append(
                    f"    B{block}: wPLI={wpli:.4f}  iCoh={icoh:.4f}"
                    f"  (n={n_s:,})")
            lines.append("")
        lines += ["Note: theta (4-8 Hz) — no EMG concern.",
                  "wPLI immune to volume conduction (zero-lag cancellation)."]
        ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
                fontsize=9, va="top", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f8ff",
                          edgecolor="#88bbdd"))

    fig.suptitle(
        f"H3 Theta wPLI — {subj}\n"
        f"Edges: {', '.join(edge_labels)} | Debiased wPLI (Vinck 2011) + iCoh (Nolte 2004)",
        fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    Saved: {out_path}")


def _plot_group_figure(df_group, out_path, rng):
    """Group-level paired dot plot: one panel per edge x window."""
    combos = [(e, w)
              for e in [f"{n1}-{n2}" for n1, n2 in SYNC_EDGES]
              for _, _, w in WINDOWS]
    n_panels = len(combos)
    n_cols = min(3, n_panels)
    n_rows_fig = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows_fig, n_cols,
                              figsize=(5 * n_cols, 6 * n_rows_fig),
                              squeeze=False)

    for idx, (edge, window) in enumerate(combos):
        r_idx, c_idx = divmod(idx, n_cols)
        ax = axes[r_idx, c_idx]
        sub = df_group[(df_group["edge"] == edge) &
                       (df_group["window"] == window)]
        if sub.empty:
            ax.set_title(f"{edge} {window}: no data")
            continue

        piv = sub.pivot(index="subject", columns="block", values="wpli")
        blocks_avail = sorted(piv.columns)
        if len(blocks_avail) < 2:
            ax.set_title(f"{edge} {window}: 1 block only")
            continue
        b1_col, b5_col = blocks_avail[0], blocks_avail[-1]
        b1 = piv[b1_col].values
        b5 = piv[b5_col].values
        valid = np.isfinite(b1) & np.isfinite(b5)
        b1v, b5v = b1[valid], b5[valid]

        if len(b1v) == 0:
            ax.set_title(f"{edge} {window}: no pairs")
            continue

        for i in range(len(b1v)):
            ax.plot([0, 1], [b1v[i], b5v[i]], "o-", color="#999999",
                    alpha=0.5, markersize=5, lw=1)
        ax.plot(0, np.mean(b1v), "D", color="#1E88E5", markersize=10,
                zorder=5, label=f"B{b1_col} mean={np.mean(b1v):.4f}")
        ax.plot(1, np.mean(b5v), "D", color="#E53935", markersize=10,
                zorder=5, label=f"B{b5_col} mean={np.mean(b5v):.4f}")

        deltas = b5v - b1v
        test_result = _paired_test(b1, b5)
        mean_d, ci_lo, ci_hi = _bootstrap_ci(deltas, rng=rng)
        p_perm = _permutation_test(deltas, rng=rng)

        primary = window == "stim" and edge == "C_broad_F-C_broad_P"
        tag = " ★PRIMARY" if primary else " (descriptive)"

        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"Block {b1_col}", f"Block {b5_col}"])
        ax.set_ylabel("wPLI")
        ax.set_title(
            f"{edge} | theta | {window}{tag}\n"
            f"Δ={mean_d:.4f} [{ci_lo:.4f},{ci_hi:.4f}]\n"
            f"{test_result['test']} p={test_result['p']:.4f}, "
            f"perm p={p_perm:.4f}",
            fontsize=9, fontweight="bold")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3, axis="y")

    for idx in range(n_panels, n_rows_fig * n_cols):
        r_idx, c_idx = divmod(idx, n_cols)
        axes[r_idx, c_idx].set_visible(False)

    fig.suptitle("H3: Theta wPLI Connectivity — Group Inference\n"
                 "Primary: C_broad_F-C_broad_P stim | Debiased wPLI (Vinck 2011)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved group figure: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────

def main():
    cfg = load_config()
    blocks = cfg.get("blocks", [1, 5])
    rng = np.random.default_rng(RNG_SEED)

    # Discover per-subject directories
    outputs_dir = pipeline_dir / "outputs"
    subjects_info = []
    for sub_dir in sorted(outputs_dir.glob("sub-*")):
        epochs_dir = sub_dir / "derivatives" / "epochs_clean"
        if epochs_dir.exists():
            subjects_info.append((sub_dir.name, epochs_dir))
    
    if not subjects_info:
        print("No subjects found.")
        return
    
    subjects = [s for s, _ in subjects_info]
    # Use first subject's epochs_dir as default (backward compat)
    epochs_dir = subjects_info[0][1]

    print(f"Subjects: {subjects}")
    print(f"Blocks: {blocks}")
    print(f"Edges: {SYNC_EDGES}")
    print(f"Band: theta {THETA_BAND} Hz")
    print(f"Metric: debiased wPLI (Vinck 2011) + iCoh (Nolte 2004)")

    # Load node channels from config
    h1_nodes = cfg.get('h1_nodes', {})
    if not h1_nodes:
        print("ERROR: h1_nodes not found in config. Check study.yml.")
        return

    # Use h1_nodes directly
    all_h1_nodes = h1_nodes

    edge_nodes = {}
    for n1, n2 in SYNC_EDGES:
        c1 = all_h1_nodes.get(n1, [])
        c2 = all_h1_nodes.get(n2, [])
        if not c1 or not c2:
            print(f"  WARNING: missing node channels for {n1} or {n2}")
            continue
        edge_nodes[(n1, n2)] = (c1, c2)
        print(f"  {n1}: {c1}")
        print(f"  {n2}: {c2}")

    all_rows = []

    for subj_name, subj_epochs_dir in subjects_info:
        print(f"\n{'='*60}")
        print(f"  H3 wPLI Connectivity: {subj_name}")
        print(f"{'='*60}")

        results_per_block: dict[int, dict] = {}

        for block in blocks:
            print(f"\n  Block {block}:")
            # wPLI uses the stored clean PAC epochs directly.
            # It does not rebuild onset epochs or analyze the saved P3b epochs.
            epochs = load_block_epochs(subj_name, block, 'pac', subj_epochs_dir)
            if epochs is None:
                print(f"    No PAC epochs for {subj_name} block {block}")
                continue

            sfreq = epochs.info["sfreq"]
            times = epochs.times
            n_trials = len(epochs)
            print(f"    {n_trials} clean trials, sfreq={sfreq:.0f} Hz")

            has_proxy = _proxy_available(sfreq)
            block_results: dict[str, float] = {}

            for n1, n2 in SYNC_EDGES:
                edge_label = f"{n1}-{n2}"
                c1, c2 = edge_nodes[(n1, n2)]
                print(f"\n    Edge: {edge_label}")

                # Theta analytic signals
                ana_n1 = _node_analytic_timeseries(
                    epochs, c1, THETA_BAND, sfreq)
                ana_n2 = _node_analytic_timeseries(
                    epochs, c2, THETA_BAND, sfreq)

                if ana_n1 is None or ana_n2 is None:
                    print(f"      Could not extract analytic signals")
                    continue

                for tmin, tmax, wlabel in WINDOWS:
                    wpli, icoh, n_samples = _compute_connectivity(
                        ana_n1, ana_n2, times, tmin, tmax)

                    is_primary = (n1, n2) == ("C_broad_F", "C_broad_P")
                    if is_primary and wlabel == "stim":
                        role = "PRIMARY"
                    else:
                        role = "secondary"

                    print(f"      {wlabel} ({tmin}-{tmax}s): "
                          f"wPLI={wpli:.4f}, iCoh={icoh:.4f}, "
                          f"n={n_samples:,} [{role}]")

                    block_results[f"wpli_{edge_label}_{wlabel}"] = wpli
                    block_results[f"icoh_{edge_label}_{wlabel}"] = icoh
                    block_results[f"n_{edge_label}_{wlabel}"] = n_samples

                    all_rows.append({
                        "subject": subj_name,
                        "block": block,
                        "window": wlabel,
                        "edge": edge_label,
                        "wpli": wpli,
                        "icoh": icoh,
                        "n_samples": n_samples,
                        "role": role,
                    })

            results_per_block[block] = block_results

        # Per-subject figure
        if results_per_block:
            fig_path = _tag_path(
                f"theta_wpli_{subj_name}.png", FIG_DIR)
            _plot_subject_diagnostic(subj_name, results_per_block, fig_path)

    # ── Save subject-level CSV ────────────────────────────────────────
    if not all_rows:
        print("\nNo data collected.")
        return

    df = pd.DataFrame(all_rows)
    csv_path = _tag_path("theta_wpli_features.csv", FEAT_DIR)
    save_feature_tables(df, csv_path.name)
    print(f"\nSaved subject CSV: {csv_path}")

    # ── Group inference ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Group Inference")
    print("=" * 60)

    group_summary_rows = []

    for edge_label in [f"{n1}-{n2}" for n1, n2 in SYNC_EDGES]:
        for _, _, wlabel in WINDOWS:
            sub = df[(df["edge"] == edge_label) &
                     (df["window"] == wlabel)]
            if sub.empty:
                continue

            piv = sub.pivot(index="subject", columns="block",
                            values="wpli")
            bcols = sorted(piv.columns)
            if len(bcols) < 2:
                continue
            b1_w = piv[bcols[0]].values
            b5_w = piv[bcols[-1]].values
            valid = np.isfinite(b1_w) & np.isfinite(b5_w)
            deltas = (b5_w - b1_w)[valid]
            if len(deltas) < 2:
                continue

            test = _paired_test(b1_w, b5_w)
            mean_d, ci_lo, ci_hi = _bootstrap_ci(deltas, rng=rng)
            p_perm = _permutation_test(deltas, rng=rng)

            primary = (wlabel == "stim" and edge_label == "C_broad_F-C_broad_P")
            role = "PRIMARY" if primary else "DESCRIPTIVE"

            row_summary = {
                "edge": edge_label,
                "band": "theta",
                "window": wlabel,
                "role": role,
                "n_subjects": int(valid.sum()),
                "mean_delta_wpli": mean_d,
                "CI_lo": ci_lo,
                "CI_hi": ci_hi,
                "paired_test": test["test"],
                "paired_p": test["p"],
                "paired_stat": test["stat"],
                "permutation_p": p_perm,
            }
            group_summary_rows.append(row_summary)

            print(f"\n  {edge_label} | theta | {wlabel} [{role}]:")
            print(f"    n = {valid.sum()}")
            print(f"    mean delta wPLI = {mean_d:.4f} "
                  f"[{ci_lo:.4f}, {ci_hi:.4f}]")
            print(f"    {test['test']}: p = {test['p']:.4f}")
            print(f"    permutation: p = {p_perm:.4f}")

    if group_summary_rows:
        df_grp = pd.DataFrame(group_summary_rows)
        gcsv_path = _tag_path("theta_wpli_group.csv", FEAT_DIR)
        save_feature_tables(df_grp, gcsv_path.name, include_per_subject=False)
        print(f"\nSaved group CSV: {gcsv_path}")

    grp_fig_path = _tag_path("theta_wpli_group.png", GRP_DIR)
    _plot_group_figure(df, grp_fig_path, rng)

    print("\nDone.")


if __name__ == "__main__":
    main()
