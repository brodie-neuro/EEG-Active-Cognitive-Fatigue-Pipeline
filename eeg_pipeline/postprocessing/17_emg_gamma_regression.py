# eeg_pipeline/analysis/20d_emg_gamma_regression.py
"""
EMG-corrected gamma: regress out EMG PC1 from scalp gamma power.

Pipeline position: runs AFTER 19_gamma_power and 20c_emg_pca_covariates.

For each block, loads:
  - Per-trial gamma envelope (55-85 Hz) at ROI channels (from step 19)
  - Per-trial EMG PC1 score (from step 20c)

Performs OLS regression:
  gamma_trial = β₀ + β₁ × emg_pc1 + residual

Outputs:
  - CSV: gamma_emg_corrected.csv (trial-level corrected gamma + raw + beta)
  - CSV: gamma_emg_regression_summary.csv (block-level R², beta, p-value)
  - Figure: emg_gamma_regression_{subj}.png (publication-quality, 4 panels)

The residuals are the EMG-corrected gamma power, usable in PAC and group-level
analyses. Single-predictor OLS — no overfitting risk.

References:
  - McMenamin et al., 2011, NeuroImage
  - Muthukumaraswamy, 2013, NeuroImage
"""
import os
import sys
from pathlib import Path

pipeline_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pipeline_dir))

from src.utils_determinism import file_sha256, save_step_qc, set_determinism_env, set_random_seeds

set_determinism_env()
set_random_seeds()

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from src.utils_io import load_config, iter_derivative_files
from src.utils_features import (
    available_channels, get_node_channels,
)

OUTPUT_DIR = pipeline_dir / "outputs" / "features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = pipeline_dir / "outputs" / "analysis_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_TAG = os.environ.get("EEG_OUTPUT_TAG", "").strip()


def _tag_path(base: str) -> Path:
    s, ext = Path(base).stem, Path(base).suffix
    if OUTPUT_TAG:
        s = f"{s}_{OUTPUT_TAG}"
    return FIG_DIR / f"{s}{ext}"


# ---------------------------------------------------------------------------
# Load per-trial data
# ---------------------------------------------------------------------------

def _load_emg_covariates():
    """Load trial-level EMG PC1 from step 20c."""
    f = OUTPUT_DIR / "emg_covariates.csv"
    if not f.exists():
        raise FileNotFoundError(f"{f} not found. Run step 20c first.")
    return pd.read_csv(f)


def _load_gamma_node_features():
    """Load trial-level gamma from step 19 (node features)."""
    f = OUTPUT_DIR / "gamma_node_features.csv"
    if not f.exists():
        raise FileNotFoundError(f"{f} not found. Run step 19 first.")
    return pd.read_csv(f)


def _load_gamma_stim_features():
    """Load block-level gamma from step 19 (stim features)."""
    f = OUTPUT_DIR / "gamma_stim_features.csv"
    if not f.exists():
        raise FileNotFoundError(f"{f} not found. Run step 19 first.")
    return pd.read_csv(f)


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

def _regress_emg_from_gamma(gamma_trials, emg_pc1_trials):
    """Single-predictor OLS: gamma = β₀ + β₁×emg_pc1 + ε.

    Returns dict with corrected (residual) gamma, beta, R², p-value.
    """
    # Remove NaN pairs
    mask = np.isfinite(gamma_trials) & np.isfinite(emg_pc1_trials)
    if mask.sum() < 5:
        raise RuntimeError("Regression requires at least 5 valid gamma/EMG trial pairs.")

    g = gamma_trials[mask]
    e = emg_pc1_trials[mask]

    slope, intercept, r_value, p_value, std_err = stats.linregress(e, g)
    predicted = intercept + slope * emg_pc1_trials
    residuals = gamma_trials - predicted

    # Corrected gamma = residuals + grand mean (preserve original scale)
    grand_mean = np.nanmean(gamma_trials)
    corrected = residuals + grand_mean

    return {
        "corrected": corrected,
        "residuals": residuals,
        "beta": slope,
        "intercept": intercept,
        "r_squared": r_value**2,
        "r_value": r_value,
        "p_value": p_value,
        "std_err": std_err,
        "n_valid": int(mask.sum()),
        "grand_mean": grand_mean,
    }


# ---------------------------------------------------------------------------
# Publication figure
# ---------------------------------------------------------------------------

def _plot_regression(subj, block_results, sorted_blocks):
    """4-panel figure per Nature Human Behaviour style.

    Panel A: Scatter + regression line (gamma vs EMG PC1) per block
    Panel B: Before/after distributions (raw vs corrected gamma)
    Panel C: Trial-by-trial gamma timecourse (raw vs corrected)
    Panel D: Regression summary table
    """
    fig = plt.figure(figsize=(16, 14))

    # Colour scheme
    colors = {1: "#1565C0", 5: "#C62828"}
    colors_light = {1: "#90CAF9", 5: "#EF9A9A"}
    block_labels = {1: "Block 1 (Baseline)", 5: "Block 5 (Fatigued)"}

    # --- Panel A: Scatter + regression line ---
    ax_a = fig.add_subplot(2, 2, 1)

    for block in sorted_blocks:
        r = block_results.get(block)
        if r is None:
            continue
        c = colors[block]
        cl = colors_light[block]
        bl = block_labels.get(block, f"Block {block}")

        mask = np.isfinite(r["gamma_raw"]) & np.isfinite(r["emg_pc1"])
        g = r["gamma_raw"][mask]
        e = r["emg_pc1"][mask]

        ax_a.scatter(e, g, s=20, alpha=0.45, color=cl, edgecolors=c,
                     linewidths=0.5, zorder=2)

        # Regression line
        reg = r["regression"]
        x_line = np.linspace(e.min(), e.max(), 100)
        y_line = reg["intercept"] + reg["beta"] * x_line
        ax_a.plot(x_line, y_line, color=c, lw=2.5, zorder=3,
                  label=f'{bl}\nβ={reg["beta"]:.3f}, R²={reg["r_squared"]:.3f}'
                        f'\np={reg["p_value"]:.4f}')

    ax_a.set_xlabel("EMG PC1 Score (Global Muscle Tension)", fontsize=11)
    ax_a.set_ylabel("Raw Gamma Envelope (µV)", fontsize=11)
    ax_a.set_title("A. Gamma–EMG Relationship", fontsize=13,
                   fontweight="bold", pad=10)
    ax_a.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax_a.grid(True, alpha=0.2)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)

    # --- Panel B: Violin/box before vs after ---
    ax_b = fig.add_subplot(2, 2, 2)
    positions = []
    labels = []
    pos = 0

    for block in sorted_blocks:
        r = block_results.get(block)
        if r is None:
            continue
        c = colors[block]
        bl = f"B{block}"

        raw = r["gamma_raw"][np.isfinite(r["gamma_raw"])]
        corr = r["gamma_corrected"][np.isfinite(r["gamma_corrected"])]

        # Raw
        vp1 = ax_b.violinplot(raw, positions=[pos], showmedians=True,
                               showextrema=False, widths=0.7)
        for body in vp1["bodies"]:
            body.set_facecolor(c)
            body.set_alpha(0.3)
        vp1["cmedians"].set_color(c)

        # Corrected
        vp2 = ax_b.violinplot(corr, positions=[pos + 1], showmedians=True,
                               showextrema=False, widths=0.7)
        for body in vp2["bodies"]:
            body.set_facecolor(c)
            body.set_alpha(0.6)
        vp2["cmedians"].set_color(c)

        positions.extend([pos, pos + 1])
        labels.extend([f"{bl}\nRaw", f"{bl}\nCorrected"])
        pos += 3

    ax_b.set_xticks(positions)
    ax_b.set_xticklabels(labels, fontsize=9)
    ax_b.set_ylabel("Gamma Envelope (µV)", fontsize=11)
    ax_b.set_title("B. Raw vs EMG-Corrected Gamma", fontsize=13,
                   fontweight="bold", pad=10)
    ax_b.grid(True, alpha=0.2, axis="y")
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    # --- Panel C: Trial timecourse ---
    ax_c = fig.add_subplot(2, 2, 3)

    for block in sorted_blocks:
        r = block_results.get(block)
        if r is None:
            continue
        c = colors[block]
        cl = colors_light[block]
        bl = block_labels.get(block, f"Block {block}")
        n = len(r["gamma_raw"])
        trials = np.arange(1, n + 1)

        ax_c.plot(trials, r["gamma_raw"], color=cl, lw=0.8, alpha=0.6)
        ax_c.plot(trials, r["gamma_corrected"], color=c, lw=1.2, alpha=0.8,
                  label=f"{bl} corrected")

    ax_c.set_xlabel("Trial Number", fontsize=11)
    ax_c.set_ylabel("Gamma Envelope (µV)", fontsize=11)
    ax_c.set_title("C. Trial-by-Trial Timecourse", fontsize=13,
                   fontweight="bold", pad=10)
    ax_c.legend(fontsize=8, loc="upper right")
    ax_c.grid(True, alpha=0.2)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)

    # --- Panel D: Summary table ---
    ax_d = fig.add_subplot(2, 2, 4)
    ax_d.axis("off")

    col_labels = ["Block", "n", "β (slope)", "R²", "p-value",
                  "Raw γ (µV)", "Corrected γ (µV)", "Δ Variance"]
    table_data = []
    for block in sorted_blocks:
        r = block_results.get(block)
        if r is None:
            continue
        reg = r["regression"]
        raw = r["gamma_raw"][np.isfinite(r["gamma_raw"])]
        corr = r["gamma_corrected"][np.isfinite(r["gamma_corrected"])]
        var_reduction = (1 - np.var(corr) / np.var(raw)) * 100 if np.var(raw) > 0 else 0

        table_data.append([
            f"B{block}",
            f"{reg['n_valid']}",
            f"{reg['beta']:.4f}",
            f"{reg['r_squared']:.4f}",
            f"{reg['p_value']:.4f}" if reg["p_value"] >= 0.0001 else "<.0001",
            f"{np.mean(raw):.3f} ± {np.std(raw):.3f}",
            f"{np.mean(corr):.3f} ± {np.std(corr):.3f}",
            f"{var_reduction:+.1f}%",
        ])

    tbl = ax_d.table(cellText=table_data, colLabels=col_labels,
                     loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 2.0)
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#E3F2FD")
        tbl[0, j].set_text_props(fontweight="bold", fontsize=9)

    ax_d.set_title("D. Regression Summary", fontsize=13,
                   fontweight="bold", pad=10)

    fig.suptitle(
        f"EMG Gamma Decontamination — {subj}\n"
        f"OLS: γ = β₀ + β₁ × EMG_PC1 + ε  |  "
        f"Corrected γ = residual + grand mean",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = _tag_path(f"emg_gamma_regression_{subj}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n  Saved figure: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = load_config()
    blocks = cfg.get("blocks", [1, 5])

    emg_df = _load_emg_covariates()
    gamma_node_df = _load_gamma_node_features()
    gamma_stim_df = _load_gamma_stim_features()

    subjects = sorted(emg_df["subject"].unique())
    sorted_blocks = sorted(blocks)

    all_trial_rows = []
    all_summary_rows = []
    qc_records = []

    for subj in subjects:
        print(f"\n{'='*60}")
        print(f"  EMG-Gamma Regression: {subj}")
        print(f"{'='*60}")

        block_results = {}

        for block in sorted_blocks:
            # Get EMG PC1 for this block
            emg_block = emg_df[
                (emg_df["subject"] == subj) & (emg_df["block"] == block)
            ].sort_values("trial").reset_index(drop=True)

            if emg_block.empty:
                raise RuntimeError(f"Block {block}: no EMG data for {subj}.")

            # Get gamma node features for this block (pre-motor window)
            gamma_block = gamma_node_df[
                (gamma_node_df["subject"] == subj) &
                (gamma_node_df["block"] == block) &
                (gamma_node_df["window"].str.contains("0.0-0.6"))
            ].reset_index(drop=True)

            if gamma_block.empty:
                # Try getting from stim features instead
                gamma_block = gamma_stim_df[
                    (gamma_stim_df["subject"] == subj) &
                    (gamma_stim_df["block"] == block) &
                    (gamma_stim_df["window_s"].str.contains("0.0-0.6"))
                ].reset_index(drop=True)

            if gamma_block.empty:
                raise RuntimeError(f"Block {block}: no gamma features for {subj}.")

            # Need trial-level gamma — check if gamma_node has per-trial
            # If not, we need to recompute from epochs
            gamma_mean = gamma_block["gamma_env_mean"].values[0] if "gamma_env_mean" in gamma_block.columns else np.nan
            print(f"\n  Block {block}: {len(emg_block)} EMG trials, "
                  f"block γ mean={gamma_mean:.4f} µV")

            # The gamma_node_features is block-level, not trial-level
            # We need to recompute trial-level gamma from epochs
            # Use the same method as step 19
            gamma_trials = _compute_trial_gamma(subj, block, cfg)

            emg_pc1 = emg_block["emg_pc1"].values

            # Match lengths (trim to minimum)
            n_min = min(len(gamma_trials), len(emg_pc1))
            gamma_trials = gamma_trials[:n_min]
            emg_pc1 = emg_pc1[:n_min]

            print(f"  Block {block}: {n_min} matched trials")
            print(f"    Raw gamma: mean={np.nanmean(gamma_trials):.4f}, "
                  f"std={np.nanstd(gamma_trials):.4f} µV")
            print(f"    EMG PC1:   mean={np.nanmean(emg_pc1):.4f}, "
                  f"std={np.nanstd(emg_pc1):.4f}")

            # Regress
            reg = _regress_emg_from_gamma(gamma_trials, emg_pc1)

            print(f"    β = {reg['beta']:.4f} (SE={reg['std_err']:.4f})")
            print(f"    R² = {reg['r_squared']:.4f}")
            print(f"    p = {reg['p_value']:.4f}")
            print(f"    Corrected gamma: mean={np.nanmean(reg['corrected']):.4f}, "
                  f"std={np.nanstd(reg['corrected']):.4f} µV")

            block_results[block] = {
                "gamma_raw": gamma_trials,
                "gamma_corrected": reg["corrected"],
                "emg_pc1": emg_pc1,
                "regression": reg,
            }

            # Save trial-level
            for i in range(n_min):
                all_trial_rows.append({
                    "subject": subj,
                    "block": block,
                    "trial": i + 1,
                    "gamma_raw": gamma_trials[i],
                    "emg_pc1": emg_pc1[i],
                    "gamma_corrected": reg["corrected"][i],
                })

            # Save block-level summary
            all_summary_rows.append({
                "subject": subj,
                "block": block,
                "n_trials": reg["n_valid"],
                "beta": reg["beta"],
                "beta_se": reg["std_err"],
                "r_squared": reg["r_squared"],
                "r_value": reg["r_value"],
                "p_value": reg["p_value"],
                "gamma_raw_mean": float(np.nanmean(gamma_trials)),
                "gamma_raw_std": float(np.nanstd(gamma_trials)),
                "gamma_corrected_mean": float(np.nanmean(reg["corrected"])),
                "gamma_corrected_std": float(np.nanstd(reg["corrected"])),
            })
            qc_records.append({
                "subject": subj,
                "block": block,
                "n_trials": int(reg["n_valid"]),
                "beta": float(reg["beta"]),
                "r_squared": float(reg["r_squared"]),
                "p_value": float(reg["p_value"]),
            })

        # Plot
        if block_results:
            _plot_regression(subj, block_results, sorted_blocks)

    # Save CSVs
    if all_trial_rows:
        df_trials = pd.DataFrame(all_trial_rows)
        out_t = OUTPUT_DIR / "gamma_emg_corrected.csv"
        df_trials.to_csv(out_t, index=False)
        print(f"\n  Saved trial data: {out_t}")

    if all_summary_rows:
        df_summary = pd.DataFrame(all_summary_rows)
        out_s = OUTPUT_DIR / "gamma_emg_regression_summary.csv"
        df_summary.to_csv(out_s, index=False)
        print(f"  Saved summary: {out_s}")
        for record in qc_records:
            save_step_qc(
                "17_emg_gamma_regression",
                record["subject"],
                record["block"],
                {
                    "status": "PASS",
                    "input_file": [
                        str(OUTPUT_DIR / "emg_covariates.csv"),
                        str(OUTPUT_DIR / "gamma_node_features.csv"),
                        str(OUTPUT_DIR / "gamma_stim_features.csv"),
                    ],
                    "input_hash": {
                        "emg_covariates": file_sha256(OUTPUT_DIR / "emg_covariates.csv"),
                        "gamma_node_features": file_sha256(OUTPUT_DIR / "gamma_node_features.csv"),
                        "gamma_stim_features": file_sha256(OUTPUT_DIR / "gamma_stim_features.csv"),
                    },
                    "output_file": str(out_s),
                    "output_hash": file_sha256(out_s),
                    "parameters_used": {},
                    "step_specific": {
                        "n_trials": record["n_trials"],
                        "beta": record["beta"],
                        "r_squared": record["r_squared"],
                        "p_value": record["p_value"],
                    },
                },
            )


# ---------------------------------------------------------------------------
# Trial-level gamma from epochs (recomputes from ICA-cleaned data)
# ---------------------------------------------------------------------------

def _compute_trial_gamma(subj, block, cfg):
    """Compute per-trial gamma envelope at node channels.

    Uses same procedure as step 19: edge-safe Butterworth 55-85 Hz,
    Hilbert envelope, crop to 0-0.6s, mean across node channels.
    """
    import mne
    from scipy.signal import hilbert as sp_hilbert

    raw_dir = pipeline_dir / "outputs" / "derivatives" / "ica_cleaned_raw"
    epochs_dir = pipeline_dir / "outputs" / "derivatives" / "epochs_clean"

    # Find ICA-cleaned raw (new per-subject layout + legacy flat)
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

    # Match to clean P3b epochs (same trial selection as step 19)
    from src.utils_features import load_block_epochs
    clean_p3b = load_block_epochs(subj, block, "p3b", epochs_dir)
    if clean_p3b is not None:
        onset_codes = set(clean_p3b.events[:, 2])
        clean_samples = set(clean_p3b.events[:, 0])
    else:
        onset_codes = {1}
        clean_samples = None

    onset_dict = {k: v for k, v in event_id.items() if v in onset_codes}
    if not onset_dict:
        raise RuntimeError(f"No onset events found for {subj} block {block}.")

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

    # Get node channels
    node_chs = get_node_channels("CF", cfg)
    avail = available_channels(node_chs, epochs.ch_names)
    if not avail:
        raise ValueError(f"No CF node channels available in epochs for {subj} block {block}. Check node config.")

    data = epochs.copy().pick(avail).get_data()  # (n_ep, n_ch, n_times)
    roi_data = data.mean(axis=1)  # (n_ep, n_times)
    sfreq = epochs.info["sfreq"]
    times = epochs.times

    # Edge-safe: filter full epoch, crop to 0-0.6s for envelope
    crop_mask = (times >= 0.0) & (times <= 0.6)
    n_trials = roi_data.shape[0]
    envelopes = np.full(n_trials, np.nan)

    for i in range(n_trials):
            filtered = mne.filter.filter_data(
                roi_data[i].astype(np.float64), sfreq,
                l_freq=55.0, h_freq=85.0,
                method="iir",
                iir_params=dict(order=4, ftype="butter"),
                verbose=False,
            )
            env = np.abs(sp_hilbert(filtered))
            envelopes[i] = float(np.mean(env[crop_mask])) * 1e6  # µV

    valid = np.isfinite(envelopes).sum()
    print(f"    Computed {valid}/{n_trials} trial gamma envelopes at {avail}")
    return envelopes


if __name__ == "__main__":
    main()
