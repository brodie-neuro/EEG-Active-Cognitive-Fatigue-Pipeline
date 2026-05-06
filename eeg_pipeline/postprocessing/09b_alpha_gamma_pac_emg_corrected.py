# eeg_pipeline/postprocessing/09b_alpha_gamma_pac_emg_corrected.py
"""
Step 09b - Alpha-gamma PAC with EMG-corrected gamma.

Sensitivity analysis for step 09. The analysis is identical to the
frontal-alpha -> parietal-gamma PAC workflow, except that trial-level EMG PC1 is
regressed out of the parietal amplitude-source signal at each time point before
gamma filtering and Hilbert amplitude extraction.

This script does not modify the primary alpha-gamma PAC features.

Inputs:
  - outputs/features/emg_covariates.csv from step 13
  - cleaned PAC epochs from step 07

Outputs:
  - outputs/features/alpha_gamma_pac_emg_corrected.csv
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MNE_DONTWRITE_HOME"] = "true"
os.environ.setdefault("_MNE_FAKE_HOME_DIR", os.path.dirname(os.path.dirname(__file__)))

import matplotlib

matplotlib.use("Agg")
import mne
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.signal import hilbert as sp_hilbert

pipeline_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pipeline_dir))

from src.utils_config import get_param
from src.utils_features import (
    available_channels,
    filter_excluded_channels,
    get_subjects_with_blocks,
    load_block_epochs,
)
from src.utils_io import discover_subjects, load_config


OUTPUT_DIR = pipeline_dir / "outputs" / "features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA_GAMMA_CFG = get_param("alpha_gamma_pac", default={}) or {}
PHASE_NODE = ALPHA_GAMMA_CFG.get("phase_node", "C_broad_F")
AMP_NODE = ALPHA_GAMMA_CFG.get("amp_node", "C_broad_P")
ALPHA_BAND = tuple(float(v) for v in ALPHA_GAMMA_CFG.get("phase_band", [8.0, 13.0]))
GAMMA_BAND = tuple(float(v) for v in ALPHA_GAMMA_CFG.get("amp_band", [55.0, 85.0]))
PAC_ANALYSIS_WINDOW = tuple(float(v) for v in ALPHA_GAMMA_CFG.get("analysis_window", [0.0, 0.6]))
N_SURROGATES = int(ALPHA_GAMMA_CFG.get("n_surrogates", 500))
N_BINS = int(ALPHA_GAMMA_CFG.get("n_bins", 12))
RANDOM_SEED = int(ALPHA_GAMMA_CFG.get("random_seed", 42))
H2_EXCLUDED_SUBJECTS = {
    str(subj) for subj in ALPHA_GAMMA_CFG.get("analysis_excluded_subjects", [])
}


def get_node_signal(epochs, node_channels: list[str]) -> np.ndarray | None:
    """Extract mean signal across available EEG channels in a node."""
    node_channels, _ = filter_excluded_channels(node_channels)
    picks = available_channels(node_channels, epochs.ch_names)
    if picks:
        types = epochs.get_channel_types(picks=picks)
        picks = [ch for ch, ch_type in zip(picks, types) if ch_type == "eeg"]
    if not picks:
        return None
    return epochs.copy().pick(picks).get_data().mean(axis=1)


def _band_phase(signal_1d: np.ndarray, sfreq: float, band: tuple[float, float]) -> np.ndarray:
    filt = mne.filter.filter_data(
        signal_1d,
        sfreq,
        band[0],
        band[1],
        verbose=False,
        method="iir",
        iir_params=dict(order=4, ftype="butter"),
    )
    return np.angle(sp_hilbert(filt))


def _gamma_amp(signal_1d: np.ndarray, sfreq: float, band: tuple[float, float]) -> np.ndarray:
    filt = mne.filter.filter_data(
        signal_1d,
        sfreq,
        band[0],
        band[1],
        verbose=False,
        method="fir",
    )
    return np.abs(sp_hilbert(filt))


def _regress_emg_from_signal(node_signal: np.ndarray, emg_pc1: np.ndarray) -> np.ndarray:
    """Regress EMG PC1 out of the node signal at each time point."""
    _, n_times = node_signal.shape
    corrected = np.zeros_like(node_signal)
    for time_idx in range(n_times):
        y = node_signal[:, time_idx]
        x = emg_pc1
        mask = np.isfinite(y) & np.isfinite(x)
        if int(mask.sum()) < 5:
            corrected[:, time_idx] = y
            continue
        slope, intercept, _, _, _ = sp_stats.linregress(x[mask], y[mask])
        residual = y - (intercept + slope * x)
        corrected[:, time_idx] = residual + np.mean(y)
    return corrected


def _modulation_index(phase: np.ndarray, amplitude: np.ndarray, n_bins: int = N_BINS) -> float:
    """Tort et al. modulation index."""
    phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    mean_amp = np.zeros(n_bins)
    for idx in range(n_bins):
        mask = (phase >= phase_bins[idx]) & (phase < phase_bins[idx + 1])
        if int(mask.sum()) > 0:
            mean_amp[idx] = amplitude[mask].mean()
    total = float(mean_amp.sum())
    if total <= 0:
        return 0.0
    probs = np.clip(mean_amp / total, 1e-30, None)
    uniform = np.ones(n_bins) / n_bins
    return float(np.sum(probs * np.log(probs / uniform)) / np.log(n_bins))


def _pac_from_precomputed(phase: np.ndarray, amplitude: np.ndarray, n_surrogates: int) -> tuple[float, dict]:
    """Trial-concatenated MI with circular-shift surrogate z-scoring."""
    n_epochs = min(phase.shape[0], amplitude.shape[0])
    if n_epochs == 0:
        return np.nan, {}

    concat_phase = phase[:n_epochs].ravel()
    concat_amp = amplitude[:n_epochs].ravel()
    mi_real = _modulation_index(concat_phase, concat_amp)

    n_samples = len(concat_phase)
    shift_range = n_samples // 4
    rng = np.random.RandomState(RANDOM_SEED)
    surrogate_mis = np.zeros(n_surrogates)
    for idx in range(n_surrogates):
        shift = rng.randint(shift_range, n_samples - shift_range)
        surrogate_mis[idx] = _modulation_index(concat_phase, np.roll(concat_amp, shift))

    mean_surr = float(np.mean(surrogate_mis))
    std_surr = float(np.std(surrogate_mis))
    z = float((mi_real - mean_surr) / std_surr) if std_surr > 0 else 0.0
    return z, {"mi_real": mi_real, "mean_surr": mean_surr, "std_surr": std_surr}


def _merge_existing(new_df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    new_df = _drop_h2_exclusions(new_df)
    if out_path.exists() and "subject" in new_df.columns:
        existing = pd.read_csv(out_path)
        existing = _drop_h2_exclusions(existing)
        new_subjects = set(new_df["subject"].astype(str))
        existing = existing[~existing["subject"].astype(str).isin(new_subjects)]
        return pd.concat([existing, new_df], ignore_index=True)
    return new_df


def _drop_h2_exclusions(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "subject" not in df.columns or not H2_EXCLUDED_SUBJECTS:
        return df
    return df[~df["subject"].astype(str).isin(H2_EXCLUDED_SUBJECTS)].copy()


def _print_original_comparison(corrected_df: pd.DataFrame) -> None:
    orig_path = OUTPUT_DIR / "alpha_gamma_pac_features.csv"
    if not orig_path.exists():
        return
    orig = pd.read_csv(orig_path)
    orig = _drop_h2_exclusions(orig)
    if "variant" in orig.columns:
        orig = orig[
            orig["variant"].isin(
                ["frontal_alpha_to_parietal_gamma", "frontal_alpha->parietal_gamma"]
            )
        ]
    comp = pd.merge(
        orig[["subject", "block", "z_score"]].rename(columns={"z_score": "pac_original"}),
        corrected_df[["subject", "block", "z_score"]].rename(columns={"z_score": "pac_corrected"}),
        on=["subject", "block"],
        how="inner",
    )
    if comp.empty:
        return
    comp["delta_corrected_minus_original"] = comp["pac_corrected"] - comp["pac_original"]
    print("\n" + "=" * 60)
    print("  Comparison: Original vs EMG-corrected alpha-gamma PAC")
    print("=" * 60)
    print(comp.to_string(index=False))
    if len(comp) >= 3 and comp["pac_original"].nunique() > 1 and comp["pac_corrected"].nunique() > 1:
        r, p = sp_stats.pearsonr(comp["pac_original"], comp["pac_corrected"])
        print(f"\n  Correlation original vs corrected: r={r:.3f}, p={p:.4f}")
    print(f"  Mean absolute delta: {comp['delta_corrected_minus_original'].abs().mean():.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Alpha-gamma PAC with EMG-corrected gamma (step 09b sensitivity analysis)."
    )
    parser.add_argument(
        "--subject",
        default="",
        help="Optional subject filter, e.g. sub-p001 or sub-p001,sub-p002.",
    )
    args = parser.parse_args()
    if args.subject.strip():
        os.environ["EEG_SUBJECT_FILTER"] = args.subject.strip()

    cfg = load_config()
    blocks = [int(b) for b in cfg.get("blocks", [1, 5])]
    h1_nodes = cfg.get("h1_nodes", {})
    if not h1_nodes:
        raise KeyError("Missing h1_nodes in study.yml.")

    emg_path = OUTPUT_DIR / "emg_covariates.csv"
    if not emg_path.exists():
        raise FileNotFoundError(f"{emg_path} not found. Run step 13 first.")
    emg_df = pd.read_csv(emg_path)

    print(
        "  Alpha-gamma EMG sensitivity params: "
        f"phase={list(ALPHA_BAND)}, amp={list(GAMMA_BAND)}, "
        f"window={PAC_ANALYSIS_WINDOW}, surrogates={N_SURROGATES}"
    )

    epochs_dir = pipeline_dir / "outputs" / "derivatives" / "epochs_clean"
    subjects = get_subjects_with_blocks(epochs_dir, "pac", blocks)
    if not subjects:
        subjects = discover_subjects(
            epochs_dir=epochs_dir,
            blocks=blocks,
            epoch_type="pac",
            require_all_blocks=False,
        )
    if not subjects:
        print("No PAC epoch files found.")
        return
    if H2_EXCLUDED_SUBJECTS:
        before = list(subjects)
        subjects = [subj for subj in subjects if str(subj) not in H2_EXCLUDED_SUBJECTS]
        removed = [subj for subj in before if str(subj) in H2_EXCLUDED_SUBJECTS]
        if removed:
            print(
                "  Applying H2 alpha-gamma exclusions to EMG-corrected PAC: "
                + ", ".join(sorted(removed))
            )
        if not subjects:
            print("No PAC epoch files remain after H2 alpha-gamma exclusions.")
            return

    rows: list[dict] = []
    phase_channels = h1_nodes.get(PHASE_NODE, [])
    amp_channels = h1_nodes.get(AMP_NODE, [])

    for subj in subjects:
        print(f"\n{'=' * 60}")
        print(f"  Alpha-gamma EMG-corrected PAC: {subj}")
        print(f"{'=' * 60}")
        for block in blocks:
            print(f"\n  --- Block {block} ---")
            epochs = load_block_epochs(subj, block, "pac", epochs_dir)
            if epochs is None or "eeg" not in epochs.get_channel_types():
                print(f"  No PAC data for block {block}")
                continue

            emg_block = (
                emg_df[(emg_df["subject"] == subj) & (emg_df["block"] == block)]
                .sort_values("trial")
                .reset_index(drop=True)
            )
            if emg_block.empty:
                print(f"  No EMG data for block {block}")
                continue

            phase_signal = get_node_signal(epochs, phase_channels)
            amp_signal = get_node_signal(epochs, amp_channels)
            if phase_signal is None or amp_signal is None:
                print(f"  Missing node signals for {PHASE_NODE}->{AMP_NODE}")
                continue

            n_epochs = min(phase_signal.shape[0], amp_signal.shape[0], len(emg_block))
            phase_signal = phase_signal[:n_epochs]
            amp_signal = amp_signal[:n_epochs]
            emg_pc1 = emg_block["emg_pc1"].to_numpy(dtype=float)[:n_epochs]
            print(f"  {n_epochs} trials matched (epochs + EMG)")

            amp_corrected = _regress_emg_from_signal(amp_signal, emg_pc1)
            sfreq = float(epochs.info["sfreq"])
            times = epochs.times
            crop_mask = (times >= PAC_ANALYSIS_WINDOW[0]) & (times <= PAC_ANALYSIS_WINDOW[1])
            if int(crop_mask.sum()) < 10:
                print("  Too few samples after PAC-window crop.")
                continue

            alpha_phase = np.zeros((n_epochs, phase_signal.shape[1]))
            gamma_amplitude = np.zeros((n_epochs, amp_corrected.shape[1]))
            for ep_idx in range(n_epochs):
                alpha_phase[ep_idx] = _band_phase(phase_signal[ep_idx], sfreq, ALPHA_BAND)
                gamma_amplitude[ep_idx] = _gamma_amp(amp_corrected[ep_idx], sfreq, GAMMA_BAND)

            alpha_phase = alpha_phase[:, crop_mask]
            gamma_amplitude = gamma_amplitude[:, crop_mask]
            z, details = _pac_from_precomputed(alpha_phase, gamma_amplitude, N_SURROGATES)
            print(f"  EMG-corrected alpha-gamma PAC: z={z:.3f}")

            rows.append(
                {
                    "subject": subj,
                    "block": block,
                    "variant": "frontal_alpha_to_parietal_gamma_emg_corrected",
                    "role": "SENSITIVITY",
                    "z_score": z,
                    "mi_real": details.get("mi_real", np.nan),
                    "mean_surr": details.get("mean_surr", np.nan),
                    "std_surr": details.get("std_surr", np.nan),
                    "n_trials": int(n_epochs),
                    "phase_band": list(ALPHA_BAND),
                    "amp_band": list(GAMMA_BAND),
                    "window": f"{PAC_ANALYSIS_WINDOW[0]}-{PAC_ANALYSIS_WINDOW[1]}",
                    "n_surrogates": N_SURROGATES,
                }
            )

    if rows:
        out = OUTPUT_DIR / "alpha_gamma_pac_emg_corrected.csv"
        df = _merge_existing(pd.DataFrame(rows), out)
        df.to_csv(out, index=False)
        print(f"\nSaved EMG-corrected alpha-gamma PAC to {out}")
        print(df.to_string(index=False))
        _print_original_comparison(df)


if __name__ == "__main__":
    main()
