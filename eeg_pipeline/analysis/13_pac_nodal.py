because# eeg_pipeline/analysis/13_pac_nodal.py
"""
Step 13 -- Phase-Amplitude Coupling Analysis (H1)

Primary: Between-region PAC -- C_broad_F theta phase x C_broad_P gamma amplitude
Descriptive: Fz_desc -> Pz_desc single-channel pair

Recovered March 8 PAC logic:
- fixed 4-8 Hz theta phase
- fixed 55-85 Hz gamma amplitude
- buffered PAC epochs, with post-filter crop to 0.0-0.6 s
- trial-concatenated surrogate z-scoring for short epochs

Outputs long format: one row per subject x block.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd

pipeline_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pipeline_dir))

from src.utils_config import get_param
from src.utils_features import (
    available_channels,
    get_subjects_with_blocks,
    load_block_epochs,
    save_feature_tables,
)
from src.utils_io import load_config

try:
    from tensorpac import Pac

    TENSORPAC_AVAILABLE = True
except ImportError:
    TENSORPAC_AVAILABLE = False
    print("Warning: tensorpac not installed. Install with: pip install tensorpac")


OUTPUT_DIR = pipeline_dir / "outputs" / "features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_TAG = os.environ.get("EEG_OUTPUT_TAG", "").strip()

H1_PAIRS = [("C_broad_F", "C_broad_P")]
DESCRIPTIVE_PAIRS = [("Fz_desc", "Pz_desc")]
BETWEEN_PAC_PAIRS = H1_PAIRS + DESCRIPTIVE_PAIRS
PAC_ANALYSIS_WINDOW = (0.0, 0.6)


def _tag_name(name: str) -> str:
    p = Path(name)
    if not OUTPUT_TAG:
        return p.name
    return f"{p.stem}_{OUTPUT_TAG}{p.suffix}"


def _find_subjects(epochs_dir: Path, blocks: list[int], epoch_type: str) -> list[str]:
    subjects = get_subjects_with_blocks(epochs_dir, epoch_type, blocks)
    if subjects:
        return subjects

    seen = set()
    for path in sorted(epochs_dir.glob(f"*_block*_{epoch_type}_clean-epo.fif")):
        stem = path.name.split("_block")[0]
        if stem.startswith("sub-"):
            seen.add(stem)
    return sorted(seen)


def _filter_excluded_channels(ch_names: list[str]) -> tuple[list[str], list[str]]:
    excluded = set(get_param("exclude_channels_analysis", "channels", default=[]) or [])
    kept = [ch for ch in ch_names if ch not in excluded]
    removed = [ch for ch in ch_names if ch in excluded]
    return kept, removed


def _pac_phase_band() -> list[float]:
    pac_cfg = get_param("pac", default={}) or {}
    return pac_cfg.get("phase_band", pac_cfg.get("theta_phase_band", [4, 8]))


def _pac_amp_band() -> list[float]:
    pac_cfg = get_param("pac", default={}) or {}
    return pac_cfg.get("amp_band", [55, 85])


def _pac_n_surrogates() -> int:
    pac_cfg = get_param("pac", default={}) or {}
    return int(pac_cfg.get("n_surrogates", pac_cfg.get("surrogates", 500)))


def _pac_seed() -> int:
    pac_cfg = get_param("pac", default={}) or {}
    return int(pac_cfg.get("random_seed", 42))


def get_node_signal(epochs: mne.Epochs, node_channels: list[str]) -> np.ndarray | None:
    """Return mean node signal as (n_epochs, n_times), EEG only."""
    node_channels, _ = _filter_excluded_channels(node_channels)
    avail = available_channels(node_channels, epochs.ch_names)
    if avail:
        types = epochs.get_channel_types(picks=avail)
        avail = [ch for ch, ch_type in zip(avail, types) if ch_type == "eeg"]
    if not avail:
        return None
    data = epochs.copy().pick(avail).get_data()
    return data.mean(axis=1)


def _theta_phase(signal_1d: np.ndarray, sfreq: float, lo: float, hi: float) -> np.ndarray:
    from scipy.signal import hilbert as sp_hilbert

    filt = mne.filter.filter_data(
        signal_1d,
        sfreq,
        lo,
        hi,
        verbose=False,
        method="iir",
        iir_params=dict(order=4, ftype="butter"),
    )
    return np.angle(sp_hilbert(filt))


def _gamma_amp(signal_1d: np.ndarray, sfreq: float, lo: float, hi: float) -> np.ndarray:
    from scipy.signal import hilbert as sp_hilbert

    filt = mne.filter.filter_data(
        signal_1d,
        sfreq,
        lo,
        hi,
        verbose=False,
        method="fir",
    )
    return np.abs(sp_hilbert(filt))


def _modulation_index(theta_phase: np.ndarray, gamma_amp: np.ndarray, n_bins: int = 12) -> float:
    phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    mean_amp = np.zeros(n_bins, dtype=float)
    for idx in range(n_bins):
        mask = (theta_phase >= phase_bins[idx]) & (theta_phase < phase_bins[idx + 1])
        if mask.sum() > 0:
            mean_amp[idx] = gamma_amp[mask].mean()
    if mean_amp.sum() == 0:
        return 0.0
    mean_amp = mean_amp / mean_amp.sum()
    mean_amp = np.clip(mean_amp, 1e-10, None)
    uniform = np.ones(n_bins, dtype=float) / n_bins
    kl_div = np.sum(mean_amp * np.log(mean_amp / uniform))
    return float(kl_div / np.log(n_bins))


def compute_pac_tensorpac(
    phase_signal: np.ndarray,
    amp_signal: np.ndarray,
    sfreq: float,
    phase_band: list[float],
    amp_band: list[float],
) -> float:
    n_surr = _pac_n_surrogates()
    seed = _pac_seed()

    p = Pac(
        idpac=(2, 1, 1),
        f_pha=phase_band,
        f_amp=amp_band,
        dcomplex="hilbert",
        verbose=False,
    )

    np.random.seed(seed)
    try:
        pac_vals = p.filterfit(sfreq, phase_signal, amp_signal, n_perm=n_surr)
        pac_mean = float(np.nanmean(pac_vals))
        if np.isnan(pac_mean):
            return compute_pac_fallback(phase_signal, amp_signal, sfreq, phase_band, amp_band)
        return pac_mean
    except Exception as exc:
        print(f"  tensorpac failed: {exc}")
        return compute_pac_fallback(phase_signal, amp_signal, sfreq, phase_band, amp_band)


def compute_pac_fallback(
    phase_signal: np.ndarray,
    amp_signal: np.ndarray,
    sfreq: float,
    phase_band: list[float],
    amp_band: list[float],
) -> float:
    """Trial-concatenated Tort MI with circular-shift surrogate z-scoring."""
    n_epochs = phase_signal.shape[0]
    n_surr = _pac_n_surrogates()
    seed = _pac_seed()

    all_phase = []
    all_amp = []
    for ep_idx in range(n_epochs):
        try:
            phase = _theta_phase(phase_signal[ep_idx], sfreq, phase_band[0], phase_band[1])
            amp = _gamma_amp(amp_signal[ep_idx], sfreq, amp_band[0], amp_band[1])
            all_phase.append(phase)
            all_amp.append(amp)
        except Exception:
            continue

    if not all_phase:
        return np.nan

    concat_phase = np.concatenate(all_phase)
    concat_amp = np.concatenate(all_amp)
    mi_real = _modulation_index(concat_phase, concat_amp)

    n_samples = len(concat_phase)
    shift_range = n_samples // 4
    if shift_range <= 0 or (n_samples - shift_range) <= shift_range:
        return np.nan

    rng = np.random.RandomState(seed)
    surr_mis = []
    for _ in range(n_surr):
        shift = rng.randint(shift_range, n_samples - shift_range)
        shifted_amp = np.roll(concat_amp, shift)
        surr_mis.append(_modulation_index(concat_phase, shifted_amp))

    mean_surr = float(np.mean(surr_mis))
    std_surr = float(np.std(surr_mis))
    return (mi_real - mean_surr) / std_surr if std_surr > 0 else 0.0


def _precompute_phase_amp(
    node_signal: np.ndarray,
    sfreq: float,
    phase_band: list[float],
    amp_band: list[float],
    times: np.ndarray | None = None,
    crop_window: tuple[float, float] | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    n_epochs = node_signal.shape[0]
    if n_epochs == 0:
        return None, None

    theta_ph = np.zeros((n_epochs, node_signal.shape[1]), dtype=float)
    gamma_am = np.zeros((n_epochs, node_signal.shape[1]), dtype=float)
    for ep_idx in range(n_epochs):
        theta_ph[ep_idx] = _theta_phase(node_signal[ep_idx], sfreq, phase_band[0], phase_band[1])
        gamma_am[ep_idx] = _gamma_amp(node_signal[ep_idx], sfreq, amp_band[0], amp_band[1])

    if crop_window is not None:
        crop_tmin, crop_tmax = crop_window
        if times is not None:
            mask = (times >= crop_tmin) & (times <= crop_tmax)
            if int(mask.sum()) < 10:
                return None, None
            theta_ph = theta_ph[:, mask]
            gamma_am = gamma_am[:, mask]
        else:
            start = max(0, int(round(crop_tmin * sfreq)))
            stop = max(start + 1, int(round(crop_tmax * sfreq)) + 1)
            stop = min(stop, theta_ph.shape[1])
            theta_ph = theta_ph[:, start:stop]
            gamma_am = gamma_am[:, start:stop]
            if theta_ph.shape[1] < 10:
                return None, None

    return theta_ph, gamma_am


def _pac_from_precomputed(
    theta_phase: np.ndarray,
    gamma_amp: np.ndarray,
    sfreq: float,
    phase_band: list[float],
    amp_band: list[float],
) -> float:
    n_epochs = min(theta_phase.shape[0], gamma_amp.shape[0])
    if n_epochs == 0:
        return np.nan

    if theta_phase.shape[1] <= 600:
        concat_phase = theta_phase[:n_epochs].ravel()
        concat_amp = gamma_amp[:n_epochs].ravel()
        mi_real = _modulation_index(concat_phase, concat_amp)

        n_surr = _pac_n_surrogates()
        seed = _pac_seed()
        n_samples = len(concat_phase)
        shift_range = n_samples // 4
        if shift_range <= 0 or (n_samples - shift_range) <= shift_range:
            return np.nan

        rng = np.random.RandomState(seed)
        surr_mis = []
        for _ in range(n_surr):
            shift = rng.randint(shift_range, n_samples - shift_range)
            shifted_amp = np.roll(concat_amp, shift)
            surr_mis.append(_modulation_index(concat_phase, shifted_amp))

        mean_surr = float(np.mean(surr_mis))
        std_surr = float(np.std(surr_mis))
        return (mi_real - mean_surr) / std_surr if std_surr > 0 else 0.0

    if TENSORPAC_AVAILABLE:
        return compute_pac_tensorpac(
            theta_phase[:n_epochs],
            gamma_amp[:n_epochs],
            sfreq,
            phase_band,
            amp_band,
        )
    return compute_pac_fallback(
        theta_phase[:n_epochs],
        gamma_amp[:n_epochs],
        sfreq,
        phase_band,
        amp_band,
    )


def main() -> None:
    cfg = load_config()
    blocks = cfg.get("blocks", [1, 5])
    epochs_dir = pipeline_dir / "outputs" / "derivatives" / "epochs_clean"

    subjects = _find_subjects(epochs_dir, blocks, "pac")
    if not subjects:
        print("No PAC epoch files found.")
        return

    h1_nodes = cfg.get("h1_nodes", {})
    if not h1_nodes:
        print("ERROR: h1_nodes not found in config. Check study.yml.")
        return

    print("Node channels (active h1_nodes):")
    for node_name, channels in h1_nodes.items():
        print(f"  {node_name}: {channels}")

    phase_band = _pac_phase_band()
    amp_band = _pac_amp_band()
    print(
        f"PAC config: phase {phase_band[0]}-{phase_band[1]} Hz, "
        f"amp {amp_band[0]}-{amp_band[1]} Hz, "
        f"surrogates {_pac_n_surrogates()}, seed {_pac_seed()}"
    )
    print(
        f"PAC analysis window (post-filter crop): "
        f"{PAC_ANALYSIS_WINDOW[0]:.1f}-{PAC_ANALYSIS_WINDOW[1]:.1f} s"
    )

    between_rows = []

    for subj in subjects:
        print(f"\n{'=' * 50}")
        print(f"  PAC Analysis: {subj}")
        print(f"{'=' * 50}")

        for block in blocks:
            print(f"\n  --- Block {block} ---")
            epochs = load_block_epochs(subj, block, "pac", epochs_dir)
            if epochs is None or "eeg" not in epochs.get_channel_types():
                print(f"  No data for block {block}")
                continue

            sfreq = epochs.info["sfreq"]
            times = epochs.times

            between_row = {"subject": subj, "block": block}
            node_signals = {
                node: get_node_signal(epochs, chs) for node, chs in h1_nodes.items()
            }

            node_phase = {}
            node_amp = {}
            for node, sig in node_signals.items():
                if sig is None:
                    continue
                theta_phase, gamma_amp = _precompute_phase_amp(
                    sig,
                    sfreq,
                    phase_band,
                    amp_band,
                    times=times,
                    crop_window=PAC_ANALYSIS_WINDOW,
                )
                if theta_phase is not None and gamma_amp is not None:
                    node_phase[node] = theta_phase
                    node_amp[node] = gamma_amp

            for phase_node, amp_node in BETWEEN_PAC_PAIRS:
                col = f"pac_between_{phase_node}_{amp_node}"
                phase_arr = node_phase.get(phase_node)
                amp_arr = node_amp.get(amp_node)
                if phase_arr is None or amp_arr is None:
                    between_row[col] = np.nan
                    continue

                pac_between = _pac_from_precomputed(
                    phase_arr,
                    amp_arr,
                    sfreq,
                    phase_band,
                    amp_band,
                )
                between_row[col] = pac_between
                print(f"  Between PAC ({phase_node}->{amp_node}): {pac_between:.3f}")

            between_rows.append(between_row)

    if between_rows:
        df_between = pd.DataFrame(between_rows)
        filename = _tag_name("pac_between_features.csv")
        save_feature_tables(df_between, filename)
        out_between = OUTPUT_DIR / filename
        print(f"\nSaved between-region PAC to {out_between}")
        print(df_between.to_string(index=False))


if __name__ == "__main__":
    main()
