"""Run PAC phase-band spectral QC for all available participants."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MNE_DONTWRITE_HOME"] = "true"

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

PIPELINE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPELINE_DIR))
os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(PIPELINE_DIR))

from src.utils_features import available_channels, filter_excluded_channels, get_subjects_with_blocks, load_block_epochs
from src.utils_io import discover_subjects, load_config

from qc.pac_phase.compute_pac_phase_qc_metrics import (
    compute_band_metrics,
    compute_welch_psd,
    crop_and_concatenate_trials,
    fit_aperiodic_and_residual,
)
from qc.pac_phase.config_pac_phase_qc import load_pac_phase_qc_config, selected_bands
from qc.pac_phase.plot_pac_phase_qc_group import plot_group_pac_phase_qc
from qc.pac_phase.plot_pac_phase_qc_individual import plot_subject_pac_phase_qc
from qc.pac_phase.summarise_pac_phase_qc import (
    make_delta_table,
    make_descriptives,
    make_status_counts,
    write_summary_report,
)


BASE_OUT = PIPELINE_DIR / "outputs" / "qc" / "pac_phase"
TABLE_DIR = BASE_OUT / "tables"
FIG_DIR = BASE_OUT / "figures"
REPORT_DIR = BASE_OUT / "reports"
LOG_DIR = BASE_OUT / "logs"


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("pac_phase_qc")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(LOG_DIR / "pac_phase_qc.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def get_phase_channels(study_cfg: dict, qc_cfg: dict) -> tuple[str, list[str]]:
    node_name = str(qc_cfg.get("phase_node", "C_broad_F"))
    h1_nodes = study_cfg.get("h1_nodes", {}) or {}
    channels = h1_nodes.get(node_name)
    if not channels:
        channels = qc_cfg.get("fallback_phase_channels", [])
    return node_name, list(channels)


def get_roi_signal(epochs, channels: list[str]) -> tuple[np.ndarray, list[str]]:
    channels, _ = filter_excluded_channels(channels)
    avail = available_channels(channels, epochs.ch_names)
    if avail:
        types = epochs.get_channel_types(picks=avail)
        avail = [ch for ch, ch_type in zip(avail, types) if ch_type == "eeg"]
    if not avail:
        raise ValueError("No available EEG channels for PAC phase QC ROI.")
    data = epochs.copy().pick(avail).get_data()
    return data.mean(axis=1), avail


def _merge_existing(new_df: pd.DataFrame, path: Path, subject_col: str = "subject") -> pd.DataFrame:
    if path.exists() and subject_col in new_df.columns:
        existing = pd.read_csv(path)
        new_subjects = set(new_df[subject_col].astype(str))
        existing = existing[~existing[subject_col].astype(str).isin(new_subjects)]
        return pd.concat([existing, new_df], ignore_index=True)
    return new_df


def main() -> None:
    parser = argparse.ArgumentParser(description="PAC phase-band spectral QC")
    parser.add_argument("--subject", default="", help="Optional subject filter, e.g. sub-p003 or sub-p003,sub-p005")
    parser.add_argument("--bands", default="", help="Comma-separated bands to run. Default: configured theta,alpha.")
    parser.add_argument("--no-plots", action="store_true", help="Skip figure generation.")
    args, _ = parser.parse_known_args()
    if args.subject.strip():
        os.environ["EEG_SUBJECT_FILTER"] = args.subject.strip()

    logger = setup_logging()
    logger.info("Starting PAC phase QC")

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    study_cfg = load_config()
    qc_cfg = load_pac_phase_qc_config()
    bands = selected_bands(
        qc_cfg,
        [b.strip() for b in args.bands.split(",") if b.strip()] if args.bands.strip() else None,
    )
    blocks = [int(b) for b in study_cfg.get("blocks", [1, 5])]
    node_name, phase_channels = get_phase_channels(study_cfg, qc_cfg)
    epochs_dir = PIPELINE_DIR / "outputs" / "derivatives" / "epochs_clean"

    subjects = get_subjects_with_blocks(epochs_dir, "pac", blocks)
    if not subjects:
        subjects = discover_subjects(epochs_dir=epochs_dir, blocks=blocks, epoch_type="pac", require_all_blocks=False)
    if not subjects:
        logger.warning("No cleaned PAC epoch files found.")
        return

    logger.info("Subjects: %s", subjects)
    logger.info("Bands: %s", bands)
    logger.info("Phase node: %s %s", node_name, phase_channels)

    all_rows: list[dict] = []
    failure_rows: list[dict] = []

    for subject_index, subject in enumerate(subjects, start=1):
        logger.info("PAC phase QC: %s", subject)
        spectra_by_block = {}
        subject_rows: list[dict] = []

        for block in blocks:
            try:
                epochs = load_block_epochs(subject, block, "pac", epochs_dir)
                if epochs is None:
                    raise FileNotFoundError(f"No cleaned pac epochs for block {block}")
                roi_signal, used_channels = get_roi_signal(epochs, phase_channels)
                concat = crop_and_concatenate_trials(
                    roi_signal,
                    epochs.times,
                    tuple(float(v) for v in qc_cfg.get("analysis_window", [0.0, 0.6])),
                )
                freqs, psd, psd_settings = compute_welch_psd(concat, epochs.info["sfreq"], qc_cfg.get("psd", {}))
                spectrum = fit_aperiodic_and_residual(freqs, psd, bands, qc_cfg.get("aperiodic", {}))
                spectra_by_block[block] = spectrum

                for band_name, band in bands.items():
                    metrics = compute_band_metrics(spectrum, band_name, band, qc_cfg.get("classification", {}))
                    row = {
                        "subject": subject,
                        "block": int(block),
                        "band": band_name,
                        "roi": node_name,
                        "roi_channels": ",".join(used_channels),
                        "n_trials": int(len(epochs)),
                        "sfreq_hz": float(epochs.info["sfreq"]),
                        "analysis_window_s": f"{qc_cfg.get('analysis_window', [0.0, 0.6])[0]}-{qc_cfg.get('analysis_window', [0.0, 0.6])[1]}",
                        "psd_method": psd_settings["method"],
                        "psd_nperseg": psd_settings["nperseg"],
                        "psd_noverlap": psd_settings["noverlap"],
                        "aperiodic_method": spectrum.aperiodic_method,
                        "aperiodic_slope": spectrum.aperiodic_slope,
                        "aperiodic_intercept": spectrum.aperiodic_intercept,
                        "aperiodic_exponent": spectrum.aperiodic_exponent,
                        "residual_noise_mad": spectrum.residual_noise_mad,
                    }
                    row.update(metrics)
                    subject_rows.append(row)
                    all_rows.append(row)
                    logger.info(
                        "%s B%s %s: %s CoM=%s peak=%s area=%.4f",
                        subject,
                        block,
                        band_name,
                        row["qc_status"],
                        f"{row['centre_of_mass_hz']:.2f}" if np.isfinite(row["centre_of_mass_hz"]) else "NA",
                        f"{row['peak_frequency_hz']:.2f}" if np.isfinite(row["peak_frequency_hz"]) else "NA",
                        row["positive_residual_area"],
                    )
            except Exception as exc:
                logger.exception("PAC phase QC failed for %s block %s", subject, block)
                failure_rows.append({"subject": subject, "block": block, "error": str(exc)})

        if subject_rows and spectra_by_block and not args.no_plots:
            try:
                subj_df = pd.DataFrame(subject_rows)
                out_fig = plot_subject_pac_phase_qc(
                    subject,
                    subject_index,
                    spectra_by_block,
                    subj_df,
                    bands,
                    FIG_DIR,
                    qc_cfg.get("plots", {}),
                )
                logger.info("Saved %s", out_fig)
            except Exception as exc:
                logger.exception("PAC phase plot failed for %s", subject)
                failure_rows.append({"subject": subject, "block": "plot", "error": str(exc)})

    if all_rows:
        metrics_df = pd.DataFrame(all_rows)
        metrics_path = TABLE_DIR / "pac_phase_qc_band_metrics.csv"
        metrics_df = _merge_existing(metrics_df, metrics_path)
        metrics_df.to_csv(metrics_path, index=False)

        delta_df = make_delta_table(metrics_df)
        delta_path = TABLE_DIR / "pac_phase_qc_delta_metrics.csv"
        delta_df.to_csv(delta_path, index=False)

        counts_df = make_status_counts(metrics_df)
        counts_path = TABLE_DIR / "pac_phase_qc_status_counts.csv"
        counts_df.to_csv(counts_path, index=False)

        desc_df = make_descriptives(metrics_df)
        desc_path = TABLE_DIR / "pac_phase_qc_descriptives.csv"
        desc_df.to_csv(desc_path, index=False)

        report_path = write_summary_report(metrics_df, counts_df, desc_df, REPORT_DIR / "pac_phase_qc_summary.md")
        logger.info("Saved PAC phase QC tables to %s", TABLE_DIR)
        logger.info("Saved PAC phase QC report to %s", report_path)

        if not args.no_plots:
            for path in plot_group_pac_phase_qc(metrics_df, FIG_DIR):
                logger.info("Saved %s", path)

    if failure_rows:
        failures_path = TABLE_DIR / "pac_phase_qc_failures.csv"
        pd.DataFrame(failure_rows).to_csv(failures_path, index=False)
        logger.warning("Saved failures to %s", failures_path)

    logger.info("PAC phase QC complete")


if __name__ == "__main__":
    main()
