"""Run P3b QC for all available participants."""

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
import pandas as pd

PIPELINE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPELINE_DIR))
os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(PIPELINE_DIR))

from src.utils_config import get_param
from src.utils_features import get_subjects_with_blocks, load_block_epochs
from src.utils_io import discover_subjects, load_config

from qc.p3b.compute_p3b_qc_metrics import compute_p3b_block_metrics, select_target_epochs
from qc.p3b.config_p3b_qc import load_p3b_qc_config
from qc.p3b.plot_p3b_qc_group import plot_group_p3b_qc
from qc.p3b.plot_p3b_qc_individual import plot_subject_p3b_qc
from qc.p3b.summarise_p3b_qc import (
    make_delta_table,
    make_descriptives,
    make_status_counts,
    write_summary_report,
)


BASE_OUT = PIPELINE_DIR / "outputs" / "qc" / "p3b"
TABLE_DIR = BASE_OUT / "tables"
FIG_DIR = BASE_OUT / "figures"
REPORT_DIR = BASE_OUT / "reports"
LOG_DIR = BASE_OUT / "logs"


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("p3b_qc")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(LOG_DIR / "p3b_qc.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def _merge_existing(new_df: pd.DataFrame, path: Path) -> pd.DataFrame:
    if path.exists() and "subject" in new_df.columns:
        existing = pd.read_csv(path)
        new_subjects = set(new_df["subject"].astype(str))
        existing = existing[~existing["subject"].astype(str).isin(new_subjects)]
        return pd.concat([existing, new_df], ignore_index=True)
    return new_df


def main() -> None:
    parser = argparse.ArgumentParser(description="P3b QC")
    parser.add_argument("--subject", default="", help="Optional subject filter, e.g. sub-p003 or sub-p003,sub-p005")
    parser.add_argument("--no-plots", action="store_true", help="Skip figure generation.")
    args, _ = parser.parse_known_args()
    if args.subject.strip():
        os.environ["EEG_SUBJECT_FILTER"] = args.subject.strip()

    logger = setup_logging()
    logger.info("Starting P3b QC")
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    study_cfg = load_config()
    qc_cfg = load_p3b_qc_config()
    blocks = [int(b) for b in study_cfg.get("blocks", [1, 5])]
    erp_cfg = get_param("erp_branch", default={}) or {}
    epoch_type = erp_cfg.get("epoch_type", "p3b_erp")
    p3b_cfg = get_param("p3b", default={}) or {}
    roi_channels = p3b_cfg.get("channels", ["Pz", "P1", "P2", "POz"])
    p3b_window = (float(p3b_cfg.get("tmin_peak", 0.300)), float(p3b_cfg.get("tmax_peak", 0.500)))
    baseline_window = tuple(float(v) for v in qc_cfg.get("baseline_window", [-0.2, 0.0]))
    plot_window = tuple(float(v) for v in qc_cfg.get("plot_window", [-0.2, 0.8]))
    epochs_dir = PIPELINE_DIR / "outputs" / "derivatives" / "epochs_clean"

    subjects = get_subjects_with_blocks(epochs_dir, epoch_type, blocks)
    if not subjects:
        subjects = discover_subjects(epochs_dir=epochs_dir, blocks=blocks, epoch_type=epoch_type, require_all_blocks=False)
    if not subjects:
        logger.warning("No cleaned %s epoch files found.", epoch_type)
        return

    logger.info("Subjects: %s", subjects)
    logger.info("P3b ROI: %s", roi_channels)

    rows: list[dict] = []
    failures: list[dict] = []

    for subject_index, subject in enumerate(subjects, start=1):
        logger.info("P3b QC: %s", subject)
        subject_rows: list[dict] = []
        payload_by_block = {}
        epochs_by_block = {}
        for block in blocks:
            try:
                epochs = load_block_epochs(subject, block, epoch_type, epochs_dir)
                if epochs is None:
                    raise FileNotFoundError(f"No cleaned {epoch_type} epochs for block {block}")
                target_epochs = select_target_epochs(epochs)
                metrics, payload = compute_p3b_block_metrics(
                    target_epochs,
                    roi_channels,
                    p3b_window,
                    baseline_window,
                    qc_cfg.get("classification", {}),
                )
                row = {
                    "subject": subject,
                    "block": int(block),
                    "epoch_type": epoch_type,
                    "p3b_window_s": f"{p3b_window[0]}-{p3b_window[1]}",
                    "baseline_window_s": f"{baseline_window[0]}-{baseline_window[1]}",
                }
                row.update(metrics)
                subject_rows.append(row)
                rows.append(row)
                payload_by_block[block] = payload
                epochs_by_block[block] = target_epochs
                logger.info(
                    "%s B%s: %s mean=%.2f uV FAL=%.0f ms n=%s",
                    subject,
                    block,
                    row["qc_status"],
                    row["p3b_mean_uV"],
                    row["p3b_fractional_area_latency_ms"],
                    row["n_epochs"],
                )
            except Exception as exc:
                logger.exception("P3b QC failed for %s block %s", subject, block)
                failures.append({"subject": subject, "block": block, "error": str(exc)})

        if subject_rows and not args.no_plots:
            try:
                subject_df = pd.DataFrame(subject_rows)
                out = plot_subject_p3b_qc(
                    subject,
                    subject_index,
                    epochs_by_block,
                    payload_by_block,
                    subject_df,
                    roi_channels,
                    p3b_window,
                    plot_window,
                    FIG_DIR,
                )
                logger.info("Saved %s", out)
            except Exception as exc:
                logger.exception("P3b QC plot failed for %s", subject)
                failures.append({"subject": subject, "block": "plot", "error": str(exc)})

    if rows:
        df = pd.DataFrame(rows)
        metrics_path = TABLE_DIR / "p3b_qc_metrics.csv"
        df = _merge_existing(df, metrics_path)
        df.to_csv(metrics_path, index=False)

        delta = make_delta_table(df)
        delta.to_csv(TABLE_DIR / "p3b_qc_delta_metrics.csv", index=False)

        counts = make_status_counts(df)
        counts.to_csv(TABLE_DIR / "p3b_qc_status_counts.csv", index=False)

        desc = make_descriptives(df)
        desc.to_csv(TABLE_DIR / "p3b_qc_descriptives.csv", index=False)

        report = write_summary_report(df, counts, desc, REPORT_DIR / "p3b_qc_summary.md")
        logger.info("Saved P3b QC tables to %s", TABLE_DIR)
        logger.info("Saved P3b QC report to %s", report)

        if not args.no_plots:
            group_fig = plot_group_p3b_qc(df, FIG_DIR)
            if group_fig:
                logger.info("Saved %s", group_fig)

    if failures:
        failures_path = TABLE_DIR / "p3b_qc_failures.csv"
        pd.DataFrame(failures).to_csv(failures_path, index=False)
        logger.warning("Saved failures to %s", failures_path)

    logger.info("P3b QC complete")


if __name__ == "__main__":
    main()
