# eeg_pipeline/src/utils_logging.py
"""
Centralised logging for the EEG pipeline.

Every script should import and call `setup_pipeline_logger(script_name)`
at the start of main(). This creates:
  - Console output (INFO level)
  - Timestamped log file in outputs/logs/ (DEBUG level)
  - Structured format with script name, timestamp, and level

Usage:
    from src.utils_logging import setup_pipeline_logger
    logger = setup_pipeline_logger('12_peak_frequencies')
    logger.info("Processing subject %s", subj)
    logger.warning("No theta peak found for %s", subj)
    logger.error("File not found: %s", path)
"""
import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_pipeline_logger(script_name: str, level: str = 'DEBUG') -> logging.Logger:
    """
    Create and configure a pipeline logger.

    Parameters
    ----------
    script_name : str
        Name of the calling script (e.g., '12_peak_frequencies').
    level : str
        Minimum log level for the file handler ('DEBUG', 'INFO', etc.).

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    # Avoid duplicate handlers if called multiple times
    logger = logging.getLogger(f'eeg_pipeline.{script_name}')
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.DEBUG))

    # --- Log directory ---
    pipeline_dir = Path(__file__).resolve().parents[1]
    log_dir = pipeline_dir / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # --- File handler (DEBUG level, full detail) ---
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"{script_name}_{timestamp}.log"

    file_fmt = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(file_fmt)
    logger.addHandler(fh)

    # --- Console handler (INFO level, cleaner) ---
    console_fmt = logging.Formatter(
        '%(levelname)-8s | %(message)s'
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(console_fmt)
    logger.addHandler(ch)

    logger.info("=" * 60)
    logger.info("Pipeline script: %s", script_name)
    logger.info("Log file: %s", log_file)
    logger.info("Started: %s", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    logger.info("=" * 60)

    return logger


def log_subject_start(logger: logging.Logger, subj: str, block=None):
    """Log the start of processing for a subject/block."""
    if block is not None:
        logger.info("--- %s | Block %s ---", subj, block)
    else:
        logger.info("--- %s ---", subj)


def log_band_decision(logger: logging.Logger, subj: str, block,
                      band_type: str, band: tuple,
                      peak=None, source: str = 'individual'):
    """Log the frequency band decision for a subject.

    Parameters
    ----------
    band_type : str
        'theta' or 'alpha'
    band : tuple
        (low, high) Hz
    peak : float or None
        Individual peak frequency if found
    source : str
        'individual', 'fallback', or 'config'
    """
    if peak is not None:
        logger.info("  %s %s band: %.1f-%.1f Hz (peak=%.2f, source=%s)",
                    subj, band_type, band[0], band[1], peak, source)
    else:
        logger.info("  %s %s band: %.1f-%.1f Hz (source=%s)",
                    subj, band_type, band[0], band[1], source)


def log_feature_result(logger: logging.Logger, subj: str, block,
                       feature_name: str, value):
    """Log an extracted feature value."""
    if value is not None and not (isinstance(value, float) and value != value):
        logger.info("  %s block %s | %s = %.4f", subj, block, feature_name, value)
    else:
        logger.warning("  %s block %s | %s = NaN (extraction failed)", subj, block, feature_name)


def log_save(logger: logging.Logger, filepath: Path, n_rows: int = None):
    """Log file save."""
    msg = f"Saved: {filepath}"
    if n_rows is not None:
        msg += f" ({n_rows} rows)"
    logger.info(msg)
