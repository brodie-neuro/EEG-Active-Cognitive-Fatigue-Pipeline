# eeg_pipeline/src/utils_naming.py
"""
File naming utility -- reads file_naming.json so all scripts use consistent names.

Usage
-----
    from src.utils_naming import get_derivative_name, get_feature_name, get_figure_name

    fname = get_derivative_name('sub-001', 1, 'asr_cleaned')
    # -> 'sub-001_block1_asr-raw.fif'
"""
import json
from pathlib import Path

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
_NAMING_CACHE = None


def _load_naming():
    """Load and cache file_naming.json."""
    global _NAMING_CACHE
    if _NAMING_CACHE is None:
        naming_path = _CONFIG_DIR / "file_naming.json"
        if not naming_path.exists():
            raise FileNotFoundError(
                f"file_naming.json not found at {naming_path}."
            )
        with open(naming_path, 'r', encoding='utf-8') as f:
            _NAMING_CACHE = json.load(f)
    return _NAMING_CACHE


def get_derivative_name(subject, block, step, epoch_type=None):
    """Build a derivative file name from the naming config.

    Parameters
    ----------
    subject : str
        Subject ID (e.g. 'sub-001').
    block : int or str
        Block number.
    step : str
        Derivative step key from file_naming.json (e.g. 'asr_cleaned',
        'p3b_epochs', 'epochs_clean').
    epoch_type : str, optional
        Epoch type for 'epochs_clean' pattern (e.g. 'p3b', 'pac').

    Returns
    -------
    str
        Formatted file name.
    """
    naming = _load_naming()
    pattern = naming.get('derivatives', {}).get(step)
    if pattern is None:
        raise KeyError(f"Unknown derivative step '{step}'. "
                       f"Available: {list(naming.get('derivatives', {}).keys())}")
    return pattern.format(subject=subject, block=block, type=epoch_type or '')


def get_feature_name(feature_type):
    """Get the CSV filename for a feature type.

    Parameters
    ----------
    feature_type : str
        Feature key (e.g. 'p3b', 'iaf', 'pac_local', 'merged').

    Returns
    -------
    str
        CSV filename.
    """
    naming = _load_naming()
    name = naming.get('features', {}).get(feature_type)
    if name is None:
        raise KeyError(f"Unknown feature type '{feature_type}'. "
                       f"Available: {list(naming.get('features', {}).keys())}")
    return name


def get_figure_name(figure_type):
    """Get the figure filename.

    Parameters
    ----------
    figure_type : str
        Figure key (e.g. 'psd_specparam', 'pac_montage', 'p3b_erp').

    Returns
    -------
    str
        PNG filename.
    """
    naming = _load_naming()
    name = naming.get('figures', {}).get(figure_type)
    if name is None:
        raise KeyError(f"Unknown figure type '{figure_type}'. "
                       f"Available: {list(naming.get('figures', {}).keys())}")
    return name


def get_rest_pattern(timepoint='pre'):
    """Get the resting-state file pattern.

    Parameters
    ----------
    timepoint : str
        'pre' or 'post'.

    Returns
    -------
    str
        File pattern string with {subject} placeholder.
    """
    naming = _load_naming()
    key = f'{timepoint}_pattern'
    return naming.get('resting_state', {}).get(key, f'sub-{{subject}}_rest-{timepoint}.vhdr')
