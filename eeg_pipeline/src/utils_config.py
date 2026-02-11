# eeg_pipeline/src/utils_config.py
"""
Centralised configuration loader for the EEG pipeline.

Loads parameters.json (algorithm settings) and study.yml (study design).
All scripts should use this module rather than hardcoding values.
"""
import json
import os
import yaml
from pathlib import Path

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
_PARAMS_CACHE = None
_STUDY_CACHE = None


def _resolve_params_path() -> Path:
    """Return parameters path, allowing runtime override."""
    env_path = os.environ.get("EEG_PARAMETERS_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return _CONFIG_DIR / "parameters.json"


def load_parameters(force_reload=False):
    """Load parameters.json -- algorithm settings.
    
    Returns
    -------
    dict
        Full parameters dictionary. Keys include:
        filtering, zapline, ica, asr, autoreject, epoching,
        p3b, specparam, band_power, pac, iaf, itf, qc
    """
    global _PARAMS_CACHE
    if _PARAMS_CACHE is None or force_reload:
        params_path = _resolve_params_path()
        if not params_path.exists():
            raise FileNotFoundError(
                f"parameters.json not found at {params_path}. "
                "Please ensure eeg_pipeline/config/parameters.json exists."
            )
        with open(params_path, 'r', encoding='utf-8') as f:
            _PARAMS_CACHE = json.load(f)
    return _PARAMS_CACHE


def load_study_config(force_reload=False):
    """Load study.yml -- study design (nodes, blocks, montage, etc.).
    
    Returns
    -------
    dict
        Study configuration dictionary.
    """
    global _STUDY_CACHE
    if _STUDY_CACHE is None or force_reload:
        study_path = _CONFIG_DIR / "study.yml"
        if not study_path.exists():
            raise FileNotFoundError(
                f"study.yml not found at {study_path}. "
                "Please ensure eeg_pipeline/config/study.yml exists."
            )
        with open(study_path, 'r', encoding='utf-8') as f:
            _STUDY_CACHE = yaml.safe_load(f)
    return _STUDY_CACHE


def get_param(section, key=None, default=None):
    """Get a specific parameter value with optional default.
    
    Parameters
    ----------
    section : str
        Top-level section name (e.g. 'ica', 'asr', 'pac').
    key : str, optional
        Key within the section. If None, returns entire section.
    default : any, optional
        Default value if key not found.
    
    Returns
    -------
    value
        Parameter value, section dict, or default.
    
    Examples
    --------
    >>> get_param('ica', 'n_components')
    25
    >>> get_param('pac')
    {'phase_band': [4, 8], 'amp_band': [55, 85], ...}
    >>> get_param('ica', 'missing_key', default=10)
    10
    """
    params = load_parameters()
    section_data = params.get(section, {})
    if key is None:
        return section_data if section_data else default
    return section_data.get(key, default)


def get_qc_threshold(metric):
    """Get a QC acceptance threshold.
    
    Parameters
    ----------
    metric : str
        QC metric name (e.g. 'max_epoch_rejection_pct').
    
    Returns
    -------
    value
        Threshold value.
    """
    return get_param('qc', metric)


def get_node_channels(node_name):
    """Get channel list for a 9-node region from study.yml.
    
    Parameters
    ----------
    node_name : str
        Node identifier (LF, CF, RF, LC, CC, RC, LP, CP, RP).
    
    Returns
    -------
    list of str
        Channel names for the node, or empty list.
    """
    cfg = load_study_config()
    return cfg.get('nodes', {}).get(node_name, [])


def get_blocks():
    """Get the blocks to compare from study.yml."""
    cfg = load_study_config()
    return cfg.get('blocks', [1, 5])


def parameters_hash():
    """Return a short hash of the current parameters.json for traceability."""
    import hashlib
    params_path = _resolve_params_path()
    content = params_path.read_bytes()
    return hashlib.md5(content).hexdigest()[:8]
