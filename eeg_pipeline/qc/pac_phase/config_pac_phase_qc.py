"""Configuration helpers for PAC phase spectral QC."""

from __future__ import annotations

from copy import deepcopy


DEFAULT_PAC_PHASE_QC = {
    "enabled": True,
    "phase_node": "C_broad_F",
    "fallback_phase_channels": ["Fz", "FCz", "F3", "F4", "FC1", "FC2", "FC3", "FC4"],
    "bands": {
        "theta": [4.0, 8.0],
        "alpha": [8.0, 13.0],
    },
    "optional_bands": {},
    "analysis_window": [0.0, 0.6],
    "psd": {
        "method": "welch",
        "fmin": 1.0,
        "fmax": 40.0,
        "nperseg_seconds": 2.0,
        "noverlap_fraction": 0.5,
        "window": "hann",
        "detrend": "constant",
    },
    "aperiodic": {
        "method": "specparam_if_available",
        "fit_range": [2.0, 40.0],
        "exclude_ranges": [[48.0, 52.0]],
        "exclude_target_bands_in_fallback": True,
        "robust_iterations": 4,
        "robust_positive_residual_z": 2.5,
        "specparam_max_n_peaks": 6,
        "specparam_peak_width_limits": [1.0, 12.0],
    },
    "classification": {
        "weak_min_positive_fraction": 0.20,
        "clear_min_positive_fraction": 0.35,
        "weak_min_positive_area": 0.01,
        "clear_min_positive_area": 0.025,
        "weak_min_peak_noise_z": 0.75,
        "clear_min_peak_noise_z": 1.25,
        "noise_floor": 1e-6,
    },
    "plots": {
        "xlim": [1.0, 30.0],
    },
}


def deep_update(base: dict, updates: dict | None) -> dict:
    """Return a recursive merge of ``updates`` into ``base``."""
    out = deepcopy(base)
    if not isinstance(updates, dict):
        return out
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_pac_phase_qc_config() -> dict:
    """Load PAC phase QC config from parameters.json with defaults."""
    from src.utils_config import get_param

    return deep_update(DEFAULT_PAC_PHASE_QC, get_param("pac_phase_qc", default={}))


def selected_bands(config: dict, band_names: list[str] | None = None) -> dict[str, tuple[float, float]]:
    """Resolve requested band names to numeric band limits."""
    all_bands = {}
    all_bands.update(config.get("bands", {}) or {})
    all_bands.update(config.get("optional_bands", {}) or {})

    if band_names:
        names = [name.strip().lower() for name in band_names if name.strip()]
    else:
        names = list((config.get("bands", {}) or {}).keys())

    resolved: dict[str, tuple[float, float]] = {}
    for name in names:
        if name not in all_bands:
            raise KeyError(f"Unknown PAC phase QC band '{name}'. Available: {sorted(all_bands)}")
        lo, hi = all_bands[name]
        resolved[name] = (float(lo), float(hi))
    return resolved
