"""Configuration helpers for P3b QC."""

from __future__ import annotations

from copy import deepcopy


DEFAULT_P3B_QC = {
    "enabled": True,
    "plot_window": [-0.2, 0.8],
    "baseline_window": [-0.2, 0.0],
    "classification": {
        "clear_min_mean_uV": 0.5,
        "weak_min_mean_uV": 0.0,
        "review_if_missing_roi_channel": True,
    },
}


def deep_update(base: dict, updates: dict | None) -> dict:
    out = deepcopy(base)
    if not isinstance(updates, dict):
        return out
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_p3b_qc_config() -> dict:
    from src.utils_config import get_param

    return deep_update(DEFAULT_P3B_QC, get_param("p3b_qc", default={}))
