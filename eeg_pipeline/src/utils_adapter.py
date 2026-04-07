"""Dataset adapter helpers for external/OpenNeuro-style datasets."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path


def _normalize_label(label: str) -> str:
    text = str(label).strip().lower()
    if "/" in text:
        # Keep right-most token for labels like "Comment/target".
        text = text.split("/")[-1].strip()
    return text


def load_adapter_config() -> dict:
    """Load adapter profile JSON from EEG_ADAPTER_PATH."""
    path_str = os.environ.get("EEG_ADAPTER_PATH", "").strip()
    if not path_str:
        return {}

    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Adapter profile not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Adapter profile must be a JSON object.")
    data["_path"] = str(path)
    return data


def apply_adapter_to_study_cfg(cfg: dict, adapter: dict) -> dict:
    """Merge adapter data settings into loaded study config."""
    if not adapter:
        return cfg

    out = dict(cfg)
    out["adapter"] = adapter

    data_cfg = dict(out.get("data", {}))
    adapter_data = adapter.get("data", {}) or {}
    for key in ("root", "pattern", "format"):
        if key in adapter_data and adapter_data[key]:
            data_cfg[key] = adapter_data[key]
    out["data"] = data_cfg
    return out


def adapt_subject_id(raw_path: str | Path, fallback_id: str) -> str:
    """Map external dataset path to pipeline block naming when configured."""
    try:
        adapter = load_adapter_config()
    except Exception:
        return fallback_id
    if not adapter:
        return fallback_id

    block_map = adapter.get("block_map", {}) or {}
    if not block_map:
        return fallback_id

    p_norm = str(raw_path).replace("\\", "/").lower()
    block_num = None
    for token, block in block_map.items():
        if str(token).lower() in p_norm:
            try:
                block_num = int(block)
                break
            except Exception:
                continue
    if block_num is None:
        return fallback_id

    regex = adapter.get("subject_regex", r"(sub-[A-Za-z0-9]+)")
    m = re.search(regex, p_norm, flags=re.IGNORECASE)
    if m:
        subject = m.group(1).lower()
    else:
        # Fallback from parsed ID, e.g. "sub-001_ses-pre_task-SART_eeg".
        subject = fallback_id.split("_")[0]

    if not subject.startswith("sub-"):
        subject = f"sub-{subject}"
    return f"{subject}_block{block_num}"


def map_event_id_labels(event_id: dict[str, int], adapter: dict | None) -> dict[str, int]:
    """Map event labels from external datasets into pipeline labels."""
    if not adapter:
        return event_id
    exact_map = {str(k).lower(): str(v) for k, v in (adapter.get("event_map", {}) or {}).items()}
    contains_map = {str(k).lower(): str(v) for k, v in (adapter.get("event_contains_map", {}) or {}).items()}
    if not exact_map and not contains_map:
        return event_id

    mapped: dict[str, int] = {}
    for label, code in event_id.items():
        norm = _normalize_label(label)

        target = exact_map.get(norm)
        if target is None:
            for token, mapped_label in contains_map.items():
                if token in norm:
                    target = mapped_label
                    break
        if target is None:
            target = label

        # Handle collisions by suffixing event code.
        if target in mapped and mapped[target] != code:
            target = f"{target}_{code}"
        mapped[target] = code
    return mapped
