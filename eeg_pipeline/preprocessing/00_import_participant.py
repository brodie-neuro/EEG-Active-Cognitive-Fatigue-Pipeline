"""
Generic participant importer for CNT recordings.

Usage:
    python preprocessing/00_import_participant.py --input path/to/file.cnt --subject sub-001

Creates:
  - raw/sub-001_block1-task.fif
  - raw/sub-001_block5-task.fif
  - raw/sub-001_rest-post.fif (if long enough)
  - config/participant_configs/sub-001.json
  - config/adapters/sub_001_cnt.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
import sys

os.environ["MNE_DONTWRITE_HOME"] = "true"
os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(Path(__file__).resolve().parents[1]))

import mne
import numpy as np

PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from src.utils_io import load_config, normalize_subject_id


RAW_DIR = PIPELINE_DIR / "raw"
PARTICIPANT_CFG_DIR = PIPELINE_DIR / "config" / "participant_configs"
ADAPTERS_DIR = PIPELINE_DIR / "config" / "adapters"


DEFAULT_RENAMES = {
    "FP1": "Fp1",
    "LHEOG": "F7",
    "RHEOG": "F8",
}


DEFAULT_CHANNEL_TYPES = {
    "F7": "emg",
    "FT7": "emg",
    "F8": "emg",
    "FT8": "emg",
    "TP7": "emg",
    "TP8": "emg",
    "BVEOG": "misc",
    "TVEOG": "misc",
    "M1": "misc",
    "M2": "misc",
    "TP9": "misc",
}


def _derive_subject_id(path: Path) -> str:
    stem = path.stem.replace(" ", "_")
    return normalize_subject_id(stem)


def _safe_adapter_name(subject_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in subject_id).replace("-", "_")


def _default_config(subject_id: str, input_path: Path) -> dict:
    defaults = _study_participant_defaults()
    return {
        "subject_id": subject_id,
        "input_path": str(input_path),
        "gap_threshold_s": 30.0,
        "exclude_warmup": True,
        "task_blocks_to_keep": [1, 5],
        "rest_buffer_s": 2.0,
        "rest_min_duration_s": 30.0,
        "channel_renames": dict(DEFAULT_RENAMES),
        "channel_types": dict(DEFAULT_CHANNEL_TYPES),
        "known_bad_eeg": _channel_list(defaults.get("known_bad_eeg", [])),
        "known_bad_emg": _channel_list(defaults.get("known_bad_emg", [])),
        "event_map": {"1": "stim", "2": "stim/offset"},
        "stim_event_label": "1",
    }


def _channel_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_values = re.split(r"[,;\s]+", value)
    elif isinstance(value, (list, tuple, set)):
        raw_values = list(value)
    else:
        raw_values = [value]

    out: list[str] = []
    seen: set[str] = set()
    for item in raw_values:
        channel = str(item).strip()
        if not channel:
            continue
        key = channel.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(channel)
    return out


def _merge_channels(*values) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for value in values:
        for channel in _channel_list(value):
            key = channel.lower()
            if key in seen:
                continue
            seen.add(key)
            merged.append(channel)
    return merged


def _study_participant_defaults() -> dict:
    try:
        return load_config().get("participant_defaults", {}) or {}
    except Exception:
        return {}


def _normalise_participant_config(cfg: dict, subject_id: str, input_path: Path) -> dict:
    defaults = _study_participant_defaults()
    cfg["subject_id"] = subject_id
    cfg["input_path"] = str(input_path)
    cfg.setdefault("gap_threshold_s", 30.0)
    cfg.setdefault("exclude_warmup", True)
    cfg.setdefault("task_blocks_to_keep", [1, 5])
    cfg.setdefault("rest_buffer_s", 2.0)
    cfg.setdefault("rest_min_duration_s", 30.0)
    cfg["channel_renames"] = {**DEFAULT_RENAMES, **(cfg.get("channel_renames", {}) or {})}
    cfg["channel_types"] = {**DEFAULT_CHANNEL_TYPES, **(cfg.get("channel_types", {}) or {})}
    cfg.setdefault("event_map", {"1": "stim", "2": "stim/offset"})
    cfg.setdefault("stim_event_label", "1")
    cfg.setdefault("notes", "")

    # Older generated configs used "bad_channels". Keep it as a legacy alias,
    # but make known_bad_eeg / known_bad_emg the canonical fields.
    cfg["known_bad_eeg"] = _merge_channels(
        defaults.get("known_bad_eeg", []),
        cfg.get("known_bad_eeg", []),
        cfg.get("bad_channels", []),
    )
    cfg["known_bad_emg"] = _merge_channels(
        defaults.get("known_bad_emg", []),
        cfg.get("known_bad_emg", []),
    )
    return cfg


def _load_or_create_config(config_path: Path, subject_id: str, input_path: Path) -> dict:
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Participant config must be a JSON object: {config_path}")
        return _normalise_participant_config(data, subject_id, input_path)
    data = _default_config(subject_id, input_path)
    data = _normalise_participant_config(data, subject_id, input_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return data


def _save_config(config_path: Path, cfg: dict) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def _save_adapter(subject_id: str, cfg: dict) -> Path:
    adapter_name = f"{_safe_adapter_name(subject_id)}_cnt"
    out = ADAPTERS_DIR / f"{adapter_name}.json"
    payload = {
        "name": adapter_name,
        "description": f"Auto-generated adapter for {subject_id}",
        "data": {
            "root": str(RAW_DIR),
            "pattern": f"{subject_id}_block*-task.fif",
            "format": "fif",
        },
        "event_map": cfg.get("event_map", {"1": "stim", "2": "stim/offset"}),
        "channel_renames": cfg.get("channel_renames", {}),
        "channel_types": cfg.get("channel_types", {}),
        "known_bad_eeg": cfg.get("known_bad_eeg", []),
        "known_bad_emg": cfg.get("known_bad_emg", []),
        "bad_channels": cfg.get("known_bad_eeg", []),
    }
    ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out


def _choose_stim_event_id(events: np.ndarray, event_id: dict[str, int], cfg: dict) -> int:
    preferred = str(cfg.get("stim_event_label", "1")).strip()
    if preferred and preferred in event_id:
        return int(event_id[preferred])
    # Fallback: most frequent event code.
    codes, counts = np.unique(events[:, 2], return_counts=True)
    if len(codes) == 0:
        raise RuntimeError("No event codes found in recording.")
    return int(codes[np.argmax(counts)])


def _segment_by_gap(stim_events: np.ndarray, sfreq: float, gap_threshold_s: float) -> list[np.ndarray]:
    if len(stim_events) == 0:
        return []
    starts = [0]
    for i in range(1, len(stim_events)):
        gap_s = (stim_events[i, 0] - stim_events[i - 1, 0]) / sfreq
        if gap_s > gap_threshold_s:
            starts.append(i)
    segments: list[np.ndarray] = []
    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else len(stim_events)
        segments.append(stim_events[start:end])
    return segments


def _smart_warmup_exclusion(segments: list[np.ndarray], sfreq: float) -> list[np.ndarray]:
    """Identify and remove warmup events, handling the edge case where warmup
    and Block 1 are merged into a single segment (no gap between them).

    Strategy:
      1. If the first segment has far fewer events than others, it is a clean
         warmup -> just drop it (standard case).
      2. If the first segment has *more* events than a typical task block,
         warmup events are prepended to Block 1 -> split the segment and
         keep only the trailing task-block-sized portion as Block 1.
      3. If only one segment exists, return it as-is (cannot determine warmup).
    """
    if len(segments) <= 1:
        print("  WARNING: Only 1 segment found; cannot separate warmup.")
        return segments

    first_n = len(segments[0])
    rest_counts = [len(s) for s in segments[1:]]
    median_block = int(np.median(rest_counts))

    # Tolerance: warmup is detected if first segment differs by >20% from median
    tolerance = 0.20
    lower = median_block * (1 - tolerance)
    upper = median_block * (1 + tolerance)

    if first_n < lower:
        # Case 1: First segment is clearly a standalone warmup (few events)
        warmup_dur = (segments[0][-1, 0] - segments[0][0, 0]) / sfreq
        print(f"  Practice/warmup detected: segment 1 has {first_n} events "
              f"({warmup_dur:.0f}s), typical block has {median_block}. Dropping.")
        return segments[1:]

    elif first_n > upper:
        # Case 2: Warmup merged with Block 1 (no gap between them)
        excess = first_n - median_block
        print(f"  Practice/warmup+Block1 merge detected: segment 1 has {first_n} events "
              f"(expected ~{median_block}). Splitting off {excess} practice events.")
        # Keep only the last median_block events as Block 1
        block1_events = segments[0][-median_block:]
        return [block1_events] + segments[1:]

    else:
        # Case 3: First segment looks like a normal task block; no warmup found
        print(f"  No separate practice/warmup segment detected: segment 1 has {first_n} events "
              f"(~{median_block} expected). Keeping all segments.")
        return segments


def _apply_channel_config(raw: mne.io.BaseRaw, cfg: dict) -> None:
    rename_map = {}
    existing_lower = {ch.lower(): ch for ch in raw.ch_names}
    for src, dst in (cfg.get("channel_renames", {}) or {}).items():
        src_key = str(src).strip().lower()
        src_actual = existing_lower.get(src_key)
        dst_name = str(dst).strip()
        if not src_actual or not dst_name:
            continue
        if src_actual == dst_name:
            continue
        rename_map[src_actual] = dst_name
    if rename_map:
        raw.rename_channels(rename_map)

    ch_types = {}
    existing_lower = {ch.lower(): ch for ch in raw.ch_names}
    for ch, ch_type in (cfg.get("channel_types", {}) or {}).items():
        actual = existing_lower.get(str(ch).strip().lower())
        if actual:
            ch_types[actual] = str(ch_type).strip().lower()
    if ch_types:
        raw.set_channel_types(ch_types, verbose=False)

    bads = []
    existing_lower = {ch.lower(): ch for ch in raw.ch_names}
    for ch in cfg.get("known_bad_eeg", []) or []:
        actual = existing_lower.get(str(ch).strip().lower())
        if actual:
            bads.append(actual)
    raw.info["bads"] = sorted(set(bads))


def _parse_blocks(blocks_arg: str | None, cfg: dict) -> list[int]:
    if blocks_arg:
        vals = [x.strip() for x in blocks_arg.split(",") if x.strip()]
        return [int(v) for v in vals]
    from_cfg = cfg.get("task_blocks_to_keep", [1, 5]) or [1, 5]
    return [int(v) for v in from_cfg]


def main() -> None:
    parser = argparse.ArgumentParser(description="Import CNT participant and split into task blocks/rest.")
    parser.add_argument("--input", required=True, help="Path to raw .cnt file.")
    parser.add_argument("--subject", default="", help="Subject id (e.g. sub-dario).")
    parser.add_argument("--config", default="", help="Optional participant config JSON path.")
    parser.add_argument("--blocks", default="", help="Comma list of task blocks to keep (default from config).")
    parser.add_argument("--gap-threshold", type=float, default=None, help="Gap threshold in seconds (default config).")
    parser.add_argument("--rest-min-duration", type=float, default=None, help="Minimum post-task rest duration (s).")
    parser.add_argument("--rest-buffer", type=float, default=None, help="Buffer after last event before rest (s).")
    parser.add_argument("--include-warmup", action="store_true", help="Treat first detected segment as Block 1.")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    subject_id = normalize_subject_id(args.subject) if args.subject.strip() else _derive_subject_id(input_path)
    cfg_path = Path(args.config).expanduser().resolve() if args.config.strip() else (PARTICIPANT_CFG_DIR / f"{subject_id}.json")
    participant_cfg = _load_or_create_config(cfg_path, subject_id, input_path)

    study_cfg = load_config()
    montage = study_cfg.get("montage", "standard_1020")

    print(f"Loading CNT: {input_path}")
    raw = mne.io.read_raw_cnt(str(input_path), preload=False, verbose=False)
    raw.set_montage(montage, on_missing="ignore")
    _apply_channel_config(raw, participant_cfg)

    events, event_id = mne.events_from_annotations(raw, verbose=False)
    if len(events) == 0:
        raise RuntimeError("No trigger events found in recording.")

    stim_id = _choose_stim_event_id(events, event_id, participant_cfg)
    stim_events = events[events[:, 2] == stim_id]
    if len(stim_events) == 0:
        raise RuntimeError(f"No stimulus events found for event id {stim_id}.")

    if participant_cfg.get("known_bad_eeg"):
        print(f"Known bad EEG channels: {participant_cfg['known_bad_eeg']}")
    if participant_cfg.get("known_bad_emg"):
        print(f"Known bad EMG channels: {participant_cfg['known_bad_emg']}")

    gap_threshold = float(args.gap_threshold if args.gap_threshold is not None else participant_cfg.get("gap_threshold_s", 30.0))
    rest_min_duration = float(
        args.rest_min_duration if args.rest_min_duration is not None else participant_cfg.get("rest_min_duration_s", 30.0)
    )
    rest_buffer = float(args.rest_buffer if args.rest_buffer is not None else participant_cfg.get("rest_buffer_s", 2.0))
    include_warmup = bool(args.include_warmup or not bool(participant_cfg.get("exclude_warmup", True)))
    blocks_to_keep = _parse_blocks(args.blocks, participant_cfg)

    segments = _segment_by_gap(stim_events, raw.info["sfreq"], gap_threshold)
    if not segments:
        raise RuntimeError("Failed to detect task segments from triggers.")

    task_segments = segments if include_warmup else _smart_warmup_exclusion(segments, raw.info["sfreq"])
    if not task_segments:
        raise RuntimeError("No task segments remain after warmup exclusion.")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    written_blocks: list[str] = []
    for block in blocks_to_keep:
        idx = block - 1
        if idx < 0 or idx >= len(task_segments):
            print(f"Skipping block {block}: not found.")
            continue
        seg = task_segments[idx]
        t_start = seg[0, 0] / raw.info["sfreq"]
        t_end = min(raw.times[-1], seg[-1, 0] / raw.info["sfreq"] + rest_buffer)
        out = RAW_DIR / f"{subject_id}_block{block}-task.fif"
        print(
            f"Saving block {block}: {t_start:.1f}s to {t_end:.1f}s "
            f"({len(seg)} kept stimulus events) -> {out.name}"
        )
        raw.copy().crop(tmin=t_start, tmax=t_end).save(out, overwrite=True, verbose=False)
        written_blocks.append(str(out))

    # Post-task rest from last stimulus onward.
    last_stim_t = stim_events[-1, 0] / raw.info["sfreq"]
    rest_start = last_stim_t + rest_buffer
    rest_end = raw.times[-1]
    rest_duration = rest_end - rest_start
    rest_out = ""
    if rest_duration >= rest_min_duration:
        rest_path = RAW_DIR / f"{subject_id}_rest-post.fif"
        print(f"Saving rest-post: {rest_start:.1f}s to {rest_end:.1f}s -> {rest_path.name}")
        raw.copy().crop(tmin=rest_start, tmax=rest_end).save(rest_path, overwrite=True, verbose=False)
        rest_out = str(rest_path)
    else:
        print(f"Post-task rest too short ({rest_duration:.1f}s); skipping rest file.")

    adapter_path = _save_adapter(subject_id, participant_cfg)
    participant_cfg["generated"] = {
        "blocks": written_blocks,
        "rest_post": rest_out,
        "adapter": str(adapter_path),
    }
    participant_cfg["gap_threshold_s"] = gap_threshold
    participant_cfg["rest_min_duration_s"] = rest_min_duration
    participant_cfg["rest_buffer_s"] = rest_buffer
    participant_cfg["exclude_warmup"] = not include_warmup
    participant_cfg["task_blocks_to_keep"] = blocks_to_keep
    _save_config(cfg_path, participant_cfg)

    print("\nImport complete.")
    print(f"Participant config: {cfg_path}")
    print(f"Adapter: {adapter_path}")


if __name__ == "__main__":
    main()
