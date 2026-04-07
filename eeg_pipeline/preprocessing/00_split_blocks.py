# steps/00_split_blocks.py
"""
Split a single raw file into two blocks based on a long inter-trigger gap.

Use case:
  - One continuous recording (e.g., Test_Brodie.cnt)
  - Block 1 then a 5-10s pause between triggers
  - Block 2 continues after the gap

This script finds the largest inter-event gap above a threshold and splits
the raw into Block 1 and Block 2, saving .fif files compatible with the pipeline.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import mne

# Add pipeline root for imports
PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from src.utils_io import load_config, read_raw  # noqa: E402


def _derive_subject_id(path: Path) -> str:
    base = path.stem
    base = base.replace(" ", "_")
    if base.lower().startswith("sub-"):
        return base
    return f"sub-{base}"


def _normalize_label(label: str) -> str:
    return str(label).strip()


def _get_event_series(raw: mne.io.BaseRaw):
    """Return (times_s, labels) for events in raw time coordinates (relative)."""
    # Prefer annotations for CNT
    if raw.annotations is not None and len(raw.annotations) > 0:
        times = np.asarray(raw.annotations.onset, dtype=float)
        labels = np.asarray([_normalize_label(d) for d in raw.annotations.description])
        # Convert absolute onsets to raw-relative if needed
        if times.size > 0 and raw.first_time:
            if times.min() >= raw.first_time - 1e-6:
                times = times - float(raw.first_time)
        return times, labels

    # Fallback: stim channel events
    try:
        events = mne.find_events(raw, verbose=False)
    except Exception:
        events = None
    if events is None or len(events) == 0:
        return np.array([]), np.array([])
    sfreq = raw.info["sfreq"]
    # events[:, 0] is already in raw-relative samples
    times = events[:, 0] / sfreq
    labels = np.asarray([str(c) for c in events[:, 2]])
    return times, labels


def _segment_by_gap(times: np.ndarray, labels: np.ndarray, gap_thresh_s: float):
    """Split an event series into contiguous segments separated by large gaps."""
    if times.size == 0:
        return []
    if times.size == 1:
        return [(times.copy(), labels.copy())]

    split_points = np.where(np.diff(times) > gap_thresh_s)[0] + 1
    bounds = np.concatenate(([0], split_points, [times.size]))
    segments = []
    for start, stop in zip(bounds[:-1], bounds[1:]):
        seg_times = times[start:stop]
        seg_labels = labels[start:stop]
        if seg_times.size:
            segments.append((seg_times, seg_labels))
    return segments


def _exclude_warmup(segments):
    """Drop or split an initial warmup segment based on event counts."""
    if len(segments) <= 1:
        return segments

    first_n = int(len(segments[0][0]))
    rest_counts = [len(seg_times) for seg_times, _seg_labels in segments[1:]]
    median_block = int(np.median(rest_counts))
    tolerance = 0.20
    lower = median_block * (1 - tolerance)
    upper = median_block * (1 + tolerance)

    if first_n < lower:
        return segments[1:]

    if first_n > upper:
        seg_times, seg_labels = segments[0]
        keep_n = median_block
        trimmed = (seg_times[-keep_n:].copy(), seg_labels[-keep_n:].copy())
        return [trimmed] + segments[1:]

    return segments


def split_by_gap(raw: mne.io.BaseRaw, gap_thresh_s: float, min_events: int, ignore_first: float):
    times, labels = _get_event_series(raw)
    if times.size < 2:
        raise RuntimeError("Not enough events found to split blocks.")

    # Ignore practice section
    mask = times >= ignore_first
    times = times[mask]
    labels = labels[mask]
    if times.size < 2:
        raise RuntimeError("Not enough events found after ignore-first.")

    segments = _segment_by_gap(times, labels, gap_thresh_s)
    if len(segments) < 2:
        largest = float(np.max(np.diff(times))) if times.size > 1 else 0.0
        raise RuntimeError(
            f"Found only {len(segments)} event segment(s); largest gap was {largest:.2f}s. "
            f"Adjust --gap-thresh if needed."
        )

    segments = _exclude_warmup(segments)
    if not segments:
        raise RuntimeError("No task segments remain after warmup exclusion.")

    small = [i + 1 for i, (seg_times, _seg_labels) in enumerate(segments) if len(seg_times) < min_events]
    if small:
        raise RuntimeError(
            f"Detected task segments with too few events: {small}. "
            f"Increase --gap-thresh or reduce --min-events."
        )

    return segments


def _trim_to_trials(raw: mne.io.BaseRaw, trial_label: str, target_trials: int, tail_pad: float):
    """Trim raw to the first N occurrences of trial_label (+ tail_pad seconds)."""
    times, labels = _get_event_series(raw)
    if times.size == 0:
        raise RuntimeError("No events found while trimming to trials.")
    trial_label = _normalize_label(trial_label)
    mask = labels == trial_label
    ev_times = times[mask]
    if len(ev_times) < target_trials:
        raise RuntimeError(
            f"Only {len(ev_times)} trial events found, fewer than target {target_trials}."
        )
    end_time = float(ev_times[target_trials - 1] + tail_pad)
    end_time = min(end_time, raw.times[-1])
    return raw.copy().crop(tmin=0.0, tmax=end_time)


def main():
    parser = argparse.ArgumentParser(
        description="Split a single raw file into block1/block2 by detecting a long event gap."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to raw file (e.g., eeg_pipeline/raw/Test_Brodie.cnt)",
    )
    parser.add_argument(
        "--out-dir",
        default=str(PIPELINE_DIR / "raw"),
        help="Output directory for split files (default: eeg_pipeline/raw)",
    )
    parser.add_argument(
        "--subject",
        default="",
        help="Optional subject ID override (e.g., sub-Brodie).",
    )
    parser.add_argument(
        "--gap-thresh",
        type=float,
        default=30.0,
        help="Minimum inter-event gap (s) to split blocks (default: 30.0).",
    )
    parser.add_argument(
        "--ignore-first",
        type=float,
        default=60.0,
        help="Ignore the first N seconds before splitting (default: 60.0).",
    )
    parser.add_argument(
        "--min-events",
        type=int,
        default=20,
        help="Minimum events required in each block (default: 20).",
    )
    parser.add_argument(
        "--block-labels",
        nargs=2,
        type=int,
        default=[1, 2],
        help="Two block numbers to use in output filenames (default: 1 2).",
    )
    parser.add_argument(
        "--trial-label",
        default="",
        help="Event label to count trials for trimming (e.g., '1'). Optional.",
    )
    parser.add_argument(
        "--target-trials",
        type=int,
        default=0,
        help="If set, trim each block to this many trials (requires --trial-label).",
    )
    parser.add_argument(
        "--tail-pad",
        type=float,
        default=2.0,
        help="Seconds to keep after last trial when trimming (default: 2.0).",
    )
    args = parser.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    cfg = load_config()
    montage = cfg.get("montage", "standard_1020")

    raw = read_raw(in_path, fmt="auto", montage=montage)
    if args.ignore_first >= raw.times[-1]:
        raise RuntimeError("ignore-first is longer than the recording.")

    segments = split_by_gap(
        raw, gap_thresh_s=args.gap_thresh, min_events=args.min_events, ignore_first=args.ignore_first
    )

    subj = args.subject.strip() or _derive_subject_id(in_path)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Detected {len(segments)} task segments after warmup exclusion.")
    for idx, (seg_times, seg_labels) in enumerate(segments, start=1):
        uniq, counts = np.unique(seg_labels, return_counts=True)
        pairs = ", ".join([f"{u}:{c}" for u, c in sorted(zip(uniq.tolist(), counts.tolist()))])
        print(
            f"  Task block {idx}: {seg_times[0]:.2f}s -> {seg_times[-1]:.2f}s "
            f"({len(seg_times)} events; {pairs})"
        )

    outputs = []
    for block_num in args.block_labels:
        seg_idx = block_num - 1
        if seg_idx < 0 or seg_idx >= len(segments):
            raise RuntimeError(
                f"Requested block label {block_num}, but only {len(segments)} task segments were found."
            )

        seg_times, _seg_labels = segments[seg_idx]
        t_start = float(seg_times[0])
        t_end = min(raw.times[-1], float(seg_times[-1] + args.tail_pad))
        block_raw = raw.copy().crop(tmin=t_start, tmax=t_end)

        if args.target_trials and args.trial_label:
            block_raw = _trim_to_trials(block_raw, args.trial_label, args.target_trials, args.tail_pad)

        out = out_dir / f"{subj}_block{block_num}-task.fif"
        block_raw.save(out, overwrite=True)
        outputs.append((block_num, block_raw, out))

    # Summary counts for assurance
    for block_num, r, out in outputs:
        times, labels = _get_event_series(r)
        if labels.size == 0:
            print(f"block{block_num}: no events found")
            continue
        uniq, counts = np.unique(labels, return_counts=True)
        pairs = ", ".join([f"{u}:{c}" for u, c in sorted(zip(uniq.tolist(), counts.tolist()))])
        print(f"block{block_num} event counts -> {pairs}")
        if args.trial_label:
            trial_label = _normalize_label(args.trial_label)
            n_trial = int((labels == trial_label).sum())
            print(f"block{block_num} trial '{trial_label}' count -> {n_trial}")
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
