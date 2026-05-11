"""
Import one CNT participant, then optionally run the full EEG pipeline.

This is the standard entry point for adding a new local participant without
writing a bespoke CNT cutter.

Example:
    python eeg_pipeline/run_new_participant.py --input "Raw Data/p014.cnt" --subject sub-p014 --known-bad-emg TP7
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parent
PARTICIPANT_CFG_DIR = PIPELINE_DIR / "config" / "participant_configs"

os.environ["MNE_DONTWRITE_HOME"] = "true"
os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(PIPELINE_DIR))

if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from src.utils_io import load_config, normalize_subject_id


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


def _derive_subject_id(input_path: Path) -> str:
    return normalize_subject_id(input_path.stem.replace(" ", "_"))


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Participant config must be a JSON object: {path}")
    return payload


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def prepare_participant_config(
    subject_id: str,
    input_path: Path,
    known_bad_eeg: str,
    known_bad_emg: str,
    notes: str,
) -> Path:
    cfg_path = PARTICIPANT_CFG_DIR / f"{subject_id}.json"
    cfg = _load_json(cfg_path)
    defaults = (load_config().get("participant_defaults", {}) or {})

    cfg["subject_id"] = subject_id
    cfg["input_path"] = str(input_path)
    cfg["known_bad_eeg"] = _merge_channels(
        defaults.get("known_bad_eeg", []),
        cfg.get("known_bad_eeg", []),
        cfg.get("bad_channels", []),
        known_bad_eeg,
    )
    cfg["known_bad_emg"] = _merge_channels(
        defaults.get("known_bad_emg", []),
        cfg.get("known_bad_emg", []),
        known_bad_emg,
    )
    cfg.setdefault("notes", "")
    if notes.strip():
        cfg["notes"] = notes.strip()

    _save_json(cfg_path, cfg)
    return cfg_path


def run_checked(cmd: list[str], cwd: Path) -> None:
    print("\n" + "=" * 70)
    print(" ".join(cmd))
    print("=" * 70)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import a CNT participant and optionally run the full EEG pipeline."
    )
    parser.add_argument("--input", required=True, help="Path to raw .cnt file.")
    parser.add_argument("--subject", default="", help="Subject id, e.g. sub-p014.")
    parser.add_argument("--known-bad-eeg", default="", help="Comma/space separated EEG bad channels.")
    parser.add_argument("--known-bad-emg", default="", help="Comma/space separated EMG bad channels.")
    parser.add_argument("--notes", default="", help="Optional participant config note.")
    parser.add_argument("--blocks", default="", help="Optional comma list of blocks to keep, default 1,5.")
    parser.add_argument(
        "--mode",
        choices=["full", "preprocess", "analysis", "qc"],
        default="full",
        help="Pipeline mode to run after import.",
    )
    parser.add_argument("--skip-pipeline", action="store_true", help="Only import/cut CNT to FIF.")
    parser.add_argument("--continue-on-error", action="store_true", help="Pass through to run_pipeline.py.")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input CNT not found: {input_path}")

    subject_id = normalize_subject_id(args.subject) if args.subject.strip() else _derive_subject_id(input_path)
    cfg_path = prepare_participant_config(
        subject_id=subject_id,
        input_path=input_path,
        known_bad_eeg=args.known_bad_eeg,
        known_bad_emg=args.known_bad_emg,
        notes=args.notes,
    )

    print(f"Participant config: {cfg_path}")
    print(f"Subject: {subject_id}")

    import_cmd = [
        sys.executable,
        str(PIPELINE_DIR / "preprocessing" / "00_import_participant.py"),
        "--input",
        str(input_path),
        "--subject",
        subject_id,
    ]
    if args.blocks.strip():
        import_cmd.extend(["--blocks", args.blocks.strip()])
    run_checked(import_cmd, PIPELINE_DIR)

    if args.skip_pipeline:
        print("\nImport complete; pipeline run skipped.")
        return

    pipeline_cmd = [
        sys.executable,
        str(PIPELINE_DIR / "run_pipeline.py"),
        "--mode",
        args.mode,
        "--subject",
        subject_id,
    ]
    if args.continue_on_error:
        pipeline_cmd.append("--continue-on-error")
    run_checked(pipeline_cmd, PIPELINE_DIR)


if __name__ == "__main__":
    main()
