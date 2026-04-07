# src/utils_io.py
import os
import json
import re
from typing import Iterable
import yaml
import mne
import mne.channels
import tempfile
import numpy as np
from pathlib import Path


def _sanitize_eeglab_events(set_path: Path) -> Path | None:
    """Create a temporary .set with invalid empty-latency events removed.

    Some EEGLAB exports include one trailing event with empty latency, which
    causes MNE's EEGLAB reader to fail with an inhomogeneous-shape ValueError.
    """
    try:
        from scipy.io import loadmat, savemat
    except Exception:
        return None

    try:
        mat = loadmat(str(set_path), squeeze_me=False, struct_as_record=True)
        events = mat.get("event")
        if not isinstance(events, np.ndarray):
            return None
        if events.dtype.names is None or "latency" not in events.dtype.names:
            return None
        if events.ndim != 2 or events.shape[0] != 1:
            return None

        keep_idx = []
        for i in range(events.shape[1]):
            lat = np.asarray(events["latency"][0, i])
            if lat.size == 0:
                continue
            keep_idx.append(i)

        if len(keep_idx) == events.shape[1]:
            return None
        if not keep_idx:
            return None

        mat["event"] = events[:, keep_idx]

        tmp = tempfile.NamedTemporaryFile(prefix="eeglab_fix_", suffix=".set", delete=False)
        tmp_path = Path(tmp.name)
        tmp.close()
        savemat(str(tmp_path), mat)
        return tmp_path
    except Exception:
        return None


def pipeline_root() -> Path:
    """Return eeg_pipeline root directory."""
    return Path(__file__).resolve().parents[1]


def outputs_root() -> Path:
    """Return the outputs directory under eeg_pipeline."""
    return pipeline_root() / "outputs"


def subject_base(subject_id: str) -> str:
    """Return subject-only id (strip block suffix), e.g. sub-001_block5 -> sub-001."""
    text = str(subject_id or "").strip()
    if not text:
        return ""
    if "_block" in text:
        text = text.split("_block", 1)[0]
    return text


def normalize_subject_id(subject_id: str) -> str:
    """Normalize subject id to sub-* form (preserving case otherwise)."""
    subj = subject_base(subject_id)
    if not subj:
        return ""
    if not subj.lower().startswith("sub-"):
        subj = f"sub-{subj}"
    return subj


def parse_subject_filter(subject_arg: str | None) -> set[str]:
    """Parse '--subject' value into lowercase normalized ids.

    Supports comma and/or whitespace separated values:
    'sub-a,sub-b' or 'sub-a sub-b'.
    """
    text = str(subject_arg or "").strip()
    if not text:
        return set()
    out: set[str] = set()
    for token in re.split(r"[,\s]+", text):
        token = token.strip()
        if not token:
            continue
        norm = normalize_subject_id(token)
        if norm:
            out.add(norm.lower())
    return out


def subject_matches(subject_or_derivative_id: str, selected_subjects: set[str] | None) -> bool:
    """Check whether a subject/block id matches the selected subject filter."""
    if not selected_subjects:
        return True
    norm = normalize_subject_id(subject_or_derivative_id).lower()
    return bool(norm) and norm in selected_subjects


def _subject_from_filename(path_like: str | Path) -> str:
    stem = Path(path_like).stem
    if "_block" in stem:
        return normalize_subject_id(stem.split("_block", 1)[0])
    if "_" in stem:
        return normalize_subject_id(stem.split("_", 1)[0])
    return normalize_subject_id(stem)


def _legacy_derivative_subdir(path: Path) -> str | None:
    """Return derivatives subdir if path is .../outputs/derivatives/<subdir>/..., else None."""
    parts = list(path.parts)
    parts_lower = [p.lower() for p in parts]
    for i in range(len(parts_lower) - 2):
        if parts_lower[i] == "outputs" and parts_lower[i + 1] == "derivatives":
            return parts[i + 2]
    return None


def subject_output_root(subject_id: str) -> Path:
    """Return per-subject output root: outputs/sub-XXX."""
    subj = normalize_subject_id(subject_id)
    return outputs_root() / subj


def subject_derivatives_dir(subject_id: str, derivative_subdir: str) -> Path:
    """Return per-subject derivatives folder."""
    return subject_output_root(subject_id) / "derivatives" / derivative_subdir


def derivative_dirs(
    derivative_subdir: str,
    subject: str | None = None,
    include_legacy: bool = True,
) -> list[Path]:
    """Return candidate derivative directories (new per-subject + legacy flat)."""
    dirs: list[Path] = []
    out_root = outputs_root()
    if subject is None:
        subject = os.environ.get("EEG_SUBJECT_FILTER", "").strip() or None

    if subject:
        filters = parse_subject_filter(subject)
        for sid in sorted(filters):
            dirs.append(out_root / sid / "derivatives" / derivative_subdir)
        if out_root.exists():
            for subj_dir in sorted(out_root.glob("sub-*")):
                subj_name = normalize_subject_id(subj_dir.name).lower()
                if subj_name in filters:
                    dirs.append(subj_dir / "derivatives" / derivative_subdir)
    else:
        if out_root.exists():
            for subj_dir in sorted(out_root.glob("sub-*")):
                dirs.append(subj_dir / "derivatives" / derivative_subdir)

    if include_legacy:
        dirs.append(out_root / "derivatives" / derivative_subdir)

    dedup: list[Path] = []
    seen: set[str] = set()
    for d in dirs:
        key = str(d.resolve()) if d.exists() else str(d)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(d)
    return dedup


def iter_derivative_files(
    derivative_subdir: str,
    pattern: str,
    subject: str | None = None,
    include_legacy: bool = True,
) -> list[Path]:
    """Find derivative files from new per-subject layout and legacy flat layout."""
    selected = parse_subject_filter(subject)
    files: list[Path] = []
    seen: set[str] = set()

    for d in derivative_dirs(derivative_subdir, subject=subject, include_legacy=include_legacy):
        if not d.exists():
            continue
        for f in sorted(d.glob(pattern)):
            if not f.is_file():
                continue
            if selected:
                subj = _subject_from_filename(f)
                if subj.lower() not in selected:
                    continue
            key = str(f.resolve())
            if key in seen:
                continue
            seen.add(key)
            files.append(f)

    return files


def discover_subjects(
    epochs_dir: str | Path | None = None,
    blocks: Iterable[int] = (1, 5),
    epoch_type: str | None = "pac",
    require_all_blocks: bool = True,
) -> list[str]:
    """Discover subjects that have clean epoch files for requested blocks.

    Scans both new per-subject outputs and legacy flat outputs.
    """
    requested = {int(b) for b in blocks}

    dirs: list[Path] = []
    if epochs_dir is not None:
        dirs.append(Path(epochs_dir))
    dirs.extend(derivative_dirs("epochs_clean", include_legacy=True))

    dedup_dirs: list[Path] = []
    seen_dirs: set[str] = set()
    for d in dirs:
        key = str(d.resolve()) if d.exists() else str(d)
        if key in seen_dirs:
            continue
        seen_dirs.add(key)
        dedup_dirs.append(d)

    selected_subjects = parse_subject_filter(os.environ.get("EEG_SUBJECT_FILTER", ""))
    subj_blocks: dict[str, set[int]] = {}
    epoch_mid = f"_{epoch_type}_clean-epo.fif" if epoch_type else "_clean-epo.fif"
    regex = re.compile(r"(.+?)_block(\d+)_")

    for d in dedup_dirs:
        if not d.exists():
            continue
        for f in d.glob(f"*{epoch_mid}"):
            m = regex.match(f.name)
            if not m:
                continue
            subj = normalize_subject_id(m.group(1))
            if selected_subjects and subj.lower() not in selected_subjects:
                continue
            try:
                block = int(m.group(2))
            except ValueError:
                continue
            subj_blocks.setdefault(subj, set()).add(block)

    if not requested:
        return sorted(subj_blocks.keys())
    if require_all_blocks:
        return sorted([s for s, have in subj_blocks.items() if requested.issubset(have)])
    return sorted([s for s, have in subj_blocks.items() if have & requested])


def _normalize_h1_nodes(node_obj) -> dict[str, list[str]] | None:
    """Normalize h1_nodes payload into {node: [channels]}."""
    if isinstance(node_obj, dict) and "h1_nodes" in node_obj:
        node_obj = node_obj.get("h1_nodes")
    if not isinstance(node_obj, dict):
        return None

    out: dict[str, list[str]] = {}
    for node_name, channels in node_obj.items():
        key = str(node_name).strip()
        if not key:
            continue
        if isinstance(channels, str):
            cleaned = [channels.strip()] if channels.strip() else []
        elif isinstance(channels, (list, tuple)):
            cleaned = [str(ch).strip() for ch in channels if str(ch).strip()]
        else:
            continue
        if cleaned:
            out[key] = cleaned
    return out or None


def _load_h1_nodes_override_from_file(path_like: str) -> dict[str, list[str]] | None:
    """Load h1_nodes override from a YAML/JSON file."""
    p = Path(os.path.expandvars(os.path.expanduser(str(path_like).strip())))
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            payload = yaml.safe_load(f)
    except Exception:
        return None
    return _normalize_h1_nodes(payload)


def load_config(path: str | None = None):
    """Load study.yml no matter where the script is launched from."""
    if path is None:
        # .../EEG_study_2/eeg_pipeline/src/utils_io.py
        base = Path(__file__).resolve().parents[1]        # .../eeg_pipeline
        path = base / "config" / "study.yml"
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    # Optional adapter profile (e.g., OpenNeuro dataset mapping).
    try:
        from src.utils_adapter import load_adapter_config, apply_adapter_to_study_cfg
        adapter_cfg = load_adapter_config()
        if adapter_cfg:
            cfg = apply_adapter_to_study_cfg(cfg, adapter_cfg)
    except Exception:
        pass

    # Optional runtime overrides (used by pipeline runner GUI/CLI).
    data_cfg = cfg.setdefault("data", {})
    env_root = os.environ.get("EEG_DATA_ROOT")
    env_pattern = os.environ.get("EEG_DATA_PATTERN")
    env_format = os.environ.get("EEG_DATA_FORMAT")
    if env_root:
        data_cfg["root"] = env_root
    if env_pattern:
        data_cfg["pattern"] = env_pattern
    if env_format:
        data_cfg["format"] = env_format

    # Optional H1 node overrides for sensitivity runs.
    # Precedence (highest last): node-set < file < JSON.
    h1_override = None

    env_nodeset = os.environ.get("EEG_H1_NODESET", "").strip()
    if env_nodeset:
        node_sets = cfg.get("h1_node_sets", {})
        if isinstance(node_sets, dict) and env_nodeset in node_sets:
            h1_override = _normalize_h1_nodes(node_sets.get(env_nodeset))
            if h1_override:
                print(f"[config] Using h1 node-set '{env_nodeset}' from study config.")
            else:
                print(f"[config] WARNING: EEG_H1_NODESET='{env_nodeset}' is invalid; using default h1_nodes.")
        else:
            print(f"[config] WARNING: EEG_H1_NODESET='{env_nodeset}' not found in study config.")

    env_nodes_file = os.environ.get("EEG_H1_NODES_FILE", "").strip()
    if env_nodes_file:
        file_override = _load_h1_nodes_override_from_file(env_nodes_file)
        if file_override:
            h1_override = file_override
            print(f"[config] Using h1_nodes override from file: {env_nodes_file}")
        else:
            print(f"[config] WARNING: could not load EEG_H1_NODES_FILE='{env_nodes_file}'.")

    env_nodes_json = os.environ.get("EEG_H1_NODES_JSON", "").strip()
    if env_nodes_json:
        try:
            payload = json.loads(env_nodes_json)
            json_override = _normalize_h1_nodes(payload)
            if json_override:
                h1_override = json_override
                print("[config] Using h1_nodes override from EEG_H1_NODES_JSON.")
            else:
                print("[config] WARNING: EEG_H1_NODES_JSON parsed but had no valid node mapping.")
        except json.JSONDecodeError as exc:
            print(f"[config] WARNING: invalid EEG_H1_NODES_JSON ({exc}).")

    if h1_override:
        cfg["h1_nodes"] = h1_override

    return cfg


def load_participant_config(subject_id: str) -> dict:
    """Load per-participant config from config/participant_configs/sub-XXX.json.

    Returns dict with at minimum:
        known_bad_eeg: list[str]  — EEG channels to pre-mark as bad
        known_bad_emg: list[str]  — EMG channels to exclude
        notes: str
    Returns empty lists if no config file exists.
    """
    base = subject_base(subject_id)  # strip block suffix
    config_dir = Path(__file__).resolve().parents[1] / "config" / "participant_configs"

    # Try with and without sub- prefix
    candidates = [
        config_dir / f"{base}.json",
        config_dir / f"{subject_id}.json",
    ]

    for p in candidates:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Ensure required keys exist
            data.setdefault("known_bad_eeg", [])
            data.setdefault("known_bad_emg", [])
            data.setdefault("notes", "")
            return data

    return {"known_bad_eeg": [], "known_bad_emg": [], "notes": ""}

def read_raw(path, fmt, montage):
    """Read a raw EEG file either using an explicit format or auto by extension."""
    p = Path(path)
    ext = p.suffix.lower()

    if fmt and fmt != "auto":
        reader = fmt.lower()
    else:
        # infer from extension
        reader = {
            ".vhdr": "brainvision",
            ".edf": "edf",
            ".fif": "fif",
            ".bdf": "bdf",
            ".mff": "egi",
            ".cnt": "cnt",
            ".set": "eeglab",
        }.get(ext, None)

    if reader == "brainvision":
        raw = mne.io.read_raw_brainvision(str(p), preload=True, verbose="ERROR")
    elif reader == "edf":
        raw = mne.io.read_raw_edf(str(p), preload=True, verbose="ERROR")
    elif reader == "fif":
        raw = mne.io.read_raw_fif(str(p), preload=True, verbose="ERROR")
    elif reader == "bdf":
        raw = mne.io.read_raw_bdf(str(p), preload=True, verbose="ERROR")
    elif reader == "egi":
        raw = mne.io.read_raw_egi(str(p), preload=True, verbose="ERROR")  # .mff
    elif reader == "cnt":
        raw = mne.io.read_raw_cnt(str(p), preload=True, verbose="ERROR")
    elif reader == "eeglab":
        try:
            raw = mne.io.read_raw_eeglab(str(p), preload=True, verbose="ERROR")  # .set
        except ValueError as exc:
            msg = str(exc).lower()
            if "inhomogeneous shape" not in msg and "setting an array element with a sequence" not in msg:
                raise
            fixed_path = _sanitize_eeglab_events(p)
            if fixed_path is None:
                raise
            try:
                raw = mne.io.read_raw_eeglab(str(fixed_path), preload=True, verbose="ERROR")
            finally:
                try:
                    fixed_path.unlink(missing_ok=True)
                except Exception:
                    pass
    else:
        raise ValueError(f"Unsupported or unknown format for file: {p.name}")

    # Optional adapter-driven channel renames/types are only for raw imports.
    # FIF files from Step 01 already carry the intended names and channel types.
    if reader != "fif":
        try:
            from src.utils_adapter import load_adapter_config
            adapter_cfg = load_adapter_config()
            rename_cfg = adapter_cfg.get("channel_renames", {}) if adapter_cfg else {}
            if rename_cfg:
                existing = {ch.lower(): ch for ch in raw.ch_names}
                rename_map = {}
                for src, dst in rename_cfg.items():
                    src_key = str(src).strip().lower()
                    actual = existing.get(src_key)
                    if actual and actual != dst:
                        rename_map[actual] = str(dst)
                if rename_map:
                    raw.rename_channels(rename_map)

            # Optional adapter-driven channel types (e.g., mark EEG channels as EOG).
            type_cfg = adapter_cfg.get("channel_types", {}) if adapter_cfg else {}
            if type_cfg:
                type_map = {}
                existing = {ch.lower(): ch for ch in raw.ch_names}
                for ch_name, ch_type in type_cfg.items():
                    actual = existing.get(str(ch_name).strip().lower())
                    if actual:
                        type_map[actual] = str(ch_type)
                if type_map:
                    raw.set_channel_types(type_map, verbose=False)
        except Exception:
            pass

    # Normalize channel names to match montage case (e.g., FP1 -> Fp1, CZ -> Cz).
    try:
        mont = mne.channels.make_standard_montage(montage)
        name_map = {name.lower(): name for name in mont.ch_names}
        rename = {}
        for ch in raw.ch_names:
            target = name_map.get(ch.lower())
            if target and target != ch:
                rename[ch] = target
        if rename:
            raw.rename_channels(rename)
    except Exception:
        pass

    # Only apply montage for raw imports (CNT, etc).
    # FIF files from Step 01 already have correct types + positions.
    if reader != "fif":
        raw.set_montage(montage, on_missing="ignore")
    return raw

def save_clean_raw(raw, out_dir, subj_id, suffix):
    out_dir = Path(out_dir)
    # Route derivatives to per-subject folder layout:
    #   outputs/sub-XXX/derivatives/<step_subdir>/...
    # while keeping compatibility with existing callers that pass:
    #   outputs/derivatives/<step_subdir>
    deriv_subdir = _legacy_derivative_subdir(out_dir)
    if deriv_subdir is not None:
        out_dir = subject_derivatives_dir(subj_id, deriv_subdir)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{subj_id}_{suffix}-raw.fif"
    raw.save(out_path, overwrite=True)
    return str(out_path)

def glob_subjects(root, pattern):
    return sorted([str(p) for p in Path(root).glob(pattern)])

def subject_id_from_path(p):
    """
    Extract subject+block identifier from filename.

    e.g. 'sub-TEST01_block1-task.vhdr' -> 'sub-TEST01_block1'
         'sub-TEST01.vhdr'             -> 'sub-TEST01'
    """
    base = Path(p).stem              # 'sub-TEST01_block1-task'
    # Strip the task suffix (everything after last hyphen if it's a task label)
    if '-task' in base:
        base = base.split('-task')[0]  # 'sub-TEST01_block1'
    elif '-' in base:
        # Fallback: keep as-is if no '-task' pattern
        pass

    try:
        from src.utils_adapter import adapt_subject_id
        return adapt_subject_id(p, base)
    except Exception:
        return base


def subj_id_from_derivative(fif_path):
    """
    Extract subject+block identifier from a derivative .fif filename.

    e.g. 'sub-TEST01_block1_cleaned-raw.fif' -> 'sub-TEST01_block1'
         'sub-TEST01_block5_ica-raw.fif'     -> 'sub-TEST01_block5'
         'sub-TEST01_cleaned-raw.fif'        -> 'sub-TEST01'

    Strips the processing suffix (e.g. '_cleaned-raw', '_ica-raw', '_asr-raw').
    """
    stem = Path(fif_path).stem          # 'sub-TEST01_block1_cleaned-raw'
    # Remove the known processing suffixes
    for suffix in ['-raw', '-epo']:
        if suffix in stem:
            stem = stem.rsplit(suffix, 1)[0]    # 'sub-TEST01_block1_cleaned'
    # Remove the last underscore-separated token (the processing label)
    parts = stem.rsplit('_', 1)
    if len(parts) == 2 and parts[1] in (
        'cleaned', 'referenced', 'notch', 'ica', 'asr',
        'p3b', 'pac', 'clean'
    ):
        return parts[0]                 # 'sub-TEST01_block1'
    return stem
