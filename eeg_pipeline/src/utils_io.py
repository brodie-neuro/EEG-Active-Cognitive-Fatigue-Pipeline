# src/utils_io.py
import yaml
import mne
from pathlib import Path

def load_config(path: str | None = None):
    """Load study.yml no matter where the script is launched from."""
    if path is None:
        # .../EEG_study_2/eeg_pipeline/src/utils_io.py
        base = Path(__file__).resolve().parents[1]        # .../eeg_pipeline
        path = base / "config" / "study.yml"
    with open(path, "r") as f:
        return yaml.safe_load(f)

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
        raw = mne.io.read_raw_eeglab(str(p), preload=True, verbose="ERROR")  # .set
    else:
        raise ValueError(f"Unsupported or unknown format for file: {p.name}")

    raw.set_montage(montage, on_missing="ignore")
    return raw

def save_clean_raw(raw, out_dir, subj_id, suffix):
    out_dir = Path(out_dir)
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
        'cleaned', 'referenced', 'zapline', 'ica', 'asr',
        'p3b', 'pac', 'clean'
    ):
        return parts[0]                 # 'sub-TEST01_block1'
    return stem
