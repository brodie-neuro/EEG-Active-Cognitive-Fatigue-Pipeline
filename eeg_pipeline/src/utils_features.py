# src/utils_features.py
"""
Shared utility functions for analysis scripts (Steps 10-16).
Handles per-block epoch loading, change-score computation,
and behavioural data integration.
"""
import numpy as np
import pandas as pd
import mne
from pathlib import Path

from src.utils_io import discover_subjects, derivative_dirs, normalize_subject_id


def save_feature_tables(df, filename, include_global=True, include_combined=True,
                        include_per_subject=True):
    """Save a feature table to global, combined, and per-subject directories.

    Ported from root pipeline. Creates:
      - outputs/features/{filename}              (global, legacy)
      - outputs/combined/features/{filename}     (combined)
      - outputs/sub-{name}/features/{filename}   (per-subject)

    Parameters
    ----------
    df : pd.DataFrame
        Feature table to save.
    filename : str
        CSV filename (e.g. 'pac_between_features.csv').
    include_global : bool
        Save to outputs/features/.
    include_combined : bool
        Save to outputs/combined/features/.
    include_per_subject : bool
        Save per-subject copies to outputs/sub-*/features/.

    Returns
    -------
    list of Path
        Paths where the file was saved.
    """
    outputs_dir = Path(__file__).resolve().parents[1] / "outputs"
    filename = Path(filename).name
    saved_paths = []

    if include_global:
        global_dir = outputs_dir / "features"
        global_dir.mkdir(parents=True, exist_ok=True)
        global_path = global_dir / filename
        df.to_csv(global_path, index=False)
        saved_paths.append(global_path)

    if include_combined:
        combined_dir = outputs_dir / "combined" / "features"
        combined_dir.mkdir(parents=True, exist_ok=True)
        combined_path = combined_dir / filename
        df.to_csv(combined_path, index=False)
        saved_paths.append(combined_path)

    if include_per_subject and "subject" in df.columns:
        for subj, sub_df in df.groupby("subject"):
            subj_dir = outputs_dir / str(subj) / "features"
            subj_dir.mkdir(parents=True, exist_ok=True)
            subj_path = subj_dir / filename
            sub_df.to_csv(subj_path, index=False)
            saved_paths.append(subj_path)

    return saved_paths


def load_block_epochs(subj, block, epoch_type, epochs_dir):
    """
    Load epoch file for a specific subject and block.

    Naming convention: {subj}_block{block}_{epoch_type}_clean-epo.fif
    Falls back to: {subj}_{epoch_type}_clean-epo.fif (legacy single-block)

    Parameters
    ----------
    subj : str
        Subject ID (e.g., 'sub-TEST01').
    block : int
        Block number (e.g., 1 or 5).
    epoch_type : str
        'p3b' or 'pac'.
    epochs_dir : Path
        Directory containing epoch files.

    Returns
    -------
    mne.Epochs or None
        Loaded epochs, or None if file not found.
    """
    subj = normalize_subject_id(subj)
    epochs_dir = Path(epochs_dir)
    fname_block = f"{subj}_block{block}_{epoch_type}_clean-epo.fif"
    fname_legacy = f"{subj}_{epoch_type}_clean-epo.fif"

    # Candidate folders: explicit folder first, then new per-subject + legacy layouts.
    cand_dirs = [epochs_dir]
    cand_dirs.extend(derivative_dirs("epochs_clean", subject=subj, include_legacy=True))

    seen: set[str] = set()
    unique_dirs: list[Path] = []
    for d in cand_dirs:
        key = str(d.resolve()) if d.exists() else str(d)
        if key in seen:
            continue
        seen.add(key)
        unique_dirs.append(d)

    # Primary: per-block naming
    for d in unique_dirs:
        block_file = d / fname_block
        if block_file.exists():
            return mne.read_epochs(block_file, preload=True, verbose=False)

    # Fallback: legacy naming (no block in filename)
    for d in unique_dirs:
        legacy_file = d / fname_legacy
        if legacy_file.exists():
            print(f"  Warning: Using legacy file (no block info): {legacy_file.name}")
            return mne.read_epochs(legacy_file, preload=True, verbose=False)

    return None


def get_subjects_with_blocks(epochs_dir, epoch_type, blocks=(1, 5)):
    """
    Find subjects that have epoch files for all requested blocks.

    Returns
    -------
    list of str
        Subject IDs with complete block data.
    """
    return discover_subjects(
        epochs_dir=epochs_dir,
        blocks=blocks,
        epoch_type=epoch_type,
        require_all_blocks=True,
    )


def compute_delta(block1_val, block5_val):
    """Compute change score: Block 5 - Block 1."""
    if np.isnan(block1_val) or np.isnan(block5_val):
        return np.nan
    return block5_val - block1_val


def load_behavioral_data(path):
    """
    Load behavioural data CSV containing d' values.

    Expected columns: subject, dprime_block1, dprime_block5, delta_dprime
    Returns None if path is None or file doesn't exist.
    """
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        print(f"Behavioural data file not found: {path}")
        return None

    df = pd.read_csv(path)
    required = ['subject']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Behavioural data missing columns: {missing}")
        return None

    return df


def get_node_channels(node_name, cfg):
    """Get channel list for a node from config."""
    return cfg.get('nodes', {}).get(node_name, [])


def get_region_nodes(region_name, cfg):
    """Get node names belonging to a region from config."""
    return cfg.get('regions', {}).get(region_name, [])


def available_channels(ch_names_requested, ch_names_available):
    """Return channels from requested list that exist in the data."""
    return [ch for ch in ch_names_requested if ch in ch_names_available]


def get_excluded_channels():
    """Load the pre-specified analysis exclusion list from parameters.json.

    These channels are excluded from ROI-based analyses only (theta, PAC,
    topography), not from preprocessing. The list was pre-specified based
    on pilot QC (WSI > median + 3*MAD).

    Returns
    -------
    list of str
        Channel names to exclude, or empty list if not configured.
    """
    from src.utils_config import get_param
    excl_cfg = get_param('exclude_channels_analysis', default={})
    return excl_cfg.get('channels', [])


def filter_excluded_channels(ch_names):
    """Remove pre-specified exclusion channels from a channel list.

    Parameters
    ----------
    ch_names : list of str
        Channel names to filter.

    Returns
    -------
    filtered : list of str
        Channels with exclusion list removed.
    excluded : list of str
        Channels that were actually removed (intersection).
    """
    excl = set(get_excluded_channels())
    # Case-insensitive matching (handle Fp1 vs FP1 variants)
    excl_lower = {ch.lower() for ch in excl}
    filtered = [ch for ch in ch_names if ch.lower() not in excl_lower]
    excluded = [ch for ch in ch_names if ch.lower() in excl_lower]
    return filtered, excluded
