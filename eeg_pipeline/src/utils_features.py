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
    epochs_dir = Path(epochs_dir)

    # Primary: per-block naming
    block_file = epochs_dir / f"{subj}_block{block}_{epoch_type}_clean-epo.fif"
    if block_file.exists():
        return mne.read_epochs(block_file, preload=True, verbose=False)

    # Fallback: legacy naming (no block in filename)
    legacy_file = epochs_dir / f"{subj}_{epoch_type}_clean-epo.fif"
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
    epochs_dir = Path(epochs_dir)
    all_subjects = set()

    for block in blocks:
        pattern = f"*_block{block}_{epoch_type}_clean-epo.fif"
        files = epochs_dir.glob(pattern)
        subjects = {f.name.split("_block")[0] for f in files}
        if not all_subjects:
            all_subjects = subjects
        else:
            all_subjects &= subjects

    return sorted(all_subjects)


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
