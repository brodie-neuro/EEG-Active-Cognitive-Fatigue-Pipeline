# postprocessing/14_merge_features.py
"""
Merge all individual feature CSVs into a single wide-format DataFrame.

Reads:
  - p3b_features.csv
  - theta_power_features.csv

  - theta_freq_features.csv
  - pac_between_features.csv
  - emg_covariates_block.csv

Outputs:
  - merged_wide.csv  (one row per subject x block, all features as columns)
"""
import os
import sys
from pathlib import Path

# Deterministic BLAS/LAPACK thread limits: set before scientific imports.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MNE_DONTWRITE_HOME"] = "true"
os.environ.setdefault("_MNE_FAKE_HOME_DIR", os.path.dirname(os.path.dirname(__file__)))
import pandas as pd

pipeline_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pipeline_dir))

FEATURES_DIR = pipeline_dir / "outputs" / "features"


def load_feature(name, merge_key=None):
    """Load a feature CSV if it exists."""
    fpath = FEATURES_DIR / name
    if not fpath.exists():
        print(f"  Not found: {name}")
        return None
    df = pd.read_csv(fpath)
    print(f"  Loaded {name}: {len(df)} rows, columns: {list(df.columns)}")
    return df


def main():
    print("=== Merging Feature Files ===\n")

    # --- Load all feature files ---
    p3b = load_feature("p3b_features.csv")
    theta_power = load_feature("theta_power_features.csv")

    theta_freq = load_feature("theta_freq_features.csv")
    theta_stim = load_feature("theta_stim_features.csv")
    pac_between = load_feature("pac_between_features.csv")
    emg_pca = load_feature("emg_covariates_block.csv")

    # --- Start with the most complete subject x block file ---
    block_dfs = [df for df in [p3b, theta_power, theta_freq, theta_stim,
                                pac_between, emg_pca] if df is not None]

    if not block_dfs:
        print("No feature files found. Run analysis steps first.")
        return

    # Merge all block-level features on (subject, block)
    merged = block_dfs[0]
    for df in block_dfs[1:]:
        # Avoid duplicate columns during merge
        overlap_cols = [c for c in df.columns if c in merged.columns
                       and c not in ['subject', 'block']]
        if overlap_cols:
            df = df.drop(columns=overlap_cols)
        merged = merged.merge(df, on=['subject', 'block'], how='outer')

    # --- Sort and save ---
    merged = merged.sort_values(['subject', 'block']).reset_index(drop=True)

    out_path = FEATURES_DIR / "merged_wide.csv"
    merged.to_csv(out_path, index=False)
    print(f"\nMerged features saved: {out_path}")
    print(f"Shape: {merged.shape}")
    print(f"Columns: {list(merged.columns)}")
    print(f"\nPreview:\n{merged.to_string(index=False)}")


if __name__ == "__main__":
    main()
