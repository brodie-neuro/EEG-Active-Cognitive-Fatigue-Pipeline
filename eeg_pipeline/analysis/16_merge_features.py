# eeg_pipeline/analysis/16_merge_features.py
"""
Merge all individual feature CSVs into a single wide-format DataFrame.

Reads:
  - p3b_features.csv
  - theta_power_features.csv
  - alpha_power_features.csv
  - theta_freq_features.csv
  - iaf_features.csv
  - pac_local_features.csv
  - pac_between_features.csv

Outputs:
  - merged_wide.csv  (one row per subject x block, all features as columns)
"""
import sys
from pathlib import Path
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
    alpha_power = load_feature("alpha_power_features.csv")
    theta_freq = load_feature("theta_freq_features.csv")
    pac_local = load_feature("pac_local_features.csv")
    pac_between = load_feature("pac_between_features.csv")
    iaf = load_feature("iaf_features.csv")

    # --- Start with the most complete subject x block file ---
    block_dfs = [df for df in [p3b, theta_power, alpha_power, theta_freq,
                                pac_local, pac_between] if df is not None]

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

    # --- IAF is per-subject (not per-block), merge differently ---
    if iaf is not None:
        # IAF has timepoint (pre/post). Use pre-task IAF for both blocks
        iaf_pre = iaf[iaf['timepoint'] == 'pre'][['subject', 'iaf', 'aperiodic_exp']]
        iaf_pre = iaf_pre.rename(columns={
            'iaf': 'iaf_pre',
            'aperiodic_exp': 'aperiodic_exp_pre'
        })

        # Also get post-task IAF where available
        iaf_post = iaf[iaf['timepoint'] == 'post'][['subject', 'iaf', 'aperiodic_exp']]
        iaf_post = iaf_post.rename(columns={
            'iaf': 'iaf_post',
            'aperiodic_exp': 'aperiodic_exp_post'
        })

        merged = merged.merge(iaf_pre, on='subject', how='left')
        merged = merged.merge(iaf_post, on='subject', how='left')

        # IAF shift (post - pre)
        if 'iaf_pre' in merged.columns and 'iaf_post' in merged.columns:
            merged['iaf_shift'] = merged['iaf_post'] - merged['iaf_pre']

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
