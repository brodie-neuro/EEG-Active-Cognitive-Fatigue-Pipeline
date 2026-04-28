# postprocessing/11_merge_features.py
"""
Merge core H1/H2/H3 feature CSVs into a single wide-format DataFrame.

Reads:
  - p3b_features.csv
  - pac_between_features.csv
  - alpha_gamma_pac_features.csv

Outputs:
  - merged_wide.csv  (one row per subject x block, core features as columns)
"""
import os
import sys
import argparse
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

from src.utils_io import parse_subject_filter, subject_matches

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
    parser = argparse.ArgumentParser(description="Merge core H1/H2/H3 feature CSVs.")
    parser.add_argument(
        "--subject",
        default=os.environ.get("EEG_SUBJECT_FILTER", ""),
        help="Optional subject filter for the merged output.",
    )
    args = parser.parse_args()
    selected_subjects = parse_subject_filter(args.subject)

    print("=== Merging Feature Files ===\n")

    # --- Load all feature files ---
    p3b = load_feature("p3b_features.csv")
    pac_between = load_feature("pac_between_features.csv")
    alpha_gamma_pac = load_feature("alpha_gamma_pac_features.csv")
    # EMG sensitivity outputs are generated after step 13 and reviewed
    # separately before final high-frequency inclusion decisions.
    block_dfs = [df for df in [p3b, pac_between, alpha_gamma_pac] if df is not None]

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
    if selected_subjects and "subject" in merged.columns:
        merged = merged[
            merged["subject"].apply(lambda s: subject_matches(str(s), selected_subjects))
        ].reset_index(drop=True)

    out_path = FEATURES_DIR / "merged_wide.csv"
    merged.to_csv(out_path, index=False)
    print(f"\nMerged features saved: {out_path}")
    print(f"Shape: {merged.shape}")
    print(f"Columns: {list(merged.columns)}")
    print(f"\nPreview:\n{merged.to_string(index=False)}")


if __name__ == "__main__":
    main()
