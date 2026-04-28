# eeg_pipeline/postprocessing/14_emg_pac_correlation.py
"""
Step 14: Group-level EMG-PAC sensitivity check.

Correlates delta-EMG (B5-B1 EMG PC1 variance) with delta-PAC (B5-B1 PAC z)
across participants. If r is weak, PAC fatigue effect is robust to EMG.

Also correlates within-block EMG variance with PAC z-score.
"""
import sys
import argparse
import os
from pathlib import Path

pipeline_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pipeline_dir))

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils_io import parse_subject_filter, subject_matches

OUTPUT_DIR = pipeline_dir / "outputs" / "features"
FIG_DIR = pipeline_dir / "outputs" / "analysis_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _corr_pair(x, y):
    """Return correlations only when enough non-constant paired data exist."""
    x = pd.Series(x, dtype="float64")
    y = pd.Series(y, dtype="float64")
    mask = x.notna() & y.notna()
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or x.nunique() < 2 or y.nunique() < 2:
        return np.nan, np.nan, np.nan, np.nan
    r, p = stats.pearsonr(x, y)
    rho, p_rho = stats.spearmanr(x, y)
    return r, p, rho, p_rho


def main():
    parser = argparse.ArgumentParser(description="Group-level EMG-PAC sensitivity check.")
    parser.add_argument(
        "--subject",
        default=os.environ.get("EEG_SUBJECT_FILTER", ""),
        help="Optional subject filter for the input rows.",
    )
    args = parser.parse_args()
    selected_subjects = parse_subject_filter(args.subject)

    # Load EMG block-level summary
    emg_path = OUTPUT_DIR / "emg_covariates_block.csv"
    if not emg_path.exists():
        raise FileNotFoundError(f"{emg_path} not found. Run step 13 first.")
    emg_df = pd.read_csv(emg_path)

    # Load PAC z-scores
    pac_path = OUTPUT_DIR / "pac_between_features.csv"
    if not pac_path.exists():
        raise FileNotFoundError(f"{pac_path} not found. Run step 08 first.")
    pac_df = pd.read_csv(pac_path)

    # Get the PAC column (should be pac_between_C_broad_F_C_broad_P)
    pac_cols = [c for c in pac_df.columns if c.startswith("pac_between")]
    if not pac_cols:
        raise KeyError("No pac_between columns found.")
    pac_col = pac_cols[0]

    # Merge EMG + PAC
    merged = pd.merge(
        emg_df[["subject", "block", "emg_pc1_std"]],
        pac_df[["subject", "block", pac_col]],
        on=["subject", "block"],
        how="inner",
    )
    merged.rename(columns={pac_col: "pac_z"}, inplace=True)
    if selected_subjects:
        merged = merged[
            merged["subject"].apply(lambda s: subject_matches(str(s), selected_subjects))
        ].reset_index(drop=True)

    print("=" * 60)
    print("Step 14: EMG-PAC Group-Level Sensitivity")
    print("=" * 60)

    if merged.empty:
        print("No overlapping EMG and PAC rows found.")
        return

    # --- Analysis 1: Within-block correlation (EMG std vs PAC z) ---
    print("\n--- Within-block: EMG variance vs PAC z-score ---")
    for block in sorted(merged["block"].unique()):
        bdf = merged[merged["block"] == block]
        r, p, rho, p_rho = _corr_pair(bdf["emg_pc1_std"], bdf["pac_z"])
        print(f"  Block {block}: r={r:.3f} (p={p:.3f}), rho={rho:.3f} (p={p_rho:.3f}), N={len(bdf)}")

    # --- Analysis 2: Delta correlation ---
    subjects = sorted(set(merged[merged["block"] == 1]["subject"]) &
                       set(merged[merged["block"] == 5]["subject"]))

    deltas = []
    for subj in subjects:
        b1 = merged[(merged["subject"] == subj) & (merged["block"] == 1)]
        b5 = merged[(merged["subject"] == subj) & (merged["block"] == 5)]
        if b1.empty or b5.empty:
            continue
        delta_emg = float(b5["emg_pc1_std"].values[0] - b1["emg_pc1_std"].values[0])
        delta_pac = float(b5["pac_z"].values[0] - b1["pac_z"].values[0])
        deltas.append({
            "subject": subj,
            "delta_emg_std": delta_emg,
            "delta_pac_z": delta_pac,
            "b1_pac_z": float(b1["pac_z"].values[0]),
            "b5_pac_z": float(b5["pac_z"].values[0]),
            "b1_emg_std": float(b1["emg_pc1_std"].values[0]),
            "b5_emg_std": float(b5["emg_pc1_std"].values[0]),
        })

    delta_df = pd.DataFrame(deltas)
    print(f"\n--- Delta analysis (B5 - B1): N = {len(delta_df)} ---")
    if len(delta_df) < 2:
        print("Not enough complete participants for delta correlation.")
        return

    r_delta, p_delta, rho_delta, p_rho_delta = _corr_pair(
        delta_df["delta_emg_std"], delta_df["delta_pac_z"]
    )
    print(f"  Pearson:  r = {r_delta:.3f}, p = {p_delta:.3f}")
    print(f"  Spearman: rho = {rho_delta:.3f}, p = {p_rho_delta:.3f}")
    print()
    print(delta_df.to_string(index=False))
    if not np.isfinite(r_delta):
        out_csv = OUTPUT_DIR / "emg_pac_delta_correlation.csv"
        delta_df.to_csv(out_csv, index=False)
        print("\nDelta correlation not estimable because the paired data are constant or incomplete.")
        print(f"  Saved delta table: {out_csv}")
        return

    # --- Interpretation ---
    print(f"\n--- Interpretation ---")
    if abs(r_delta) < 0.3:
        print("  WEAK correlation: PAC fatigue effect is NOT driven by EMG changes.")
        print("  The delta-PAC finding is robust to EMG contamination.")
    elif abs(r_delta) < 0.6:
        print("  MODERATE correlation: EMG may partly confound the PAC finding.")
        print("  Variant C (corrected PAC) is recommended.")
    else:
        print("  STRONG correlation: EMG is a plausible confound for PAC.")
        print("  Variant C (corrected PAC) is essential.")

    # --- Figure ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel A: B1 EMG vs B1 PAC
    ax = axes[0]
    b1 = merged[merged["block"] == 1]
    ax.scatter(b1["emg_pc1_std"], b1["pac_z"], s=80, c="#1E88E5", edgecolors="k", zorder=5)
    for _, row in b1.iterrows():
        ax.annotate(row["subject"].replace("sub-", ""), (row["emg_pc1_std"], row["pac_z"]),
                     fontsize=7, ha="left", va="bottom")
    r1, p1, _, _ = _corr_pair(b1["emg_pc1_std"], b1["pac_z"])
    ax.set_xlabel("EMG PC1 SD", fontsize=11)
    ax.set_ylabel("PAC z-score", fontsize=11)
    ax.set_title(f"Block 1: r={r1:.2f}, p={p1:.3f}", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel B: B5 EMG vs B5 PAC
    ax = axes[1]
    b5 = merged[merged["block"] == 5]
    ax.scatter(b5["emg_pc1_std"], b5["pac_z"], s=80, c="#E53935", edgecolors="k", zorder=5)
    for _, row in b5.iterrows():
        ax.annotate(row["subject"].replace("sub-", ""), (row["emg_pc1_std"], row["pac_z"]),
                     fontsize=7, ha="left", va="bottom")
    r5, p5, _, _ = _corr_pair(b5["emg_pc1_std"], b5["pac_z"])
    ax.set_xlabel("EMG PC1 SD", fontsize=11)
    ax.set_ylabel("PAC z-score", fontsize=11)
    ax.set_title(f"Block 5: r={r5:.2f}, p={p5:.3f}", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel C: Delta EMG vs Delta PAC
    ax = axes[2]
    ax.scatter(delta_df["delta_emg_std"], delta_df["delta_pac_z"],
               s=100, c="#7B1FA2", edgecolors="k", zorder=5)
    for _, row in delta_df.iterrows():
        ax.annotate(row["subject"].replace("sub-", ""),
                     (row["delta_emg_std"], row["delta_pac_z"]),
                     fontsize=7, ha="left", va="bottom")
    # Regression line
    x = delta_df["delta_emg_std"].values
    y = delta_df["delta_pac_z"].values
    m, b_int, _, _, _ = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 50)
    ax.plot(x_line, b_int + m * x_line, "k--", lw=1.5, alpha=0.5)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.axvline(0, color="gray", lw=0.5, ls="--")
    ax.set_xlabel("Delta EMG PC1 SD (B5 - B1)", fontsize=11)
    ax.set_ylabel("Delta PAC z (B5 - B1)", fontsize=11)
    ax.set_title(f"Delta: r={r_delta:.2f}, p={p_delta:.3f}", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("EMG-PAC Sensitivity\nDoes EMG change predict PAC change?",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = FIG_DIR / "emg_pac_sensitivity_variantB.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n  Saved figure: {out}")

    # Save delta table
    out_csv = OUTPUT_DIR / "emg_pac_delta_correlation.csv"
    delta_df.to_csv(out_csv, index=False)
    print(f"  Saved delta table: {out_csv}")


if __name__ == "__main__":
    main()
