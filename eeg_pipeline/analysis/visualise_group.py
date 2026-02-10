# eeg_pipeline/analysis/visualise_group.py
"""
Group-level visualisation from merged_wide.csv.

Generates publication-quality figures comparing Block 1 (baseline) vs
Block 5 (fatigued) across all subjects:

  1. Paired comparison panels (violin + paired scatter)
  2. Summary statistics table (saved as CSV)
  3. Effect size / p-value annotation

Usage:
    python visualise_group.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

pipeline_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pipeline_dir))

FEATURES_DIR = pipeline_dir / "outputs" / "features"
FIG_DIR = pipeline_dir / "outputs" / "group_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# --- Feature definitions for plotting ---
# (column_name, display_label, unit, higher_in_fatigue_expected)
FEATURES = [
    ('p3b_amp_uV',       'P3b Amplitude',    'uV',     False),
    ('p3b_lat_ms',       'P3b Latency',      'ms',     True),
    ('theta_power_log',  'Theta Power',      'log(uV^2/Hz)', True),
    ('alpha_power_log',  'Alpha Power',       'log(uV^2/Hz)', False),
    ('f_theta',          'Theta Frequency',  'Hz',     False),
    ('pac_RF_Fz',        'Local PAC (Fz)',   'z-MI',   False),
    ('pac_between_RF_RP','Between PAC (RF-RP)', 'z-MI', False),
]


def cohens_d(x1, x2):
    """Compute Cohen's d for paired samples."""
    diff = np.array(x2) - np.array(x1)
    return np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0.0


def plot_paired_panel(df, features, save_path):
    """
    Create a multi-panel figure with violin + paired scatter for each feature.

    Parameters
    ----------
    df : pd.DataFrame
        merged_wide.csv with columns: subject, block, + feature columns.
    features : list of tuples
        Each: (col_name, label, unit, expected_direction).
    save_path : Path
        Output figure path.
    """
    # Filter to features that exist in the data
    valid_features = [(c, l, u, d) for c, l, u, d in features if c in df.columns]
    n = len(valid_features)
    if n == 0:
        print("No valid features found for plotting.")
        return

    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5.5 * nrows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Get paired data
    b1 = df[df['block'] == 1].sort_values('subject').reset_index(drop=True)
    b5 = df[df['block'] == 5].sort_values('subject').reset_index(drop=True)

    if len(b1) == 0 or len(b5) == 0:
        print("Need both Block 1 and Block 5 data for paired plots.")
        return

    for idx, (col, label, unit, _) in enumerate(valid_features):
        ax = axes[idx]
        vals1 = b1[col].dropna().values
        vals5 = b5[col].dropna().values

        # Ensure paired (same subjects in both)
        min_n = min(len(vals1), len(vals5))
        vals1 = vals1[:min_n]
        vals5 = vals5[:min_n]

        if min_n < 1:
            ax.set_title(label, fontsize=12, fontweight='bold')
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue

        # Violin plot
        parts = ax.violinplot([vals1, vals5], positions=[0, 1],
                              showmeans=True, showmedians=False)
        for pc, color in zip(parts['bodies'], ['#64B5F6', '#EF5350']):
            pc.set_facecolor(color)
            pc.set_alpha(0.4)
        parts['cmeans'].set_color('black')
        parts['cbars'].set_color('grey')
        parts['cmins'].set_color('grey')
        parts['cmaxes'].set_color('grey')

        # Paired scatter with connecting lines
        for i in range(min_n):
            ax.plot([0, 1], [vals1[i], vals5[i]], 'o-',
                    color='#455A64', alpha=0.5, markersize=5, linewidth=1)

        # Mean markers
        ax.scatter([0], [np.mean(vals1)], color='#1565C0', s=100, zorder=5,
                   edgecolors='white', linewidths=1.5, marker='D')
        ax.scatter([1], [np.mean(vals5)], color='#C62828', s=100, zorder=5,
                   edgecolors='white', linewidths=1.5, marker='D')

        # Stats
        if min_n >= 2:
            t_stat, p_val = stats.ttest_rel(vals1, vals5)
            d = cohens_d(vals1, vals5)
            sig = '*' if p_val < 0.05 else ('~' if p_val < 0.10 else 'ns')
            ax.text(0.5, 0.97,
                    f'd = {d:.2f}, p = {p_val:.3f} {sig}',
                    ha='center', va='top', transform=ax.transAxes,
                    fontsize=9, color='#37474F',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#ECEFF1',
                              edgecolor='#B0BEC5', alpha=0.8))

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Block 1\n(Baseline)', 'Block 5\n(Fatigued)'], fontsize=10)
        ax.set_ylabel(f'{label} ({unit})', fontsize=10)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Hide unused axes
    for idx in range(len(valid_features), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Block 1 vs Block 5: Paired Comparisons',
                 fontsize=16, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved paired comparison panel: {save_path}")


def generate_summary_table(df, features, save_path):
    """Generate a summary statistics table as CSV."""
    valid = [(c, l, u, d) for c, l, u, d in features if c in df.columns]
    rows = []

    b1 = df[df['block'] == 1]
    b5 = df[df['block'] == 5]

    for col, label, unit, _ in valid:
        v1 = b1[col].dropna().values
        v5 = b5[col].dropna().values
        min_n = min(len(v1), len(v5))

        row = {
            'Feature': label,
            'Unit': unit,
            'N': min_n,
            'B1_Mean': np.mean(v1) if len(v1) > 0 else np.nan,
            'B1_SD': np.std(v1, ddof=1) if len(v1) > 1 else np.nan,
            'B5_Mean': np.mean(v5) if len(v5) > 0 else np.nan,
            'B5_SD': np.std(v5, ddof=1) if len(v5) > 1 else np.nan,
        }

        if min_n >= 2:
            v1p, v5p = v1[:min_n], v5[:min_n]
            row['Delta_Mean'] = np.mean(v5p - v1p)
            t, p = stats.ttest_rel(v1p, v5p)
            row['t'] = t
            row['p'] = p
            row['Cohens_d'] = cohens_d(v1p, v5p)
        else:
            row.update({'Delta_Mean': np.nan, 't': np.nan, 'p': np.nan, 'Cohens_d': np.nan})

        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(save_path, index=False, float_format='%.4f')
    print(f"Saved summary table: {save_path}")
    print(summary.to_string(index=False))
    return summary


def plot_iaf_shift(df, save_path):
    """Plot individual alpha frequency pre vs post shift."""
    if 'iaf_pre' not in df.columns or 'iaf_post' not in df.columns:
        print("IAF columns not found, skipping IAF shift plot.")
        return

    # IAF is per-subject (same across blocks), take unique subjects
    iaf_df = df[['subject', 'iaf_pre', 'iaf_post']].drop_duplicates('subject').dropna()

    if len(iaf_df) < 1:
        print("No IAF data available.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    pre = iaf_df['iaf_pre'].values
    post = iaf_df['iaf_post'].values

    # Paired scatter
    for i in range(len(pre)):
        ax.plot([0, 1], [pre[i], post[i]], 'o-', color='#455A64',
                alpha=0.6, markersize=8, linewidth=1.5)

    # Mean markers
    ax.scatter([0], [np.mean(pre)], color='#1565C0', s=120, zorder=5,
               edgecolors='white', linewidths=2, marker='D', label=f'Mean Pre: {np.mean(pre):.2f} Hz')
    ax.scatter([1], [np.mean(post)], color='#C62828', s=120, zorder=5,
               edgecolors='white', linewidths=2, marker='D', label=f'Mean Post: {np.mean(post):.2f} Hz')

    if len(pre) >= 2:
        t, p = stats.ttest_rel(pre, post)
        d = cohens_d(pre, post)
        ax.set_title(f'IAF Shift (Pre vs Post Fatigue)\nd = {d:.2f}, p = {p:.3f}',
                     fontsize=13, fontweight='bold')
    else:
        ax.set_title('IAF Shift (Pre vs Post Fatigue)', fontsize=13, fontweight='bold')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Pre-Task', 'Post-Task'], fontsize=11)
    ax.set_ylabel('Individual Alpha Frequency (Hz)', fontsize=11)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved IAF shift plot: {save_path}")


def plot_delta_bar(df, features, save_path):
    """Bar chart of change scores (Delta = B5 - B1) with error bars."""
    valid = [(c, l, u, d) for c, l, u, d in features if c in df.columns]
    if not valid:
        return

    b1 = df[df['block'] == 1].sort_values('subject').reset_index(drop=True)
    b5 = df[df['block'] == 5].sort_values('subject').reset_index(drop=True)
    min_n = min(len(b1), len(b5))
    if min_n < 1:
        return

    names = []
    means = []
    sems = []
    colors = []

    for col, label, unit, expect_increase in valid:
        v1 = b1[col].dropna().values[:min_n]
        v5 = b5[col].dropna().values[:min_n]
        if len(v1) == 0 or len(v5) == 0:
            continue
        delta = v5 - v1

        # Normalize to % change for comparability
        baseline_mean = np.mean(v1) if np.mean(v1) != 0 else 1
        pct_change = 100 * np.mean(delta) / abs(baseline_mean)

        names.append(label)
        means.append(pct_change)
        sems.append(100 * np.std(delta, ddof=1) / (abs(baseline_mean) * np.sqrt(len(delta)))
                     if len(delta) > 1 else 0)
        colors.append('#EF5350' if pct_change > 0 else '#64B5F6')

    if not names:
        return

    fig, ax = plt.subplots(1, 1, figsize=(max(8, len(names) * 1.5), 5))
    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=sems, capsize=4, color=colors, edgecolor='#37474F',
                  linewidth=0.8, alpha=0.85)
    ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=10)
    ax.set_ylabel('% Change (B5 - B1)', fontsize=11)
    ax.set_title('Fatigue-Related Changes (B5 vs B1)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved delta bar chart: {save_path}")


def main():
    print("=== Group-Level Visualisation ===\n")

    merged_path = FEATURES_DIR / "merged_wide.csv"
    if not merged_path.exists():
        print(f"merged_wide.csv not found at {merged_path}.")
        print("Run 16_merge_features.py first.")
        return

    df = pd.read_csv(merged_path)
    n_subjects = df['subject'].nunique()
    n_blocks = df['block'].nunique()
    print(f"Loaded {len(df)} rows: {n_subjects} subjects, {n_blocks} blocks")
    print(f"Columns: {list(df.columns)}\n")

    # 1. Paired comparison panels
    plot_paired_panel(df, FEATURES,
                      FIG_DIR / "group_paired_comparisons.png")

    # 2. Summary statistics table
    generate_summary_table(df, FEATURES,
                           FIG_DIR / "group_summary_stats.csv")

    # 3. IAF shift plot
    plot_iaf_shift(df, FIG_DIR / "group_iaf_shift.png")

    # 4. Delta bar chart
    plot_delta_bar(df, FEATURES,
                   FIG_DIR / "group_delta_barchart.png")

    print(f"\nAll group figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
