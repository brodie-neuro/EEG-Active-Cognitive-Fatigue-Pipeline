"""Summaries for PAC phase QC outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _table_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows available."
    df = df.copy()
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in cols) + " |")
    return "\n".join(lines)


def make_delta_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create one row per subject with B5-B1 deltas for each band."""
    rows = []
    for subject, sub_df in df.groupby("subject"):
        row = {"subject": subject}
        for band, band_df in sub_df.groupby("band"):
            b1 = band_df[band_df["block"] == 1]
            b5 = band_df[band_df["block"] == 5]
            if b1.empty or b5.empty:
                continue
            b1 = b1.iloc[0]
            b5 = b5.iloc[0]
            prefix = str(band).lower()
            for metric, out_name in [
                ("centre_of_mass_hz", "com_delta_hz"),
                ("peak_frequency_hz", "peak_frequency_delta_hz"),
                ("positive_residual_area", "residual_power_delta"),
            ]:
                v1 = b1.get(metric, np.nan)
                v5 = b5.get(metric, np.nan)
                row[f"{prefix}_{out_name}"] = float(v5 - v1) if np.isfinite(v1) and np.isfinite(v5) else np.nan
            row[f"{prefix}_qc_status_change"] = f"{b1.get('qc_status')}->{b5.get('qc_status')}"
        rows.append(row)
    return pd.DataFrame(rows).sort_values("subject")


def make_status_counts(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["band", "block", "qc_status"])
        .size()
        .reset_index(name="n")
        .sort_values(["band", "block", "qc_status"])
    )


def make_descriptives(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "aperiodic_slope",
        "centre_of_mass_hz",
        "peak_frequency_hz",
        "positive_residual_area",
    ]
    rows = []
    for (band, block), sub in df.groupby(["band", "block"]):
        for metric in metrics:
            values = pd.to_numeric(sub[metric], errors="coerce").dropna()
            rows.append(
                {
                    "band": band,
                    "block": block,
                    "metric": metric,
                    "n": int(values.size),
                    "mean": float(values.mean()) if not values.empty else np.nan,
                    "sd": float(values.std(ddof=1)) if values.size > 1 else np.nan,
                    "median": float(values.median()) if not values.empty else np.nan,
                    "min": float(values.min()) if not values.empty else np.nan,
                    "max": float(values.max()) if not values.empty else np.nan,
                }
            )
    return pd.DataFrame(rows)


def write_summary_report(
    metrics: pd.DataFrame,
    counts: pd.DataFrame,
    descriptives: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Write a concise Markdown QC summary."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# PAC Phase Spectral QC Summary",
        "",
        "This QC check used the cleaned PAC epoch stream, the broad frontal phase node, and the same 0.0-0.6 s analysis-window samples used by the PAC analyses. Welch spectra were fit with an aperiodic background, and theta/alpha support was summarised from positive residual spectral mass using centre of mass as the primary frequency estimate.",
        "",
        "## Status Counts",
        "",
        _table_to_markdown(counts) if not counts.empty else "No status counts available.",
        "",
        "## Descriptives",
        "",
        _table_to_markdown(descriptives.round(4)) if not descriptives.empty else "No descriptives available.",
        "",
        "## Suggested Reporting Paragraph",
        "",
        "PAC phase-band QC was performed on the same cleaned frontal epoch samples used for theta-gamma and alpha-gamma PAC estimation. For each participant and block, the broad frontal spectrum was parameterised into aperiodic and residual components, and theta/alpha centre-of-mass estimates were computed only when positive in-band residual spectral support was present. Participant-blocks without meaningful residual support were marked phase-indeterminate rather than assigned a forced frequency estimate.",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
