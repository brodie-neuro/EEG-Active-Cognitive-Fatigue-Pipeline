"""Summaries for P3b QC outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _table_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows available."
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in cols) + " |")
    return "\n".join(lines)


def make_delta_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subject, sub in df.groupby("subject"):
        b1 = sub[sub["block"] == 1]
        b5 = sub[sub["block"] == 5]
        if b1.empty or b5.empty:
            continue
        b1 = b1.iloc[0]
        b5 = b5.iloc[0]
        row = {"subject": subject}
        for metric in ["p3b_mean_uV", "p3b_fractional_area_latency_ms", "p3b_peak_uV"]:
            v1 = b1.get(metric, np.nan)
            v5 = b5.get(metric, np.nan)
            row[f"{metric}_delta"] = float(v5 - v1) if np.isfinite(v1) and np.isfinite(v5) else np.nan
        row["p3b_qc_status_change"] = f"{b1.get('qc_status')}->{b5.get('qc_status')}"
        rows.append(row)
    return pd.DataFrame(rows).sort_values("subject")


def make_status_counts(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["block", "qc_status"]).size().reset_index(name="n").sort_values(["block", "qc_status"])


def make_descriptives(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["p3b_mean_uV", "p3b_fractional_area_latency_ms", "p3b_peak_uV", "n_epochs"]
    rows = []
    for block, sub in df.groupby("block"):
        for metric in metrics:
            values = pd.to_numeric(sub[metric], errors="coerce").dropna()
            rows.append(
                {
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


def write_summary_report(df: pd.DataFrame, counts: pd.DataFrame, descriptives: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# P3b QC Summary",
        "",
        "P3b QC used the cleaned ERP-branch `p3b_erp` epochs and the configured P3b ROI/window. Participant dashboards show ROI waveforms, single-trial ROI images, planned ROI channel traces, and P3b-window topographies for Block 1, Block 5, and delta.",
        "",
        "## Status Counts",
        "",
        _table_to_markdown(counts) if not counts.empty else "No status counts available.",
        "",
        "## Descriptives",
        "",
        _table_to_markdown(descriptives.round(4)) if not descriptives.empty else "No descriptives available.",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
