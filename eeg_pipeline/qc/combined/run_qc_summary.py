"""Build a compact combined QC index from PAC phase and P3b QC outputs."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

PIPELINE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPELINE_DIR))


OUT_DIR = PIPELINE_DIR / "outputs" / "qc" / "combined"
PAC_METRICS = PIPELINE_DIR / "outputs" / "qc" / "pac_phase" / "tables" / "pac_phase_qc_band_metrics.csv"
P3B_METRICS = PIPELINE_DIR / "outputs" / "qc" / "p3b" / "tables" / "p3b_qc_metrics.csv"


def _load(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _status_summary(statuses: pd.Series) -> str:
    vals = [str(v) for v in statuses.dropna().tolist()]
    if not vals:
        return "missing"
    if any(v in {"indeterminate", "review"} for v in vals):
        return "review"
    if any(v == "weak" for v in vals):
        return "weak"
    return "clear"


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Combined QC summary")
    parser.add_argument("--subject", default="", help="Optional subject filter retained for runner compatibility.")
    args, _ = parser.parse_known_args()
    selected = {s.strip().lower() for s in args.subject.replace(",", " ").split() if s.strip()}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pac = _load(PAC_METRICS)
    p3b = _load(P3B_METRICS)

    subjects = set()
    if not pac.empty:
        subjects.update(pac["subject"].astype(str))
    if not p3b.empty:
        subjects.update(p3b["subject"].astype(str))
    if selected:
        subjects = {s for s in subjects if s.lower() in selected}

    rows = []
    for subject in sorted(subjects):
        row = {"subject": subject}
        if not pac.empty:
            sub = pac[pac["subject"].astype(str) == subject]
            for band in sorted(sub["band"].dropna().unique()):
                band_sub = sub[sub["band"] == band]
                row[f"pac_{band}_overall_qc"] = _status_summary(band_sub["qc_status"])
                for block in sorted(band_sub["block"].dropna().unique()):
                    hit = band_sub[band_sub["block"] == block].iloc[0]
                    row[f"pac_{band}_B{int(block)}_qc"] = hit["qc_status"]
                    row[f"pac_{band}_B{int(block)}_com_hz"] = hit.get("centre_of_mass_hz")
        if not p3b.empty:
            sub = p3b[p3b["subject"].astype(str) == subject]
            row["p3b_overall_qc"] = _status_summary(sub["qc_status"]) if not sub.empty else "missing"
            for block in sorted(sub["block"].dropna().unique()):
                hit = sub[sub["block"] == block].iloc[0]
                row[f"p3b_B{int(block)}_qc"] = hit["qc_status"]
                row[f"p3b_B{int(block)}_mean_uV"] = hit.get("p3b_mean_uV")
        rows.append(row)

    index = pd.DataFrame(rows)
    status_cols = [col for col in index.columns if col.endswith("_qc") or col.endswith("_overall_qc")]
    if status_cols:
        index[status_cols] = index[status_cols].fillna("missing")
    index_path = OUT_DIR / "qc_index.csv"
    index.to_csv(index_path, index=False)

    counts = []
    for col in index.columns:
        if col.endswith("_qc") or col.endswith("_overall_qc"):
            for status, n in index[col].value_counts(dropna=False).items():
                counts.append({"field": col, "status": status, "n": int(n)})
    counts_df = pd.DataFrame(counts)
    counts_path = OUT_DIR / "qc_index_status_counts.csv"
    counts_df.to_csv(counts_path, index=False)

    report = OUT_DIR / "qc_index.md"
    lines = [
        "# Combined QC Index",
        "",
        f"PAC phase metrics: `{PAC_METRICS}`",
        f"P3b metrics: `{P3B_METRICS}`",
        "",
        "## Participant Index",
        "",
        _table_to_markdown(index.round(4) if not index.empty else index),
        "",
        "## Status Counts",
        "",
        _table_to_markdown(counts_df),
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved combined QC index: {index_path}")
    print(f"Saved combined QC status counts: {counts_path}")
    print(f"Saved combined QC report: {report}")


if __name__ == "__main__":
    main()
