# Integrated QC Workflows

This folder contains automated QC-only workflows that run alongside the main
preprocessing and the primary confirmatory and secondary analyses.

The QC outputs are descriptive. They do not alter the primary confirmatory
theta-gamma PAC (H1), secondary alpha-gamma PAC (H2), or secondary P3b (H3)
feature tables.

## Run

Run all integrated QC checks:

```powershell
python eeg_pipeline/run_pipeline.py --mode qc
```

Run QC for one participant:

```powershell
python eeg_pipeline/run_pipeline.py --mode qc --subject sub-p003
```

The full pipeline also runs QC after the main analysis steps:

```powershell
python eeg_pipeline/run_pipeline.py --mode full
```

## PAC Phase QC

Script:

```text
eeg_pipeline/qc/pac_phase/run_pac_phase_qc.py
```

Purpose:

- verify that the low-frequency phase bands used by PAC have plausible
  oscillatory support above the aperiodic background
- use the same cleaned `pac` epochs and the same `0.0-0.6 s` PAC analysis
  window
- use the broad frontal phase node (`C_broad_F`)
- compute Welch spectra with 2 s Hann windows, 50% overlap, and 4x
  zero-padding for a smoother frequency grid; the true resolution is still
  set by the 2 s window
- remove the aperiodic component with specparam over 2-20 Hz; no fallback
  aperiodic model is used
- compute residual spectral centre of mass only when positive residual support
  is present

Default bands:

- theta: 4-8 Hz
- alpha: 8-13 Hz

Outputs:

```text
eeg_pipeline/outputs/qc/pac_phase/tables/
eeg_pipeline/outputs/qc/pac_phase/figures/
eeg_pipeline/outputs/qc/pac_phase/reports/
eeg_pipeline/outputs/qc/pac_phase/logs/
```

Key tables:

- `pac_phase_qc_band_metrics.csv`
- `pac_phase_qc_delta_metrics.csv`
- `pac_phase_qc_status_counts.csv`
- `pac_phase_qc_descriptives.csv`

Participant figures are numbered:

```text
01_sub-p003_pac_phase_qc.png
```

## P3b QC

Script:

```text
eeg_pipeline/qc/p3b/run_p3b_qc.py
```

Purpose:

- use the cleaned ERP branch (`p3b_erp`)
- visualise the configured P3b ROI and measurement window
- produce participant dashboards with ROI waveform, single-trial images,
  ROI channel traces, and topography

Outputs:

```text
eeg_pipeline/outputs/qc/p3b/tables/
eeg_pipeline/outputs/qc/p3b/figures/
eeg_pipeline/outputs/qc/p3b/reports/
eeg_pipeline/outputs/qc/p3b/logs/
```

Key tables:

- `p3b_qc_metrics.csv`
- `p3b_qc_delta_metrics.csv`
- `p3b_qc_status_counts.csv`
- `p3b_qc_descriptives.csv`

Participant figures are numbered:

```text
01_sub-p003_p3b_qc.png
```

## Combined Index

Script:

```text
eeg_pipeline/qc/combined/run_qc_summary.py
```

Outputs:

```text
eeg_pipeline/outputs/qc/combined/qc_index.csv
eeg_pipeline/outputs/qc/combined/qc_index.md
```
