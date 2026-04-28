# EEG Preprocessing Pipeline

This document summarizes the active public preprocessing path. The live
settings are defined in `eeg_pipeline/config/parameters.json` and the step
scripts under `eeg_pipeline/preprocessing/`.

## Step 01: Import and Filtering

Raw participant blocks are imported, channel types are assigned, and EMG
channels are excluded from EEG-specific processing. Two filtered streams are
created:

- Main oscillatory stream: `1-100 Hz`, used for ASR, ICA, theta-gamma PAC, and alpha-gamma PAC.
- ERP-only stream: `0.1-100 Hz`, used only for conservative P3b estimation.

The ERP branch does not feed ASR, ICA, PAC, or alpha-gamma PAC outputs.

## Step 02: Simple Average Reference

Bad EEG channels flagged during import/QC, plus any subject-specific known bad
channels from participant config, are interpolated. A simple average reference
is then applied to both the main oscillatory stream and the ERP-only stream.

This replaces the older robust-reference path in order to keep the
pipeline deterministic and easier to audit.

## Step 03: FIR Notch Filtering

Line noise is handled with deterministic FIR notch filters at `50 Hz` and
`100 Hz` on both streams. The current public path does not use Zapline/DSS.

## Step 04: Artifact Subspace Reconstruction

ASR is applied only to the main `1 Hz` high-pass oscillatory stream. The current
cutoff is `30`, with clean-window calibration enabled. ASR modification rates
are logged per block and blocks above the configured QC threshold are flagged
for review.

## Step 05: ICA and ICLabel

Extended Infomax ICA is fitted directly on the ASR-cleaned main oscillatory
stream with `n_components = 25` and `random_state = 42`. ICLabel is then used to
remove high-confidence ocular, cardiac, and muscle components according to the
configured thresholds.

## Step 06: Epoching

Stimulus-locked epochs are created for:

| Epoch set | Source stream | Window | Baseline | Purpose |
|:----------|:--------------|:-------|:---------|:--------|
| `p3b_erp` | ERP-only `0.1-100 Hz` branch | `-0.2 to 0.8 s` | `-0.2 to 0.0 s` | H3 P3b |
| `pac` | Main `1-100 Hz` branch | `-0.5 to 1.8 s` | none | H1/H2 PAC |

For blocks with prepended practice trials, the configured task-relevant onsets
are retained and practice onsets are discarded automatically.

## Step 07: Autoreject

Autoreject performs epoch-level repair and rejection with the configured
cross-validation grid. It runs on the relevant epoch sets, including the
dedicated `p3b_erp` branch and the main `pac` epochs.

## Active Analysis Mapping

| Hypothesis | Step | Measure |
|:-----------|:-----|:--------|
| H1 | `postprocessing/08_pac_nodal.py` | theta-gamma PAC |
| H2 | `postprocessing/09_alpha_gamma_pac.py` | alpha-gamma PAC |
| H3 | `postprocessing/10_erp_p3b.py` | posterior P3b cluster |

`run_pipeline.py --mode full` runs the core path through step 11. EMG
sensitivity is a follow-up workflow: run `13_emg_pca_covariates.py`, then
`08b_pac_emg_corrected.py`, then `14_emg_pac_correlation.py`.
