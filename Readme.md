# EEG Active Cognitive Fatigue Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MNE-Python](https://img.shields.io/badge/MNE--Python-1.6+-green.svg)](https://mne.tools/)

**Author:** Brodie E. Mangan  
**Affiliation:** University of Stirling  

## Overview

This repository contains a scripted EEG preprocessing and analysis pipeline for studying active cognitive fatigue. It is designed for 64-channel EEG, with a specific emphasis on:

- preserving task-relevant gamma-band signal rather than cleaning it away aggressively
- quantifying frontoparietal theta–gamma PAC and P3b as confirmatory measures, with alpha–gamma PAC as an exploratory descriptive measure
- using a simplified default preprocessing path based on average reference and deterministic notch filtering
- tracking preprocessing decisions with deterministic settings and step-level QC logs
- handling known subject-specific bad channels explicitly through per-participant config

The pipeline is written in Python around MNE-Python and related EEG tooling.

The active preprocessing path is a single shared `1-100 Hz` continuous stream. In practice, that means a shared `1 Hz` high-pass oscillatory path: ASR and ICA both run directly on the same incoming stream rather than creating internal fit-only copies.

A separate `0.1-100 Hz` ERP branch also exists, but only for conservative P3b estimation. That branch does not feed ASR, ICA, PAC, alpha-gamma PAC, or the main oscillatory outputs.

## Why This Repo Exists

EEG preprocessing choices can materially alter downstream findings. Different referencing methods, artifact rejection thresholds, and decomposition-based cleaning steps introduce analytical variability that is rarely reported but can change effect sizes, signs, and statistical conclusions.

High-frequency EEG measures are especially vulnerable. Gamma-band activity overlaps spectrally with scalp muscle (EMG) contamination, and aggressive automated cleaning risks removing genuine neural signal alongside artifact. Most published EEG pipelines do not explicitly control for this.

This pipeline addresses these problems in three ways:

1. **Deterministic execution.** Fixed random seeds, single-threaded BLAS/LAPACK, locked dependency versions, and per-step QC logging ensure that results are reproducible on the same hardware and across architectures. The simplified preprocessing path (average reference + FIR notch filter) was chosen specifically because it avoids eigendecomposition-dependent steps that we found to be the primary source of cross-host numerical divergence.

2. **Explicit EMG control.** Rather than assuming gamma-band activity is neural, the pipeline includes dedicated EMG channels, PCA-based muscle covariates, and a regression-based exclusion criterion. Blocks where EMG explains more than 25% of gamma variance are flagged for exclusion.

3. **Documented parameter choices.** All major parameters are centralised in config files with written rationale ([PARAMETERS.md](PARAMETERS.md)), separating confirmatory analyses (theta-gamma PAC and P3b) from exploratory/descriptive alpha-gamma PAC and EMG sensitivity checks.

The goal here is not just another EEG pipeline, but a more auditable, reproducible workflow where preprocessing decisions are visible and their consequences can be evaluated.

## Current Scope

The main CLI runner currently automates:

- preprocessing steps 01–07
- core postprocessing steps 08, 10, and 13

Additional postprocessing scripts exist for QC aggregation, EMG/PAC sensitivity, and exploratory alpha-gamma PAC, but they are not currently part of `python eeg_pipeline/run_pipeline.py --mode full`.

For PAC, `--mode full` is therefore not the final inclusion workflow by itself. Final high-frequency inclusion or exclusion decisions should only be made after running `16_emg_pca_covariates.py` and `17_emg_pac_correlation.py`.

### Confirmatory vs. Ancillary Analyses

| Type | Steps | Measures |
|:-----|:------|:---------|
| **Confirmatory** | 08, 10 | P3b mean amplitude and fractional area latency (Luck, 2014), fixed 4–8 Hz theta–gamma PAC |
| **Exploratory / Descriptive** | 18 | Between-region alpha–gamma PAC (C_broad_F → C_broad_P) |
| **Quality control** | 14 | Preprocessing QC summary |
| **EMG sensitivity** | 10b, 16, 17 | EMG covariates, EMG-corrected PAC, group-level EMG–PAC delta correlation |

P3b in step 08 is estimated from the dedicated `p3b_erp` branch (`0.1 Hz` high-pass). P3b latency uses 50% fractional area latency rather than argmax peak detection. The `0.1 Hz` high-pass preserves ERP amplitude fidelity but also admits slow positive drift, which can pull argmax toward the window edge rather than the true P3b peak. Fractional area latency (the timepoint at which 50% of the cumulative positive area in the P3b window has been reached) is robust to this drift and is recommended over argmax for ERP latency estimation (Luck, 2014). Theta-gamma PAC, alpha-gamma PAC, and the shared oscillatory outputs stay on the main `1 Hz` path.

Method hierarchy:

- The shared `1 Hz` high-pass path is the live oscillatory pipeline for ASR, ICA, theta-gamma PAC, and alpha-gamma PAC.
- A separate `0.1 Hz` ERP branch exists only for conservative P3b estimation.
- Confirmatory PAC phase band is fixed at `4-8 Hz` (theta-gamma, H1).
- Exploratory alpha-gamma PAC phase band is `8-13 Hz` (step 18).
- Alpha-gamma PAC is a descriptive follow-up and is not a required input for confirmatory analyses.

## Design Principles

- Mostly config-driven: algorithm settings live in `eeg_pipeline/config/parameters.json`, while `eeg_pipeline/config/study.yml` holds study structure, paths, montage, and node definitions.
- Deterministic by design: fixed seeds, single-threaded numerical execution, and per-step provenance logging are used throughout the main pipeline.
- EMG-aware: gamma analyses are built around explicit muscle-contamination control rather than assuming high-frequency activity is automatically neural.
- Explicit method surface: the repo-level parameter rationale is documented in [PARAMETERS.md](PARAMETERS.md).

### Cross-Host Reproducibility

During pipeline validation, we tested reproducibility across multiple GCP VM instances (AMD EPYC and Intel architectures). The preprocessing path (average reference + FIR notch filter) avoids eigendecomposition-dependent steps, which we found to be the primary source of cross-host numerical divergence in EEG pipelines.

**Internal validation results:**
- Same host, repeated runs → byte-identical results
- Cross-architecture (AMD vs Intel) → consistent effect directions and near-identical outputs for matched inputs

## Repository Layout

```text
EEG_study_2/
  eeg_pipeline/
    run_pipeline.py
    config/
      parameters.json
      study.yml
      file_naming.json
    preprocessing/
      00_*.py
      01_import_qc.py
      02_simple_reference.py
      03_notch_filter.py
      04_asr.py
      05_ica_iclabel.py
      06_epoch.py
      07_autoreject.py
    postprocessing/
      08_erp_p3b.py
      10_pac_nodal.py
      10b_pac_emg_corrected.py
      13_merge_features.py
      14_preprocessing_qc_summary.py
      16_emg_pca_covariates.py
      17_emg_pac_correlation.py
      18_alpha_gamma_pac.py
    src/
  PARAMETERS.md
  requirements.txt
  LICENSE
```

## Pipeline Steps

### Utility / setup scripts

- `00_split_blocks.py`: split participant recordings into per-block FIF files
- `00_import_participant.py`: import helper for participant-level ingestion
- `00_make_fake_data.py`: synthetic test data generation

### Pipeline scripts

| Step | Script | Purpose |
|:-----|:-------|:--------|
| 01 | `01_import_qc.py` | import, channel typing, shared `1-100 Hz` oscillatory bandpass plus separate `0.1-100 Hz` ERP-branch filtering, initial QC |
| 02 | `02_simple_reference.py` | average re-reference for both the shared oscillatory path and the dedicated ERP branch |
| 03 | `03_notch_filter.py` | FIR notch filter at 50 Hz for both branches |
| 04 | `04_asr.py` | Artifact Subspace Reconstruction on the shared `1 Hz` high-pass stream (cutoff = 30 SD) |
| 05 | `05_ica_iclabel.py` | Extended Infomax ICA plus ICLabel-based artifact classification on that same shared `1 Hz` high-pass stream |
| 06 | `06_epoch.py` | create main-path `p3b` and `pac` epochs plus separate ERP-branch `p3b_erp` epochs |
| 07 | `07_autoreject.py` | epoch-level repair and rejection for `p3b`, `p3b_erp`, and `pac` epoch sets |
| 08 | `08_erp_p3b.py` | P3b mean amplitude and fractional area latency (50%) from the dedicated `p3b_erp` branch |
| 10 | `10_pac_nodal.py` | theta–gamma phase-amplitude coupling |
| 10b | `10b_pac_emg_corrected.py` | PAC recomputed after regressing EMG PC1 out of parietal gamma signal at each time point (sensitivity check) |
| 13 | `13_merge_features.py` | merge core feature tables |
| 16 | `16_emg_pca_covariates.py` | PCA-based EMG covariates from temporalis bipolar (F7−FT7, F8−FT8) and posterior neck monopolar (TP7, TP8) channels |
| 17 | `17_emg_pac_correlation.py` | group-level delta-EMG vs delta-PAC correlation to test whether PAC fatigue effect is driven by EMG changes |
| 18 | `18_alpha_gamma_pac.py` | between-region alpha–gamma phase-amplitude coupling (exploratory/descriptive; C_broad_F → C_broad_P) |

### Ancillary scripts not run by `--mode full`

- `14_preprocessing_qc_summary.py`
- `16_emg_pca_covariates.py`
- `10b_pac_emg_corrected.py`
- `17_emg_pac_correlation.py`
- `18_alpha_gamma_pac.py`

Steps 16 and 17 remain part of the EMG sensitivity workflow even though they are not launched by `--mode full`.

### EMG sensitivity analysis (steps 10b, 17)

Because gamma-band activity overlaps spectrally with scalp muscle (EMG) contamination, the pipeline includes a two-stage EMG sensitivity workflow to verify that PAC findings are not driven by myogenic artifact:

1. **Step 10b — EMG-corrected PAC.** Regresses EMG PC1 out of the parietal signal at every time point across trials, then recomputes PAC from the corrected signal using the same method as step 10 (trial-concatenated MI, 500 circular-shift surrogates, seed = 42). If the corrected PAC values are highly correlated with uncorrected values, the theta–gamma coupling effect survives EMG removal.

2. **Step 17 — Group-level delta correlation.** Correlates participant-level change in EMG variance (B5 − B1) with change in PAC z-score (B5 − B1). If the correlation is weak, the PAC fatigue effect is not driven by participants changing their muscle tension between blocks.

The primary PAC result (step 10) is reported as the main finding. Steps 10b and 17 are reported as a sensitivity analysis confirming that the PAC effect is not attributable to myogenic contamination.

The time-point regression in step 10b is intentionally conservative: it removes any trial-wise parietal variance linearly associated with EMG, not only proven muscle artifact. This makes it a robust upper-bound test — if PAC survives this aggressive correction, the theta–gamma coupling effect cannot be attributed to peripheral EMG.

## Parameters and Method Rationale

The short repo-facing explanation is in [PARAMETERS.md](PARAMETERS.md).

That file is the right place to answer:

- what the major parameters are
- where they are set
- why they were chosen
- which settings are confirmatory versus descriptive
- where the manuscript should stay aligned with the code

Current parameter surface:

- `eeg_pipeline/config/parameters.json`: main preprocessing and many analysis parameters
- `eeg_pipeline/config/study.yml`: study-level paths, block structure, montage, and node definitions
- `eeg_pipeline/config/participant_configs/*.json` (user-created, not shipped): optional per-subject overrides such as known bad channels. Each study creates its own.
- selected script-level constants in analysis scripts: still part of the live method until consolidated

## Determinism and Runtime Notes

The main runner enforces a deterministic execution profile where possible:

- `random_state=42` for ICA, Autoreject, and PAC surrogates
- single-threaded numerical execution via `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and `OMP_NUM_THREADS=1`
- `PYTHONHASHSEED=0`
- MNE home isolation for pipeline runs
- per-step QC JSON logs and run-level provenance manifests

Practical notes:

- the default public path is `01_import_qc -> 02_simple_reference -> 03_notch_filter -> 04_asr -> 05_ica_iclabel -> 06_epoch -> 07_autoreject`
- step-level QC logging is part of the primary audit trail
- ASR and ICA are memory-hungry; 64 GB RAM is a practical minimum.

## Installation

```bash
git clone https://github.com/brodie-neuro/EEG-Active-Cognitive-Fatigue-Pipeline.git
cd EEG-Active-Cognitive-Fatigue-Pipeline
pip install -r requirements.txt
```

`requirements.txt` is the pinned working environment file for the repo.

Core libraries used by the active pipeline include `mne`, `asrpy`, `autoreject`, `mne-icalabel`, `pandas`, `numpy`, `scipy`, and `matplotlib`.

## Quick Start

### 1. Prepare block-level FIF files

```bash
python eeg_pipeline/preprocessing/00_split_blocks.py \
  --input eeg_pipeline/raw/participant.cnt \
  --out-dir eeg_pipeline/raw \
  --subject sub-XXX \
  --block-labels 1 5
```

### 2. Run the main pipeline

```bash
python eeg_pipeline/run_pipeline.py --mode full --subject sub-XXX
```

### 3. Run stages separately if needed

```bash
python eeg_pipeline/run_pipeline.py --mode preprocess --subject sub-XXX
python eeg_pipeline/run_pipeline.py --mode analysis --subject sub-XXX
```

### 4. Run ancillary analyses manually

Examples:

```bash
python eeg_pipeline/postprocessing/16_emg_pca_covariates.py
python eeg_pipeline/postprocessing/17_emg_pac_correlation.py
python eeg_pipeline/postprocessing/10b_pac_emg_corrected.py
```

## Outputs

The pipeline writes outputs under `eeg_pipeline/outputs/`, including:

- cleaned derivatives
- epoch sets
- feature CSVs
- QC figures
- step-level QC JSON logs
- run-level provenance manifests

Core feature tables are written under `eeg_pipeline/outputs/features/`.

## Documentation

Public-facing method documentation in this repo:

- [PARAMETERS.md](PARAMETERS.md): parameter choices and method rationale

Additional working notes may exist in local or private `docs/` material during active development, but `PARAMETERS.md` is intended to be the stable repo-level reference.

## License

MIT License. See [LICENSE](LICENSE).
