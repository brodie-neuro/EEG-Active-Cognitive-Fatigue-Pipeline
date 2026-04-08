# EEG Active Cognitive Fatigue Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MNE-Python](https://img.shields.io/badge/MNE--Python-1.6+-green.svg)](https://mne.tools/)

**Author:** Brodie E. Mangan  
**Affiliation:** University of Stirling  

## Overview

This repository contains a scripted EEG preprocessing and analysis pipeline for studying active cognitive fatigue. It is designed for 64-channel EEG, with a specific emphasis on:

- preserving task-relevant gamma-band signal rather than cleaning it away aggressively
- quantifying frontoparietal theta–gamma PAC, theta-band wPLI, and P3b
- using a simplified default preprocessing path based on average reference and deterministic notch filtering
- tracking preprocessing decisions with deterministic settings and step-level QC logs
- handling known subject-specific bad channels explicitly through per-participant config

The pipeline is written in Python around MNE-Python and related EEG tooling.

The active preprocessing path is a single shared `1-100 Hz` continuous stream. In practice, that means a shared `1 Hz` high-pass preprocessing path: ASR and ICA both run directly on the same incoming stream rather than creating internal fit-only copies.

## Why This Repo Exists

EEG preprocessing choices can materially alter downstream findings. Different referencing methods, artifact rejection thresholds, and decomposition-based cleaning steps introduce analytical variability that is rarely reported but can change effect sizes, signs, and statistical conclusions.

High-frequency EEG measures are especially vulnerable. Gamma-band activity overlaps spectrally with scalp muscle (EMG) contamination, and aggressive automated cleaning risks removing genuine neural signal alongside artifact. Most published EEG pipelines do not explicitly control for this.

This pipeline addresses these problems in three ways:

1. **Deterministic execution.** Fixed random seeds, single-threaded BLAS/LAPACK, locked dependency versions, and per-step QC logging ensure that results are reproducible on the same hardware and across architectures. The simplified preprocessing path (average reference + FIR notch filter) was chosen specifically because it avoids eigendecomposition-dependent steps that we found to be the primary source of cross-host numerical divergence.

2. **Explicit EMG control.** Rather than assuming gamma-band activity is neural, the pipeline includes dedicated EMG channels, PCA-based muscle covariates, and a regression-based exclusion criterion. Blocks where EMG explains more than 25% of gamma variance are flagged for exclusion.

3. **Documented parameter choices.** All major parameters are centralised in config files with written rationale ([PARAMETERS.md](PARAMETERS.md)), separating confirmatory analyses (PAC, wPLI, P3b, fixed-band frontal midline theta) from descriptive follow-up measures (frontal-midline theta peak summaries, spectral parameterisation).

The goal here is not just another EEG pipeline, but a more auditable, reproducible workflow where preprocessing decisions are visible and their consequences can be evaluated.

## Current Scope

The main CLI runner currently automates:

- preprocessing steps 01–07
- core postprocessing steps 08–13

Additional postprocessing scripts exist for QC aggregation, spectral parameterisation, and EMG/gamma follow-up analyses (steps 14–17), but they are not currently part of `python eeg_pipeline/run_pipeline.py --mode full`.

For PAC and gamma, `--mode full` is therefore not the final inclusion workflow by itself. Final high-frequency inclusion or exclusion decisions should only be made after running `16_emg_pca_covariates.py` and `17_emg_gamma_regression.py`.

### Confirmatory vs. Ancillary Analyses

| Type | Steps | Measures |
|:-----|:------|:---------|
| **Confirmatory** | 08–11 | P3b amplitude/latency, fixed-band frontal midline theta power, fixed 4–8 Hz theta–gamma PAC, theta wPLI |
| **Descriptive** | 12, 15 | Gamma power, frontal-midline theta peak summaries, spectral parameterisation |
| **Quality control** | 14, 16, 17 | Preprocessing QC summary, EMG covariates, EMG–gamma regression |

Method hierarchy:

- PAC phase band is fixed at `4-8 Hz`.
- Frontal midline theta power is fixed-band `4-8 Hz`.
- Specparam outputs and frontal-midline theta peak summaries are descriptive follow-ups, not required inputs for PAC or fixed-band theta power.

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
      09_band_power.py
      10_pac_nodal.py
      11_theta_wpli.py
      12_gamma_power.py
      13_merge_features.py
      14_preprocessing_qc_summary.py
      15_theta_stim_specparam.py
      16_emg_pca_covariates.py
      17_emg_gamma_regression.py
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

### Main runner steps

| Step | Script | Purpose |
|:-----|:-------|:--------|
| 01 | `01_import_qc.py` | import, channel typing, shared `1-100 Hz` bandpass filtering, initial QC |
| 02 | `02_simple_reference.py` | average re-reference (validated as reproducible across hardware) |
| 03 | `03_notch_filter.py` | FIR notch filter at 50 Hz |
| 04 | `04_asr.py` | Artifact Subspace Reconstruction on the shared `1 Hz` high-pass stream (cutoff = 30 SD) |
| 05 | `05_ica_iclabel.py` | Extended Infomax ICA plus ICLabel-based artifact classification on that same shared `1 Hz` high-pass stream |
| 06 | `06_epoch.py` | create stimulus-locked epoch sets (P3b and PAC windows) |
| 07 | `07_autoreject.py` | epoch-level repair and rejection |
| 08 | `08_erp_p3b.py` | P3b amplitude and latency extraction |
| 09 | `09_band_power.py` | frontal midline theta power (fixed 4-8 Hz) |
| 10 | `10_pac_nodal.py` | theta–gamma phase-amplitude coupling |
| 11 | `11_theta_wpli.py` | theta-band weighted phase lag index |
| 12 | `12_gamma_power.py` | gamma power features |
| 13 | `13_merge_features.py` | merge core feature tables |

### Ancillary scripts not run by `--mode full`

- `14_preprocessing_qc_summary.py`
- `15_theta_stim_specparam.py`
- `16_emg_pca_covariates.py`
- `17_emg_gamma_regression.py`

`15_theta_stim_specparam.py` is a descriptive frontal-midline theta peak / spectral-parameterisation follow-up and is not required for PAC computation or fixed-band theta-power extraction. By contrast, steps 16 and 17 remain part of the EMG-gating workflow for final PAC and gamma inclusion decisions even though they are not launched by `--mode full`.

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

Autoreject note: `cv` belongs to AutoReject only. The repo uses the library-default `cv=10`; it does not affect ICLabel or ICA classification.

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
- ASR and ICA are memory-hungry; 64 GB RAM is a practical minimum, with more headroom recommended for comfortable full-study runs

## Installation

```bash
git clone https://github.com/brodie-neuro/EEG-Active-Cognitive-Fatigue-Pipeline.git
cd EEG-Active-Cognitive-Fatigue-Pipeline
pip install -r requirements.txt
```

`requirements.txt` is the pinned working environment file for the repo.

Core libraries used by the active pipeline include `mne`, `asrpy`, `autoreject`, `mne-icalabel`, `specparam`, `pandas`, `numpy`, `scipy`, and `matplotlib`.

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
python eeg_pipeline/postprocessing/15_theta_stim_specparam.py
python eeg_pipeline/postprocessing/16_emg_pca_covariates.py
python eeg_pipeline/postprocessing/17_emg_gamma_regression.py
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
