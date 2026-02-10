# EEG Active Cognitive Fatigue Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![MNE-Python](https://img.shields.io/badge/MNE--Python-1.6+-green.svg)](https://mne.tools/)

**Author:** Brodie E. Mangan  
**Affiliation:** University of Stirling  
**Status:** Active Development

---

## Overview

A modular EEG preprocessing and analysis pipeline for high-density (64-channel) EEG data recorded during the WAND cognitive fatigue protocol. Built with `mne-python` and optimised for gamma-band preservation and theta-gamma cross-frequency coupling analysis.

The pipeline implements the analysis plan for the Missing Link study (Mangan & Kourtis), testing whether cognitive fatigue disrupts the theta-gamma neural code supporting working memory.

---

## Pipeline Structure

### Pre-Processing (`eeg_pipeline/steps/`)

| Step | Script | Function |
|:-----|:-------|:---------|
| 00 | `00_make_fake_data.py` | Generate synthetic EEG for testing (Block 1 + Block 5) |
| 00 | `00_view_raw.py` | Visual inspection of raw files |
| 01 | `01_import_qc.py` | Import, filter, bad channel detection |
| 02 | `02_clean_reference.py` | Robust average reference (PyPREP) |
| 04 | `04_zapline.py` | Line noise removal (Zapline spatial filtering) |
| 05 | `05_ica_iclabel.py` | ICA + ICLabel automated classification |
| 06 | `06_asr.py` | Artifact Subspace Reconstruction |
| 07 | `07_epoch.py` | Epoching (P3b stimulus-locked + PAC maintenance-locked) |
| 08 | `08_autoreject.py` | Automated epoch rejection/repair |
| 09 | `09_features.py` | Basic feature extraction |

### Post-Processing / Analysis (`eeg_pipeline/analysis/`)

All analyses compare **Block 1** (baseline) with **Block 5** (fatigued). Change scores (Δ = Block 5 − Block 1) serve as the primary variables.

| Step | Script | Hypothesis | Function |
|:-----|:-------|:-----------|:---------|
| 1 | `10_erp_p3b.py` | **H5** | P3b ERP analysis (confirmatory anchor) |
| 2–3 | `12_peak_frequencies.py` | **H6, H2** | IAF (specparam) + theta peak frequency |
| 4 | `13_pac_nodal.py` | **H1, H3** | Between-region PAC (RF→RP) + local PAC |
| 5 | `11_band_power.py` | — | Frontal midline theta power (feeds H4) |
| 6–7 | `16_regression_model.py` | **H4** | Merge features → CSV for R modelling (lme4/lmerTest) |

### Baseline Comparison (`baseline_pipeline/`)

Traditional preprocessing pipeline for methods comparison.

---

## Hypotheses

| ID | Hypothesis | Test |
|:---|:-----------|:-----|
| **H1** | Fatigue disrupts frontoparietal CFC (RF→RP) | Between-region PAC decline |
| **H2** | Theta rhythm becomes mistuned under fatigue | Theta peak frequency slowing (specparam) |
| **H3** | Local PAC degrades more frontally than parietally | Region × Block interaction |
| **H4** | Combined markers disambiguate effort from breakdown | Integrated regression model |
| **H5** | P3b amplitude declines with fatigue | Confirmatory t-test |
| **H6** | IAF slows with fatigue | Confirmatory t-test |

---

## Synthetic Data

The fake data generator (`00_make_fake_data.py`) creates realistic 64-channel EEG with:

- **1/f pink noise** background with realistic amplitudes
- **Oscillatory components**: posterior alpha (~10 Hz), frontal theta (~5.5 Hz)
- **Task events**: stimulus onsets (target/nontarget), responses (correct/error/miss)
- **ERPs**: N1, P2, P3b at posterior sites (targets only)
- **PAC**: theta-gamma coupling injected during maintenance window
- **Artifacts**: blinks (EOG leak), EMG bursts, line noise (50/100 Hz)
- **Fatigue simulation**: Block 5 has reduced P3b (60%), slower theta (−0.8 Hz), halved PAC

---

## Key Dependencies

| Package | Purpose |
|:--------|:--------|
| `mne` | EEG processing core |
| `specparam` | Spectral parameterisation (periodic vs aperiodic) |
| `tensorpac` | Phase-amplitude coupling (MI + surrogates) |
| `pyprep` | Robust referencing (RANSAC) |
| `meegkit` | Zapline line noise removal, ASR |
| `autoreject` | Epoch-level artifact rejection |
| `mne-icalabel` | Automated ICA component classification |
| `scikit-learn` | Regression modelling |

---

## Installation

```bash
git clone https://github.com/brodie-neuro/EEG-Active-Cognitive-Fatigue-Pipeline.git
pip install -r requirements.txt
```

---

## Configuration

All pipeline parameters are set in `eeg_pipeline/config/study.yml`:

- **Data paths and format** (BrainVision `.vhdr`)
- **Filter settings** (0.1–120 Hz, 50 Hz notch)
- **Block comparison** (Block 1 vs Block 5)
- **Specparam settings** for spectral decomposition
- **PAC parameters** (theta 4–8 Hz, high gamma 55–85 Hz, 200 surrogates)
- **9-node structure** and 3-region aggregation

---

## Documentation

See `docs/` for detailed methodology:

- `preprocessing_pipeline.md` — Pre-processing rationale and steps
- `justification_hypotheses.md` — Hypothesis justification and model comparison strategy
- `methodology_pac_zscore.md` — Why surrogate z-scores for PAC
- `docs/figures/` — MATLAB-generated artifact visualisation figures

---

## License

All Rights Reserved © Brodie E. Mangan