# EEG Active Cognitive Fatigue Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![MNE-Python](https://img.shields.io/badge/MNE--Python-1.6+-green.svg)](https://mne.tools/)

**Author:** Brodie E. Mangan  
**Status:** Active Development

---

## Overview

A modular EEG preprocessing and analysis pipeline for high-density (64-channel) EEG data. Built with `mne-python` and optimised for gamma-band preservation.

---

## Pipeline Structure

### Preprocessing (`eeg_pipeline/steps/`)

| Step | Script | Function |
|:-----|:-------|:---------|
| 00 | `00_make_fake_data.py` | Generate synthetic EEG for testing |
| 00 | `00_view_raw.py` | Visual inspection of raw files |
| 01 | `01_import_qc.py` | Import, filter, bad channel detection |
| 02 | `02_clean_reference.py` | Robust average reference (PyPREP) |
| 04 | `04_zapline.py` | Line noise removal (Zapline) |
| 05 | `05_ica_iclabel.py` | ICA + ICLabel automated classification |
| 06 | `06_asr.py` | Artifact Subspace Reconstruction |
| 07 | `07_epoch.py` | Epoching |
| 08 | `08_autoreject.py` | Automated epoch rejection |
| 09 | `09_features.py` | Feature extraction |

### Analysis (`eeg_pipeline/analysis/`)

| Step | Script | Function |
|:-----|:-------|:---------|
| 10 | `10_erp_p3b.py` | ERP analysis |
| 11 | `11_band_power.py` | Band power extraction |
| 12 | `12_peak_frequencies.py` | Peak frequency analysis |
| 13 | `13_pac_nodal.py` | Phase-amplitude coupling |
| 14 | `14_connectivity.py` | Connectivity analysis |
| 15 | `15_network_louvain.py` | Network analysis |
| 16 | `16_regression_model.py` | Statistical modelling |

### Baseline Comparison (`baseline_pipeline/`)

Traditional preprocessing pipeline for methods comparison.

---

## Installation

```bash
git clone https://github.com/brodie-neuro/EEG-Active-Cognitive-Fatigue-Pipeline.git
pip install -r requirements.txt
```

---

## Documentation

See `docs/` for detailed methodology.

---

## License

All Rights Reserved © Brodie E. Mangan