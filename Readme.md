# EEG Active Cognitive Fatigue Pipeline

**Author:** Brodie E. Mangan  
**Status:** In Development  
**License:** Private / All Rights Reserved

## Project Overview
This repository contains the preprocessing and analysis pipeline for a high-density (64-channel) EEG study investigating ** Active Cognitive Fatigue** using the WAND paradigm.

The pipeline is built in Python using `mne-python` and integrates state-of-the-art artefact rejection techniques to isolate neural markers of fatigue, specifically focusing on:
* **ERPs:** P3b amplitude and latency changes.
* **Oscillations:** Individual Alpha Frequency (IAF) and Theta power.
* **Connectivity:** Phase-Amplitude Coupling (PAC) in the Theta-Gamma bands.

## Pipeline Architecture
The analysis is modularised into sequential steps found in the `steps/` directory:

| Step | Script | Function |
| :--- | :--- | :--- |
| **00** | `00_make_fake_data.py` | Generates synthetic EEG data with specific artefacts (blinks, muscle noise) for pipeline testing. |
| **00** | `00_view_raw.py` | Visual inspection tool for raw `.vhdr` or `.fif` files. |
| **01** | `01_import_qc.py` | Imports raw data, applies basic filters, and runs initial QC (Bad Channel Detection). |
| **02** | `02_clean_reference.py` | Applies **PyPREP** (Robust Referencing) and interpolates bad channels. |

## Installation & Requirements
The environment is managed via `pip`. To replicate the analysis environment:

```bash
# Clone the repository
git clone [https://github.com/brodie-neuro/EEG-Active-Cognitive-Fatigue-Pipeline.git](https://github.com/brodie-neuro/EEG-Active-Cognitive-Fatigue-Pipeline.git)

# Install dependencies
pip install -r requirements.txt