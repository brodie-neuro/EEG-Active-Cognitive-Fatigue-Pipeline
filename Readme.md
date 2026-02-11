# EEG Active Cognitive Fatigue Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MNE-Python](https://img.shields.io/badge/MNE--Python-1.6+-green.svg)](https://mne.tools/)

**Author:** Brodie E. Mangan
**Affiliation:** University of Stirling
**Status:** Active Development

---

## Overview

A modular EEG preprocessing and analysis pipeline for high-density (64-channel) EEG data recorded during the WAND cognitive fatigue protocol. Built with `mne-python` and optimised for gamma-band preservation and theta-gamma cross-frequency coupling analysis.

The pipeline implements the analysis plan for the Missing Link study (Mangan & Kourtis), testing whether cognitive fatigue disrupts the theta-gamma neural code supporting working memory.

### Key Design Principles

- **Config-driven**: All algorithm parameters live in a single JSON file -- no hardcoded thresholds
- **Quality-controlled**: Every preprocessing step generates QC figures and assessment reports
- **Individualised bands**: Subject-specific peak frequencies via specparam, not canonical bands
- **Reproducible**: Parameter file hash tracked in assessment reports for full traceability

---

## Repository Structure

```
EEG_study_2/
  eeg_pipeline/
    parameter_gui.py          # Themed parameter editor (tooltips, presets, hash)
    pipeline_runner_gui.py    # Themed pipeline runner (raw path + preset + live log)
    config/
      study.yml              # Data paths, montage, block definitions
      parameters.json        # All algorithm parameters (single source of truth)
      file_naming.json       # File naming conventions
    steps/                   # Preprocessing scripts (01-09)
    analysis/                # Feature extraction & visualisation (10-16)
    src/                     # Shared utilities
      utils_io.py            # I/O helpers (load, save, glob)
      utils_qc.py            # Basic QC filters & bad channel detection
      utils_config.py        # parameters.json loader with caching & helpers
      utils_naming.py        # file_naming.json integration
      utils_report.py        # QC report generation (figures + markdown)
      utils_features.py      # Feature extraction helpers
      utils_clean.py         # Cleaning utilities (robust reference)
    outputs/                 # All generated outputs (gitignored)
      derivatives/           # Intermediate processing files (.fif)
      features/              # Extracted feature CSVs
      analysis_figures/      # Per-subject analysis plots
      group_figures/         # Group-level comparison plots
      qc/                    # Per-subject QC figures
      reports/               # Per-subject assessment reports (.md)
  docs/                      # Methodology documentation
  raw/                       # Raw EEG data (gitignored)
```

---

## Experimental Protocol

```
[3 min rest] -> [~60 min practice] -> [break] -> [65-70 min WAND induction] -> [3 min rest]
     ^                                                                               ^
  IAF pre                                                                         IAF post
  (resting)                                                                       (resting)
```

- **Resting state** (pre + post): Eyes-closed, ~3 min each -> IAF extraction
- **Task blocks**: Block 1 (baseline) and Block 5 (fatigued) -> theta, PAC, ERP analysis
- **Fatigue induction**: 65-70 min sustained n-back task (WAND protocol)

---

## Pipeline Steps

### Preprocessing (`eeg_pipeline/steps/`)

| Step | Script | Function | Config Section |
|:-----|:-------|:---------|:---------------|
| 00 | `00_make_fake_data.py` | Generate synthetic EEG for testing | -- |
| 00 | `00_view_raw.py` | Visual inspection of raw files | -- |
| 01 | `01_import_qc.py` | Import, filter, bad channel detection | `filtering` |
| 02 | `02_clean_reference.py` | Robust average reference (PyPREP) | -- |
| 04 | `04_zapline.py` | Line noise removal (ZapLine spatial filtering) | `zapline`, `filtering.notch_freq` |
| 05 | `05_ica_iclabel.py` | ICA + ICLabel automated classification | `ica` |
| 06 | `06_asr.py` | Artifact Subspace Reconstruction | `asr` |
| 07 | `07_epoch.py` | Epoching (P3b stimulus-locked + PAC maintenance) | `epoching` |
| 08 | `08_autoreject.py` | Automated epoch rejection/repair | `autoreject` |
| 09 | `09_features.py` | Basic feature extraction | -- |

### Analysis (`eeg_pipeline/analysis/`)

All analyses compare **Block 1** (baseline) with **Block 5** (fatigued).

| Step | Script | Function | Config Section |
|:-----|:-------|:---------|:---------------|
| 10 | `10_erp_p3b.py` | P3b ERP amplitude & latency at Pz | `p3b` |
| 11 | `11_band_power.py` | Frontal theta + posterior alpha power | `band_power`, `iaf`, `itf` |
| 12 | `12_peak_frequencies.py` | IAF (resting) + ITF (task) via specparam | `specparam`, `iaf`, `itf` |
| 13 | `13_pac_nodal.py` | Local + between-region PAC (theta-gamma) | `pac` |
| 16 | `16_merge_features.py` | Merge all feature CSVs into wide format | -- |
| -- | `visualise_outputs.py` | Per-subject PSD, PAC, ERP, IAF plots | -- |
| -- | `visualise_group.py` | Group-level paired comparisons & stats | -- |

---

## GUI Workflow (Recommended for Researchers)

The project now includes two connected GUIs for a front-to-back workflow:

1. **Parameter Settings GUI** (`eeg_pipeline/parameter_gui.py`)
2. **Pipeline Runner GUI** (`eeg_pipeline/pipeline_runner_gui.py`)

### What each GUI does

- `parameter_gui.py`
  - Edit `parameters.json` with field-level tooltips.
  - Save/load parameter presets (`eeg_pipeline/config/presets/*.json`).
  - Shows a parameter hash fingerprint for reproducibility tracking.
  - Includes a **Run Pipeline** button to open the runner directly.

- `pipeline_runner_gui.py`
  - Run preprocessing only, analysis only, or full pipeline.
  - Select raw data folder/pattern/format for the run (runtime override).
  - Optionally run with a selected preset without overwriting `parameters.json`.
  - Live step-by-step console log and progress bar.
  - Includes an **Open Settings** button to return to the parameter GUI.

### Commands

```bash
# Open parameter settings GUI
python eeg_pipeline/parameter_gui.py

# Open pipeline runner GUI
python eeg_pipeline/pipeline_runner_gui.py
```

### Presets and what actually runs

- Saving a preset creates a JSON file in `eeg_pipeline/config/presets/`.
- A preset is used only if:
  1. You load it in the settings GUI and click **Save** (writes to `parameters.json`), or
  2. You select it in the runner GUI and tick **Use selected preset for this run**.

### Epoching and triggers

Epoching is automatic in Step 07 (`eeg_pipeline/steps/07_epoch.py`):
- Events are read from annotations.
- P3b epochs are created from stimulus-onset events.
- PAC epochs are created from stimulus-onset events shifted by the configured stimulus duration offset.

---

## Parameter Configuration (`parameters.json`)

All algorithm parameters are centralised in `eeg_pipeline/config/parameters.json`. This is the **single source of truth** for tuning the pipeline without editing code.

### Key Sections

| Section | Controls | Key Parameters |
|:--------|:---------|:---------------|
| `filtering` | Bandpass & notch | `hp_freq`, `lp_freq`, `notch_freq` |
| `zapline` | Line noise removal | `n_harmonics` |
| `ica` | ICA decomposition | `n_components`, `method`, `iclabel_thresholds` |
| `asr` | Burst cleaning | `cutoff` (lower = more aggressive) |
| `autoreject` | Epoch rejection | `n_interpolate`, `consensus`, `cv` |
| `epoching` | Epoch windows | P3b: `tmin`/`tmax`/`baseline`; PAC: offset |
| `p3b` | P3b analysis | `channels` (default: `["Pz"]`), `tmin_peak`, `tmax_peak` |
| `specparam` | Spectral fitting | `freq_range`, `peak_width_limits`, `max_n_peaks` |
| `pac` | Phase-amplitude coupling | `phase_band`, `amp_band`, `n_surrogates` |
| `qc` | Quality thresholds | `max_epoch_rejection_pct`, `max_ica_components_rejected` |

### Parameter Tuning Workflow

```
1. Process first subject
2. Review QC plots in outputs/qc/
3. Adjust parameters.json if needed
4. Re-run affected step(s) only
5. Compare QC before/after -> iterate
6. Final parameters automatically recorded in assessment reports
```

### Accessing Parameters in Code

```python
from src.utils_config import get_param

ica_components = get_param('ica', 'n_components')           # -> 25
eye_threshold = get_param('ica', 'iclabel_thresholds', 'eye')  # -> 0.80
asr_cutoff = get_param('asr', 'cutoff')                     # -> 20
```

---

## Quality Control Framework

Every preprocessing step generates:

1. **QC figures** saved to `outputs/qc/<subject>/block<N>/`
2. **Assessment reports** (markdown) saved to `outputs/reports/`
3. **Automatic pass/fail** based on configurable thresholds

### Assessment Criteria

| Step | Metric | Threshold | Action if Failed |
|:-----|:-------|:----------|:-----------------|
| ZapLine | 50 Hz power reduction | >10 dB attenuation | Adjust n_harmonics |
| ICA | Components rejected | 1-8 typical | Review IC labels |
| ICA | Brain ICs remaining | >15 | May be over-rejecting |
| ASR | Data modified | <30% | Raise cutoff |
| Autoreject | Epochs rejected | <30% | Adjust consensus |
| P3b | Peak latency | 250-600 ms | Check epoch timing |

### QC Report Module

```python
from src.utils_report import QCReport, qc_psd_overlay

qc = QCReport('sub-001', block=1)
qc.log_step('04_zapline', status='PASS', metrics={...})
qc.add_figure('psd_overlay', fig)
qc.save_report()  # -> outputs/reports/sub-001_block1_assessment.md
```

---

## Individualised Frequency Bands

All frequency band analyses use **specparam** (spectral parameterisation) to extract subject-specific peak frequencies:

- **Individual Alpha Frequency (IAF):** Extracted from resting-state recordings at posterior channels. Alpha band = IAF +/- 2 Hz.
- **Individual Theta Frequency (ITF):** Extracted from task maintenance epochs at frontal midline (Fz). Theta band = ITF +/- 2 Hz, capped at IAF - 1 Hz to prevent alpha leakage.
- **Fallback:** When specparam cannot detect a clear peak, canonical bands (theta 4-8 Hz, alpha 8-13 Hz) are used.
- **Band inversion guards:** If calculated band bounds invert, defaults are used automatically.

---

## File Naming

All file naming patterns are defined in `eeg_pipeline/config/file_naming.json` and accessed via `src/utils_naming.py`:

```python
from src.utils_naming import get_derivative_name, get_feature_name, get_figure_name

get_derivative_name('sub-001', 1, 'asr_cleaned')  # -> 'sub-001_block1_asr-raw.fif'
get_feature_name('p3b')                            # -> 'p3b_features.csv'
get_figure_name('p3b_erp')                         # -> 'p3b_erp_filtered.png'
```

---

## Synthetic Data

### Lay Overview

The fake data generator (`00_make_fake_data.py`) creates two realistic task blocks:

- **Block 1** = baseline (less fatigued)
- **Block 5** = fatigued (after cognitive load)

It is designed so the "tired" block looks clearly different from baseline in both behavior and EEG:

- More misses and more errors
- Slower responses
- Smaller and later P3b
- Slower alpha/theta frequencies
- Lower posterior alpha power
- Higher frontal theta power
- Lower frontal->parietal theta-gamma coupling

This gives a practical mock-run dataset that should show the expected direction of effects in output CSVs and plots.

### Technical Summary

Current synthetic generator behavior:

- **Channels**: 64 EEG channels + 4 aux channels (`VEOG`, `HEOG`, `EMG_L`, `EMG_R`)
- **Timing model**: fixed `STIM_DUR` (0.2 s) and separate maintenance trigger at `MAINTENANCE_OFFSET` (0.8 s)
- **Event stream**: deterministic-rate target/miss/error assignment (reduced random inversion risk between blocks)
- **EEG background**: pink 1/f noise + slow drift + line noise (50/100 Hz)
- **Rhythms**: posterior alpha, fronto-midline theta, tonic high gamma
- **ERP model**: N1/P2 sensory components + target-only P3b (fatigue reduces amplitude and delays latency)
- **PAC model**: directed RF theta phase -> RP gamma amplitude coupling in maintenance epochs
- **Artifact model**: blinks with frontal leak, EMG bursts around responses, EOG/EMG auxiliary channels

Expected fatigue direction for `sub-TEST01` mock runs:

| Measure | Block 5 vs Block 1 |
|:--------|:-------------------|
| Misses / errors | Higher |
| P3b amplitude | Lower |
| P3b latency | Higher |
| Theta power (frontal) | Higher |
| Alpha power (posterior) | Lower |
| Theta peak frequency (`f_theta`) | Lower |
| IAF (post vs pre) | Lower |
| Between-region PAC (`RF->RP`) | Lower |

### Quick Mock-Run Verification Checklist

Run this when you want to confirm the synthetic generator is producing the expected fatigue pattern.

1. Regenerate synthetic data:
```bash
python eeg_pipeline/steps/00_make_fake_data.py
```

2. Run preprocessing + analysis:
```bash
python eeg_pipeline/steps/01_import_qc.py
python eeg_pipeline/steps/02_clean_reference.py
python eeg_pipeline/steps/04_zapline.py
python eeg_pipeline/steps/05_ica_iclabel.py
python eeg_pipeline/steps/06_asr.py
python eeg_pipeline/steps/07_epoch.py
python eeg_pipeline/steps/08_autoreject.py
python eeg_pipeline/analysis/10_erp_p3b.py
python eeg_pipeline/analysis/11_band_power.py
python eeg_pipeline/analysis/12_peak_frequencies.py
python eeg_pipeline/analysis/13_pac_nodal.py
python eeg_pipeline/analysis/16_merge_features.py
```

3. Open `eeg_pipeline/outputs/features/merged_wide.csv` and compare `sub-TEST01` Block 5 vs Block 1:
- `p3b_amp_uV`: lower
- `p3b_lat_ms`: higher
- `theta_power_log`: higher
- `alpha_power_log`: lower
- `f_theta`: lower
- `pac_between_RF_RP`: lower
- `iaf_shift`: negative (post < pre)

4. Open `eeg_pipeline/raw/sub-TEST01_block1-task.vhdr` and `eeg_pipeline/raw/sub-TEST01_block5-task.vhdr` in MNE/BrainVision viewer and confirm:
- Block 5 has visibly weaker P3b-like positivity at posterior channels
- Block 5 has higher miss/error event counts
- Block 5 has slower alpha/theta rhythms

---

## Output Files

### Per-Subject Features (`outputs/features/`)

| File | Contents |
|:-----|:---------|
| `p3b_features.csv` | P3b amplitude (uV) and peak latency (ms) |
| `theta_power_features.csv` | Log frontal midline theta power |
| `alpha_power_features.csv` | Log posterior alpha power |
| `theta_freq_features.csv` | Individual theta frequency (Hz) |
| `iaf_features.csv` | Individual alpha frequency pre/post |
| `pac_local_features.csv` | Node-level PAC (z-scored MI) |
| `pac_between_features.csv` | Between-region PAC (RF->RP) |
| `merged_wide.csv` | All features joined (one row per subject x block) |

### Group Figures (`outputs/group_figures/`)

| File | Contents |
|:-----|:---------|
| `group_paired_comparisons.png` | Violin + paired scatter for all features |
| `group_delta_barchart.png` | % change bar chart (B5 - B1) |
| `group_iaf_shift.png` | IAF pre vs post fatigue |
| `group_summary_stats.csv` | Mean, SD, t, p, Cohen's d per feature |

---

## Dependencies

| Package | Purpose |
|:--------|:--------|
| `mne` | EEG processing core |
| `specparam` | Spectral parameterisation (periodic vs aperiodic) |
| `tensorpac` | Phase-amplitude coupling (MI + surrogates) |
| `pyprep` | Robust referencing (RANSAC) |
| `meegkit` | ZapLine line noise removal, ASR |
| `autoreject` | Epoch-level artifact rejection |
| `mne-icalabel` | Automated ICA component classification |
| `pandas` | Feature data management |
| `scipy` | Statistical testing |
| `matplotlib` | Visualisation |

---

## Installation

```bash
git clone https://github.com/brodie-neuro/EEG-Active-Cognitive-Fatigue-Pipeline.git
cd EEG-Active-Cognitive-Fatigue-Pipeline
pip install -r requirements.txt
```

---

## Quick Start

```bash
# Optional GUI workflow
python eeg_pipeline/parameter_gui.py
python eeg_pipeline/pipeline_runner_gui.py

# 1. Generate synthetic test data
python eeg_pipeline/steps/00_make_fake_data.py

# 2. Run preprocessing (steps 01-08)
python eeg_pipeline/steps/01_import_qc.py
python eeg_pipeline/steps/02_clean_reference.py
python eeg_pipeline/steps/04_zapline.py
python eeg_pipeline/steps/05_ica_iclabel.py
python eeg_pipeline/steps/06_asr.py
python eeg_pipeline/steps/07_epoch.py
python eeg_pipeline/steps/08_autoreject.py

# 3. Run analysis (steps 10-13)
python eeg_pipeline/analysis/10_erp_p3b.py
python eeg_pipeline/analysis/11_band_power.py
python eeg_pipeline/analysis/12_peak_frequencies.py
python eeg_pipeline/analysis/13_pac_nodal.py

# 4. Merge features & visualise
python eeg_pipeline/analysis/16_merge_features.py
python eeg_pipeline/analysis/visualise_outputs.py
python eeg_pipeline/analysis/visualise_group.py
```

---

## Testing

Current automated tests:

- `eeg_pipeline/tests/test_step05_ica.py`
  - Verifies ICA derivative outputs and basic signal integrity.
- `eeg_pipeline/tests/test_gui_smoke.py`
  - Smoke tests both GUIs by creating hidden Tk windows and verifying initialization.
  - Skips automatically when no display server is available.

Run tests:

```bash
python -m pytest eeg_pipeline/tests/test_gui_smoke.py -q
python -m pytest eeg_pipeline/tests/test_step05_ica.py -q
```

Note: There are currently no pixel/screenshot regression tests for GUI appearance.

---

## Documentation

See `docs/` for detailed methodology:

- `implementation_plan_qc_framework.md` -- QC framework, assessment criteria, parameter tuning guide
- `preprocessing_pipeline.md` -- Preprocessing rationale and steps
- `methodology_pac_zscore.md` -- Why surrogate z-scores for PAC

---

## License

All Rights Reserved (c) Brodie E. Mangan
