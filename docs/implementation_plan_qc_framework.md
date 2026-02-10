# Implementation Plan: QC, Logging, Assessment & Parameter Framework

## Overview

This plan addresses the need for:
1. **Visual QC outputs** at every preprocessing stage
2. **Comprehensive logging** at every stage
3. **JSON-driven algorithm parameters** (adjustable without editing scripts)
4. **Per-subject assessment reports** (pass/fail criteria)
5. **Individual + merged data files** with flexible visualisation
6. **`file_naming.json` integration** across all scripts

---

## 1. JSON Parameter Configuration (`parameters.json`)

### Goal
A single JSON file that controls all algorithm parameters for every pipeline step. Users can tune preprocessing without touching Python code.

### File: `eeg_pipeline/config/parameters.json`

```json
{
  "filtering": {
    "hp_freq": 0.1,
    "lp_freq": 120.0,
    "notch_freq": 50.0,
    "notch_width": 2.0
  },
  "zapline": {
    "n_harmonics": 4,
    "chunk_length": 30
  },
  "ica": {
    "n_components": 25,
    "method": "infomax",
    "random_state": 42,
    "iclabel_thresholds": {
      "eye": 0.80,
      "heart": 0.80,
      "muscle": 0.90,
      "channel_noise": 0.95,
      "line_noise": 0.95,
      "other": 0.99
    }
  },
  "asr": {
    "cutoff": 20,
    "method": "euclid",
    "win_len": 0.5
  },
  "autoreject": {
    "n_interpolate": [1, 4, 8, 16],
    "consensus": [0.1, 0.5, 1.0],
    "cv": 5,
    "random_state": 42
  },
  "epoching": {
    "p3b": {
      "tmin": -0.2,
      "tmax": 0.8,
      "baseline": [-0.2, 0.0]
    },
    "pac": {
      "tmin_offset": 0.8,
      "tmax_offset": 1.8,
      "baseline": null
    }
  },
  "specparam": {
    "freq_range": [1, 30],
    "peak_width_limits": [1, 8],
    "max_n_peaks": 6,
    "min_peak_height": 0.1,
    "aperiodic_mode": "fixed"
  },
  "pac": {
    "phase_band": [4, 8],
    "amp_band": [55, 85],
    "n_surrogates": 200,
    "trim": 0.10
  }
}
```

### Integration Pattern
Every script loads parameters via a utility:
```python
from src.utils_config import load_parameters
params = load_parameters()  # Loads parameters.json
ica_params = params['ica']
```

### Adjustability Workflow
1. Run pipeline on a subject
2. Inspect QC outputs (see Section 2)
3. If over/under-cleaned: adjust `parameters.json` values
4. Re-run affected step(s) only
5. Compare QC outputs — iterate until satisfactory

---

## 2. Visual QC Outputs at Every Stage

### Directory Structure
```
eeg_pipeline/outputs/qc/
  sub-001/
    block1/
      01_import_raw_psd.png           # Raw PSD before any processing
      02_post_reference_psd.png       # PSD after re-referencing
      03_post_zapline_psd.png         # PSD after ZapLine (50 Hz removed?)
      04_ica_components.png           # IC topographies + labels
      04_ica_sources_timeseries.png   # IC time series (first 10s)
      04_ica_removed_overlay.png      # Before/after ICA overlay
      05_asr_cleaning_overlay.png     # ASR: original vs cleaned segments
      05_asr_variance_timeseries.png  # Channel variance over time
      06_epochs_drop_log.png          # Autoreject drop log
      06_epochs_evoked.png            # Grand average ERP after rejection
      07_psd_final.png                # Final PSD (post all cleaning)
    block5/
      ... (same structure)
  sub-002/
    ...
```

### Per-Step QC Details

| Step | Script | QC Outputs | What to Check |
|------|--------|-----------|--------------|
| 01 | `01_import_qc.py` | Raw PSD, channel map, event summary | Flat/noisy channels, correct montage, correct event count |
| 02 | `02_clean_reference.py` | PSD overlay (before/after), bad channel list | Reference applied correctly, no extreme channels |
| 04 | `04_zapline.py` | PSD overlay showing 50Hz removal | Line noise adequately removed without distorting signal |
| 05 | `05_ica_iclabel.py` | IC topographies, component labels, removed components overlay | Correct artifact identification, not over-rejecting brain ICs |
| 06 | `06_asr.py` | Variance timeseries, cleaned segments, PSD overlay | Not overcleaning (>30% data modified = warning) |
| 07 | `07_epoch.py` | Event count summary, epoch duration check | Correct number of epochs, correct timing |
| 08 | `08_autoreject.py` | Drop log, rejection summary, evoked overlay | Acceptable rejection rate (<30%), ERP shape preserved |

### Implementation
Each step script gets a `generate_qc(subj, block, data_before, data_after, params)` function.

---

## 3. Comprehensive Logging

### Current State
`utils_logging.py` exists with console + file logging. Needs expansion.

### Enhanced Logging Per Step
Each step produces a structured log entry:

```
eeg_pipeline/outputs/logs/
  sub-001/
    block1_step01_import.log
    block1_step02_reference.log
    block1_step04_zapline.log
    block1_step05_ica.log
    block1_step06_asr.log
    block1_step07_epoch.log
    block1_step08_autoreject.log
    block1_step10_p3b.log
    block1_step11_bandpower.log
    block1_step12_peaks.log
    block1_step13_pac.log
```

### Log Content Standard
Every log entry must include:
- Timestamp
- Input file path
- Parameters used (from `parameters.json`)
- Key metrics (channels count, epoch count, rejection rate, etc.)
- Warnings/errors
- Output file path
- Pass/fail assessment (see Section 4)

---

## 4. Per-Subject Assessment Reports

### Goal
After processing each subject, generate an assessment markdown file summarising whether each step met quality criteria.

### File: `eeg_pipeline/outputs/reports/sub-001_block1_assessment.md`

### Assessment Criteria

| Step | Metric | Acceptable Range | Action if Failed |
|------|--------|-----------------|-----------------|
| Import | Channel count | Expected channel count (e.g. 64) | Re-check data file |
| Import | Event count | > 0 events per block | Check trigger setup |
| Reference | Bad channels | < 10% of total | Inspect manually |
| ZapLine | 50 Hz power reduction | > 10 dB attenuation | Adjust n_harmonics |
| ICA | Components rejected | 1-8 (typical) | Review IC labels |
| ICA | Brain ICs remaining | > 15 | May be over-rejecting |
| ASR | Data modified % | < 30% | Lower cutoff parameter |
| Autoreject | Epochs rejected | < 30% of total | Adjust consensus |
| Epochs | Trial count | > 20 per condition | Check for data loss |
| P3b | Peak latency | 250-600 ms | Check epoch timing |
| P3b | Peak amplitude | > 0.5 uV | Check data quality |
| Theta peak | Frequency | 3-9 Hz | Check specparam fit |
| IAF | Frequency | 6-14 Hz | Check posterior PSD |
| PAC | MI > 0 | Positive after z-scoring | Check epoch quality |

### Report Format
```markdown
# Subject Assessment: sub-001, Block 1

## Summary
- **Overall**: PASS (11/12 criteria met)
- **Date processed**: 2026-02-10  
- **Parameters file**: parameters.json (hash: abc123)

## Step-by-Step Assessment

### Step 01: Import & QC
- Status: PASS
- Channels: 64 (expected: 64)
- Events detected: 156
- Duration: 480.2 s

### Step 05: ICA & ICLabel
- Status: PASS  
- Components removed: 4 (eye: 2, heart: 1, muscle: 1)
- Brain ICs remaining: 21
- [QC figure: 05_ica_components.png]

### Step 06: ASR
- Status: WARNING
- Data modified: 28.3% (threshold: 30%)
- Consider: Raising ASR cutoff from 20 to 25
...
```

---

## 5. Individual + Merged Data Architecture

### Per-Subject Feature Files (already exists, keep as-is)
```
eeg_pipeline/outputs/features/
  p3b_features.csv          # All subjects, long format
  theta_power_features.csv
  alpha_power_features.csv
  theta_freq_features.csv
  iaf_features.csv
  pac_local_features.csv
  pac_between_features.csv
```

Each CSV uses **append mode** — new subjects add rows. Columns: `subject`, `block`, plus feature columns.

### Merged Feature File
```
eeg_pipeline/outputs/features/merged_features.csv
```
Created by `16_merge_features.py`:
- Joins all individual feature CSVs on `subject + block`
- One row per subject × block with ALL features as columns
- Includes IAF (joined on subject, using pre-task value for both blocks)

### Group-Level Visualisation
```
eeg_pipeline/analysis/visualise_group.py
```
- Reads `merged_features.csv`
- Produces group-level plots:
  - **Violin/box plots**: B1 vs B5 for each feature
  - **Paired scatter plots**: Each subject's paired change
  - **Summary table**: Mean +/- SD per block, effect size, p-value
- Can be invoked for any subset or all subjects

### Individual Visualisation (already exists)
```
eeg_pipeline/analysis/visualise_outputs.py
```
- Per-subject PSD, ERP, PAC plots
- Uses dynamic subject discovery (now fixed)

### Workflow
```
1. Process sub-001: run steps 01-08, then 10-13
   → Individual feature rows appended to CSVs
   → Individual QC plots + assessment report generated

2. Process sub-002: same pipeline
   → Rows appended to same CSVs

3. When ready: run 16_merge_features.py
   → merged_features.csv created

4. Visualise: run visualise_group.py
   → Group-level figures generated
```

---

## 6. `file_naming.json` Integration

### Current State
`eeg_pipeline/config/file_naming.json` exists but is not imported by any script.

### Integration Plan
Create `src/utils_naming.py`:
```python
import json
from pathlib import Path

_NAMING = None

def load_naming():
    global _NAMING
    if _NAMING is None:
        cfg_path = Path(__file__).parents[1] / "config" / "file_naming.json"
        with open(cfg_path) as f:
            _NAMING = json.load(f)
    return _NAMING

def get_derivative_name(subject, block, step, suffix):
    """Build standardised derivative file name."""
    naming = load_naming()
    pattern = naming['derivatives'].get(step, '{subject}_block{block}_{step}{suffix}')
    return pattern.format(subject=subject, block=block, step=step, suffix=suffix)

def get_feature_name(feature_type):
    """Get standardised feature CSV filename."""
    naming = load_naming()
    return naming['features'].get(feature_type, f'{feature_type}_features.csv')
```

Replace all hardcoded file patterns in steps and analysis scripts with calls to `utils_naming`.

---

## 7. Priority Order for Implementation

| Priority | Task | Status | Impact |
|----------|------|--------|--------|
| 1 | Create `parameters.json` + `utils_config.py` loader | DONE | HIGH |
| 2 | Integrate parameters.json into steps 04-08, 10, 12 | DONE | HIGH |
| 3 | Create `utils_report.py` assessment report generator | DONE | HIGH |
| 4 | Create `utils_naming.py` for `file_naming.json` | DONE | MEDIUM |
| 5 | Create `16_merge_features.py` | DONE | HIGH |
| 6 | Create `visualise_group.py` | TODO | HIGH |
| 7 | Integrate QC report into steps 01-02 | TODO | MEDIUM |
| 8 | Build parameter GUI (tkinter) | TODO | HIGH |
| 9 | Enhance logging in all steps | TODO | MEDIUM |

---

## 8. Parameter Tuning Workflow (Assessment Methodology)

### When to Adjust Parameters
```
1. Run pipeline on first subject
2. Review QC plots:
   a. ICA: Are eye/heart components correctly identified?
      → If missing blinks: lower iclabel_thresholds.eye (e.g. 0.7)
      → If brain ICs removed: raise threshold (e.g. 0.9)
   
   b. ASR: What % of data is modified?
      → >30%: raise cutoff (e.g. 25 → 30)
      → <5%: lower cutoff (e.g. 25 → 15) — may not be cleaning enough
   
   c. Autoreject: What % of epochs rejected?
      → >30%: widen consensus range or increase n_interpolate
      → <5%: may need stricter rejection
   
   d. ZapLine: Is 50 Hz still visible?
      → Increase n_harmonics
      → Check if 100 Hz harmonic also needs removal
3. Adjust parameters.json
4. Re-run ONLY the affected step (and downstream)
5. Compare QC plots
6. Document final parameters in assessment report
```

### Recommended Assessment Protocol for Each Subject
1. **Visual scan**: Look at raw PSD — any obvious pathology?
2. **ICA review**: Inspect component topographies — correct labelling?
3. **ASR check**: Compare pre/post variance — overcleaned?
4. **Epoch quality**: Drop rate acceptable? ERP shape present?
5. **Feature sanity**: Are values within expected ranges?
6. **Sign off**: Mark subject as QC-passed or flag for review

---

## Notes
- All QC outputs should use `matplotlib.use('Agg')` backend
- Log files use UTF-8 encoding (avoid Unicode issues on Windows)
- Parameters.json should be version-controlled (tracked in git)
- assessment reports should be gitignored (per-subject, local)
