# Baseline EEG Preprocessing Pipeline

This folder contains a **traditional/baseline preprocessing pipeline** for comparison with the advanced pipeline in `eeg_pipeline/`.

## Purpose

To demonstrate that the advanced pipeline (Zapline + ICLabel + ASR + Node aggregation) provides measurable improvements over standard approaches.

## Comparison Metrics

When comparing pipelines on the same dataset, measure:

1. **Gamma-band SNR** - Power spectral density at 30-80 Hz post-cleaning
2. **Epoch Retention Rate** - % of trials surviving artifact rejection
3. **PAC Effect Size** - Cohen's d for condition differences (e.g., high vs low fatigue)
4. **PAC Z-Score Magnitude** - Higher = stronger coupling detected
5. **Muscle Artifact Residual** - EMG-EEG coherence post-cleaning

## Pipeline Steps

| Step | Baseline (Traditional) | Advanced (eeg_pipeline) |
|:-----|:-----------------------|:------------------------|
| Line noise | Notch filter (50 Hz) | Zapline spatial filtering |
| Artifact removal | ICA + manual/ADJUST | ICLabel (automated, 80% threshold) |
| Transient artifacts | Epoch rejection only | ASR + Autoreject |
| Reference | Average reference | Robust reference (PyPREP) |
| PAC calculation | Electrode-level MI | Z-scored → Trimmed mean → Nodes |

## Expected Outcome

The advanced pipeline should show:
- Higher epoch retention (less data loss)
- Cleaner gamma band (less muscle contamination)
- Larger PAC effect sizes (better sensitivity)
