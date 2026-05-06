# Study-Specific Analysis Exclusions

This document records participant-level exclusions for the current active cognitive fatigue dataset analysed in this repository. These exclusions are not general pipeline rules and should not be reused automatically for other datasets.

Researchers applying this pipeline to a new dataset should create their own exclusion log from that dataset's QC outputs, behavioural logs, and pre-specified analysis decisions. Feature extraction and QC scripts should still compute all available participant rows where possible, so excluded cases remain auditable.

## Current Analysis Rules

| Analysis | Excluded participants | Applies to | Reason |
|:--|:--|:--|:--|
| H1 theta-gamma PAC | none | Primary theta-gamma PAC inference and theta-gamma EMG sensitivity checks | No current participant-level exclusion is applied to H1. |
| H2 alpha-gamma PAC | `sub-p005`, `sub-p011` | H2 alpha-gamma PAC inference, alpha-gamma EMG-corrected PAC sensitivity, and alpha-gamma EMG/PAC delta correlation | `sub-p005`: frontal alpha phase QC showed weak, fragmented, edge-dominated residual support after aperiodic removal, so fixed 8-13 Hz frontal alpha phase is not treated as interpretable. `sub-p011`: Block 5 alpha-gamma PAC has too few usable trials in the current feature table, so paired H2 inference is not treated as reliable. |
| H3 P3b | `sub-p005` | Secondary P3b inference | P3b QC showed a large negative-going posterior deflection rather than a plausible P3b response in the configured window. |

## Implementation Notes

- H2 alpha-gamma exclusions for this dataset are centralised in `alpha_gamma_pac.analysis_excluded_subjects` in `eeg_pipeline/config/parameters.json`.
- The H2 exclusion set is applied by `09b_alpha_gamma_pac_emg_corrected.py` and `15_emg_alpha_gamma_correlation.py`.
- The H3 P3b exclusion is applied at the H3 inference/reporting stage. P3b feature extraction and P3b QC still retain the participant for audit.
- Raw feature tables are not deleted or rewritten as the primary exclusion mechanism. Analysis-stage filtering is preferred so that QC review remains transparent.
- The exclusions above are analysis-specific. For example, `sub-p005` is excluded from H2 alpha-gamma and H3 P3b, but not from H1 theta-gamma PAC.
- For a different dataset, clear or replace the study-specific subject IDs in `alpha_gamma_pac.analysis_excluded_subjects` and update this document accordingly.
