# Parameters and Method Rationale

This file is the repo-facing summary of why the main pipeline parameters were chosen.

The manuscript should carry the full literature narrative and formal methods text. This file has a different job: it lets a collaborator, reviewer, or future maintainer understand the live parameter choices without having to reverse-engineer them from the code.

## Where Parameters Currently Live

- `eeg_pipeline/config/parameters.json`: main preprocessing settings and many analysis settings
- `eeg_pipeline/config/study.yml`: study paths, block structure, and study-level settings
- `eeg_pipeline/config/participant_configs/*.json` (user-created, not shipped): optional per-subject overrides such as known bad channels. Each study creates its own.
- selected analysis scripts: some scientifically meaningful constants still remain in code and should be treated as part of the live method until they are moved into config

That last point matters. The repo is mostly config-driven, but not yet fully centralized in one file.

## Guiding Philosophy

The pipeline is built around four practical goals:

1. preserve useful neural signal, especially gamma, rather than cleaning aggressively by default
2. make subject-specific exceptions explicit instead of handling them informally
3. keep the main preprocessing path deterministic and auditable
4. separate confirmatory analyses from descriptive follow-up measures

## Core Preprocessing Choices

| Area | Current setting | Why it is used | Publication note |
|:-----|:----------------|:---------------|:-----------------|
| Bandpass filtering | `0.1-100 Hz` | Keeps slow ERP content while retaining higher-frequency activity needed for gamma analyses. | Keep manuscript and code aligned on the exact filter design. |
| Line-noise handling | deterministic FIR notch at `50 Hz` and `100 Hz` | Removes mains contamination without introducing another decomposition-heavy preprocessing step. | Keep the manuscript aligned with the simplified public path rather than older internal variants. |
| EEG reference | simple average reference after interpolating only known bad EEG channels | Keeps the reference step explicit and auditable while avoiding fresh data-driven bad-channel branching during production runs. | Subject-specific known bad EEG channels should remain locked in participant config. |
| Subject-specific bad channels | `participant_configs/*.json` with `known_bad_eeg` and `known_bad_emg` | Makes known hardware or contact failures explicit and reproducible. | This is a strength of the repo and should be preserved. |
| ASR burst cleaning | `cutoff = 30`, `method = euclid`, `win_len = 0.5`, `use_clean_windows = true` | This is intentionally conservative. A higher cutoff than the often-cited default of 20 means the step targets only more extreme bursts and is less likely to flatten task-related signal. | The repo currently flags blocks at a relatively permissive modification threshold (`qc.max_asr_modified_pct = 65`) so heavily reconstructed blocks can be reviewed rather than silently treated as routine. |
| ICA decomposition | `n_components = 25`, `method = infomax`, `random_state = 42`, fit on a `1 Hz` high-pass copy | Fitting on a higher-pass copy is standard practice for more stable ICA, while applying the solution back to the broader-band data preserves low-frequency ERP content. | The rationale is good, but `25` components should remain explicitly defended rather than treated as arbitrary. |
| ICLabel thresholds | eye `0.85`, heart `0.85`, muscle `0.90` | These are fairly conservative automatic rejection thresholds intended to remove obvious artifacts without over-pruning the decomposition. | Good candidate for a brief citation-backed note in a methods appendix or release note. |
| Epoching for P3b | `-0.2 to 0.8 s`, baseline `-0.2 to 0.0 s` | Standard stimulus-locked ERP windowing with pre-stimulus baseline correction. | P3b logic should stay aligned with the actual trigger structure used in the task. |
| Epoching for PAC / connectivity | `-0.5 to 1.8 s`, no baseline | Gives enough temporal context for low-frequency phase estimation and edge trimming in time-frequency analyses. | This broader window is methodologically sensible for oscillatory work. |
| Autoreject search grid | `n_interpolate = [1, 4, 8, 16]`, `consensus = [0.1, 0.5, 1.0]`, `cv = 5` | Standard automated repair / rejection tradeoff without hand-tuning each subject. | Reasonable default grid; useful to keep documented rather than implicit. |

## Main Analysis Choices

| Area | Current setting | Why it is used | Publication note |
|:-----|:----------------|:---------------|:-----------------|
| Confirmatory PAC nodes | `C_broad_F` and `C_broad_P` bilateral composites | Broad bilateral nodes are a sensible choice for 64-channel EEG where precise source separation is limited. They support a cleaner confirmatory frontoparietal story than many tiny ad hoc ROIs. | The node definitions should stay identical across README, manuscript, and scripts. |
| P3b channel set | config currently lists `Pz`, `P1`, `P2`, `POz` | Keeps the analysis anchored to a posterior P3b region while still allowing a canonical `Pz` interpretation. | If the manuscript presents P3b as strictly `Pz`, keep that wording aligned with the code output. |
| P3b time window | `0.3-0.5 s` | Conventional post-stimulus window for P3b-like activity in this task family. | Stronger if tied explicitly to task timing and literature. |
| PAC phase band | `4-8 Hz` theta | Canonical working-memory theta range, suitable for confirmatory PAC hypotheses. | Defend as a canonical confirmatory choice rather than an individualized one. |
| PAC amplitude band | `55-85 Hz` gamma | Narrow enough to avoid lower-frequency bleed while still targeting task-relevant high gamma. | Important to keep coupled with the EMG-control story. |
| PAC null model | `500` surrogates | Enough to generate a stable z-scored modulation index without making runtime unreasonable. | Reasonable compromise between speed and null stability. |
| PAC edge trimming | `trim = 0.1 s` | Reduces edge artifacts in filtered analytic signals. | Sensible implementation detail worth keeping explicit. |
| Theta synchrony | confirmatory frontoparietal theta wPLI in the stimulus window | Fits the hypothesis that fatigue alters communication across a control-relevant network rather than only local amplitude. | The confirmatory window should be documented consistently in script and manuscript. |
| Spectral parameterization | `freq_range = 3-30`, `peak_width_limits = [1, 6]`, `max_n_peaks = 6`, `zero_pad_factor = 4` | Separates periodic peaks from the aperiodic background and improves frequency resolution for short stimulus-locked epochs. | Zero padding should be described carefully as improving frequency sampling, not creating new information. |
| Default theta band | theta `4-8` | Good fallback range when individualized peaks are unavailable or not being used. | Make clear whether the final release treats this as a default, confirmatory band, or fallback only. |

## EMG and Gamma Quality Control

| Area | Current setting | Why it is used | Publication note |
|:-----|:----------------|:---------------|:-----------------|
| EMG channel strategy | temporalis bipolar pairs plus posterior neck channels | Gives direct physiological access to likely muscle contamination rather than inferring it only indirectly from scalp EEG. | One of the strongest design features in the pipeline. |
| EMG summary metric | PCA on derived EMG signals, retain PC1 | Produces a compact global muscle-tension covariate for later QC. | Defensible and interpretable. |
| Gamma contamination criterion | exclude blocks where EMG explains more than `25%` of gamma variance (`R^2 > 0.25`) | Makes the gamma story more credible by requiring a practical separation between neural and myogenic signal. | This threshold should stay clearly tied to the repo's conservative gamma policy. |

## QC Thresholds

These thresholds are intended to keep the pipeline automated while still making poor-quality blocks visible:

| Metric | Current value | Rationale | Publication note |
|:-------|:--------------|:----------|:-----------------|
| `qc.max_bad_channels_pct` | `10` | Prevents heavily compromised blocks from being treated as routine. | Reasonable operational threshold. |
| `qc.max_ica_components_rejected` | `8` | Flags unusually artifact-dominated decompositions. | Better as a QC warning than a hard scientific rule. |
| `qc.min_brain_ics_remaining` | `15` | Tries to avoid over-pruning the ICA solution. | Good sanity-check threshold. |
| `qc.max_asr_modified_pct` | `65` | Flags very heavily reconstructed blocks for review without treating moderate ASR intervention as an automatic failure. | Make sure the manuscript distinguishes QC flagging from automatic exclusion. |
| `qc.max_epoch_rejection_pct` | `30` | Signals when too much trial data has been lost. | Sensible automation threshold. |
| `qc.min_trials_per_block` | `50` | Ensures a minimum amount of data survives for downstream analyses. | Should stay justified in terms of stability and interpretability, not just convenience. |
| `qc.p3b_latency_range_ms` | `250-600` | Broad plausibility window for posterior ERP peaks. | Sensible QC bound. |
| `qc.p3b_min_amplitude_uv` | `0.5` | Avoids treating near-flat peaks as meaningful P3b responses. | More of a sanity check than a strict inferential rule. |

## Determinism Choices

The pipeline also makes deliberate engineering choices that support reproducibility:

- `random_state = 42` across stochastic steps where possible
- `PYTHONHASHSEED = 0`
- single-thread BLAS / LAPACK / OpenMP execution
- per-step QC JSON logs with hashes
- run-level provenance manifests

These are not scientific parameters in the same sense as a theta band or ICA threshold, but they are still part of the method. They make reruns more defensible and debugging much easier.

## What Needs To Stay Aligned Before Publication

Before a public release, keep these surfaces saying the same thing:

- manuscript methods
- `PARAMETERS.md`
- `eeg_pipeline/config/parameters.json`
- script-level constants that still have not been moved into config
- README statements about what is automated and what is ancillary

If those diverge, the repo becomes harder to defend even when the underlying parameter choices are reasonable.
