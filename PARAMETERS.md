# Parameters and Method Rationale

This file is the repo-facing summary of why the main pipeline parameters were chosen.

The manuscript should carry the full literature narrative and formal methods text. This file has a different job: it lets a collaborator, reviewer, or future maintainer understand the live parameter choices without having to reverse-engineer them from the code.

## Where Parameters Currently Live

- `eeg_pipeline/config/parameters.json`: main preprocessing settings and many analysis settings
- `eeg_pipeline/config/study.yml`: study paths, block structure, montage, and node definitions
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
| Bandpass filtering | shared preprocessing stream at `1-100 Hz` (`1 Hz` high-pass) | Keeps the main automated path simple: ASR and ICA both operate on the same incoming continuous stream rather than creating internal fit-only copies. | Keep manuscript and code aligned on the exact filter design and the fact that this is one shared preprocessing path. |
| ERP-only branch | separate `0.1-100 Hz` branch for `p3b_erp` | Preserves a conservative low-cut path for P3b estimation without changing the live oscillatory path used by ASR, ICA, PAC, theta power, theta wPLI, gamma, or specparam outputs. | This distinction should be explicit anywhere the methods describe P3b versus oscillatory analyses. |
| Line-noise handling | deterministic FIR notch at `50 Hz` and `100 Hz` | Removes mains contamination without introducing another decomposition-heavy preprocessing step. | Keep the manuscript aligned with the simplified public path rather than older internal variants. |
| EEG reference | simple average reference after interpolating only known bad EEG channels | Keeps the reference step explicit and auditable while avoiding fresh data-driven bad-channel branching during production runs. | Subject-specific known bad EEG channels should remain locked in participant config. |
| Subject-specific bad channels | `participant_configs/*.json` with `known_bad_eeg` and `known_bad_emg` | Makes known hardware or contact failures explicit and reproducible. | This is a strength of the repo and should be preserved. |
| ASR burst cleaning | `cutoff = 30`, `method = euclid`, `win_len = 0.5`, `use_clean_windows = true` on the shared `1 Hz` high-pass stream | This is intentionally conservative. A higher cutoff than the often-cited default of 20 means the step targets only more extreme bursts and is less likely to flatten task-related signal. ASR runs directly on the same shared preprocessing stream that comes out of the import/filter step. | The repo currently flags blocks at a relatively permissive modification threshold (`qc.max_asr_modified_pct = 65`) so heavily reconstructed blocks can be reviewed rather than silently treated as routine. |
| ICA decomposition | `n_components = 25`, `method = infomax`, `fit_params.extended = true`, `random_state = 42` on the shared `1 Hz` high-pass stream | ICA fits directly on the same shared `1 Hz` high-pass stream used by ASR rather than making an extra fit-only high-pass copy. The repo uses Extended Infomax explicitly for reproducibility, and `25` components is a conservative dimensionality choice for a PAC-focused cleaning stage: enough to isolate dominant artefact sources without over-fragmenting the decomposition. | This rationale should stay explicit in repo docs and manuscript rather than leaving `25` to look arbitrary. |
| ICLabel thresholds | eye `0.85`, heart `0.85`, muscle `0.90` | These are fairly conservative automatic rejection thresholds intended to remove obvious artifacts without over-pruning the decomposition. | Good candidate for a brief citation-backed note in a methods appendix or release note. |
| Epoching for P3b | `-0.2 to 0.8 s`, baseline `-0.2 to 0.0 s` on the dedicated `p3b_erp` branch | Standard stimulus-locked ERP windowing with pre-stimulus baseline correction, while keeping the main oscillatory path unchanged. | P3b logic should stay aligned with the actual trigger structure used in the task. |
| Epoching for PAC / connectivity | `-0.5 to 1.8 s`, no baseline | Gives enough temporal context for low-frequency phase estimation and edge trimming in time-frequency analyses. | This broader window is methodologically sensible for oscillatory work. |
| Autoreject search grid | `n_interpolate = [1, 4, 8, 16]`, `consensus = [0.1, 0.5, 1.0]`, `cv = 10` | Standard automated repair / rejection tradeoff without hand-tuning each subject. Here `cv` is the AutoReject cross-validation fold count only, not an ICLabel or ICA parameter. The repo now uses the upstream library default of `10`. | Reasonable default grid; useful to keep documented rather than implicit. |

## Main Analysis Choices

| Area | Current setting | Why it is used | Publication note |
|:-----|:----------------|:---------------|:-----------------|
| Confirmatory PAC nodes | `C_broad_F` and `C_broad_P` bilateral composites | Broad bilateral nodes are a sensible choice for 64-channel EEG where precise source separation is limited. They support a cleaner confirmatory frontoparietal story than many tiny ad hoc ROIs. | The node definitions should stay identical across README, manuscript, and scripts. |
| P3b channel set | config currently lists `Pz`, `P1`, `P2`, `POz` on the dedicated ERP branch | Keeps the analysis anchored to a posterior P3b region while still allowing a canonical `Pz` interpretation, without changing the main oscillatory outputs. | If the manuscript presents P3b as strictly `Pz`, keep that wording aligned with the code output. |
| P3b time window | `0.3-0.5 s` | Conventional post-stimulus window for P3b-like activity in this task family. | Stronger if tied explicitly to task timing and literature. |
| PAC phase band | `4-8 Hz` theta | Canonical working-memory theta range, suitable for confirmatory PAC hypotheses. | Defend as a canonical confirmatory choice rather than an individualized one. |
| PAC amplitude band | `55-85 Hz` gamma | Narrow enough to avoid lower-frequency bleed while still targeting task-relevant high gamma. | Important to keep coupled with the EMG-control story. |
| PAC null model | `500` surrogates | Enough to generate a stable z-scored modulation index without making runtime unreasonable. | Reasonable compromise between speed and null stability. |
| PAC edge trimming | `trim = 0.1 s` | Reduces edge artifacts in filtered analytic signals. | Sensible implementation detail worth keeping explicit. |
| Theta synchrony | confirmatory frontoparietal theta wPLI in the stimulus window | Fits the hypothesis that fatigue alters communication across a control-relevant network rather than only local amplitude. | The confirmatory window should be documented consistently in script and manuscript. |
| Spectral parameterization | `freq_range = 3-30`, `peak_width_limits = [1, 6]`, `max_n_peaks = 6`, `zero_pad_factor = 4` | Separates periodic peaks from the aperiodic background and improves frequency resolution for short stimulus-locked epochs. In this repo it is a descriptive follow-up, not a required preprocessing dependency for PAC. | Zero padding should be described carefully as improving frequency sampling, not creating new information. |
| Theta power band | fixed theta `4-8` for FM-theta power | Keeps the confirmatory theta-power measure aligned with the same canonical theta regime used elsewhere, without individualising the analysis band. | This should be stated plainly as the live study position. |
| Theta peak summaries | stimulus-period frontal-midline theta peak summaries from the descriptive specparam workflow | Provides descriptive frequency summaries without feeding back into the confirmatory theta-power or PAC band definitions. | Keep this distinction explicit in README and manuscript. |

Method hierarchy to keep explicit:

- The shared `1 Hz` high-pass path is the live oscillatory pipeline for ASR, ICA, PAC, theta power, theta wPLI, gamma, and descriptive specparam outputs.
- A separate `0.1 Hz` ERP branch exists only for conservative P3b estimation.
- PAC phase band is fixed `4-8 Hz`.
- Frontal midline theta power is fixed-band `4-8 Hz`.
- Specparam outputs and frontal-midline theta peak summaries are descriptive only.

## EMG and Gamma Quality Control

| Area | Current setting | Why it is used | Publication note |
|:-----|:----------------|:---------------|:-----------------|
| EMG channel strategy | temporalis bipolar pairs plus posterior neck channels | Gives direct physiological access to likely muscle contamination rather than inferring it only indirectly from scalp EEG. | One of the strongest design features in the pipeline. |
| EMG summary metric | PCA on derived EMG signals, retain PC1 | Produces a compact global muscle-tension covariate for later QC. | Defensible and interpretable. |
| Gamma contamination criterion | exclude blocks where EMG explains more than `25%` of gamma variance (`R^2 > 0.25`) | Makes the gamma story more credible by requiring a practical separation between neural and myogenic signal. | This threshold should stay clearly tied to the repo's conservative gamma policy. |

`python eeg_pipeline/run_pipeline.py --mode full` stops at step 13. That means final PAC and gamma inclusion decisions still require manually running the EMG follow-up steps (`16_emg_pca_covariates.py` and `17_emg_gamma_regression.py`) before treating those outputs as inferentially final.

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
