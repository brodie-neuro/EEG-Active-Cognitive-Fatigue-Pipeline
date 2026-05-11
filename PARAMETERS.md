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
4. separate confirmatory analyses from QC and sensitivity follow-up measures

## Sample Size and Stopping Rule

Data collection follows a **Sequential Bayes Factor Design** (Schönbrodt et al., 2017) driven exclusively by the primary hypothesis (H1: Theta-Gamma PAC). 

To account for Bayes Factor volatility at small sample sizes and protect against premature stopping, we pre-established a minimum sample size of **$N_{min} = 20$**. Data collection is planned to terminate when the Bayes Factor for H1 exceeds $BF_{10} > 5$ (Moderate-to-Strong evidence), up to a resource-constrained maximum of **$N_{max} = 30$**. 

### Rationale for H1 vs. H2/H3
- **H1 (Theta-Gamma PAC):** Serves as the sole primary sample-size-justifying endpoint. This is theoretically grounded in our working memory account of active cognitive fatigue (Mangan & Kourtis, 2026, *Journal of Cognitive Neuroscience*), and supported by adjacent empirical evidence demonstrating theta-gamma network alterations during fatigue-inducing states in sleep deprivation and clinical populations.
- **H2 (Alpha-Gamma PAC) & H3 (P3b):** These are strictly secondary. H2 is theoretically motivated by Miller et al.'s (2018) Working Memory 2.0 framework, but lacks the adjacent empirical scalp-EEG evidence seen for theta-gamma. Therefore, an *a priori* sample-size stopping rule for H2 is not appropriate. H2 and H3 are evaluated opportunistically at the final sample size dictated by the H1 stopping rule.


## Core Preprocessing Choices

| Area | Current setting | Why it is used | Publication note |
|:-----|:----------------|:---------------|:-----------------|
| Bandpass filtering | shared preprocessing stream at `1-100 Hz` (`1 Hz` high-pass) | Keeps the main automated path simple: ASR and ICA both operate on the same incoming continuous stream rather than creating internal fit-only copies. | Keep manuscript and code aligned on the exact filter design and the fact that this is one shared preprocessing path. |
| ERP-only branch | separate `0.1-100 Hz` branch for `p3b_erp` | Preserves a conservative low-cut path for P3b estimation without changing the live oscillatory path used by ASR, ICA, PAC, or alpha-gamma PAC outputs. | This distinction should be explicit anywhere the methods describe P3b versus oscillatory analyses. |
| Line-noise handling | deterministic FIR notch at `50 Hz` and `100 Hz` | Removes mains contamination without introducing another decomposition-heavy preprocessing step. | Keep the manuscript aligned with the simplified public path rather than older internal variants. |
| EEG reference | simple average reference after interpolating import/QC-flagged bad EEG channels plus any subject-specific known bad EEG channels | Keeps the reference step explicit and auditable while making bad-channel handling reproducible through QC logs and participant config. | Manuscript wording should describe both automatic QC flags and locked subject-specific bad-channel overrides. |
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
| Alpha-gamma PAC | secondary theoretically motivated H2 between-region alpha(8-13 Hz)-gamma(55-85 Hz) PAC in stimulus window | C_broad_F->C_broad_P, mirroring the frontoparietal topology of H1 theta-gamma PAC but indexing alpha-mediated executive gating and prioritisation (Miller et al., 2018). Same MI + surrogate method as H1. | Keep labelled as secondary H2 rather than as the primary sample-size-justifying endpoint. Report effect estimates, intervals, p-values, and Bayes factors without treating a null result as definitive absence of effect. |

| PAC phase-band QC | `pac_phase_qc` defaults to theta `4-8 Hz` and alpha `8-13 Hz` on `C_broad_F`, using cleaned PAC epochs and the `0.0-0.6 s` PAC analysis window. Welch uses 2 s Hann windows with 50% overlap and `nfft_factor = 4` zero-padding. Aperiodic removal requires specparam with a `2-20 Hz` fit range. | QC-only check that the phase-providing bands used by theta-gamma and alpha-gamma PAC show plausible residual spectral support above the aperiodic background. Centre of mass is computed only from positive residual spectral mass. Zero-padding improves the plotted/readout frequency grid but does not increase true spectral resolution. | Report as interpretability/QC support, not as a new primary power analysis. Do not treat missing residual support as a forced peak estimate. |
| P3b visual QC | `p3b_qc` uses the configured `p3b_erp` branch, P3b ROI, and `0.3-0.5 s` measurement window | Produces participant dashboards with ROI waveform, single-trial images, channel traces, and topography so P3b plausibility can be reviewed quickly. | Keep descriptive; this does not change the P3b feature extraction method. |

Method hierarchy to keep explicit:

- The shared `1 Hz` high-pass path is the live oscillatory pipeline for ASR, ICA, theta-gamma PAC, and alpha-gamma PAC.
- A separate `0.1 Hz` ERP branch exists only for conservative P3b estimation.
- Primary H1 theta-gamma PAC phase band is fixed `4-8 Hz` (step 08).
- Secondary H2 alpha-gamma PAC phase band is fixed `8-13 Hz` (step 09).
- Steps 09 and 09b both read the H2 settings from `alpha_gamma_pac` in `parameters.json`, so the alpha-gamma EMG sensitivity script differs from theta-gamma 08b only by the configured phase band and output target.
- `alpha_gamma_pac.analysis_excluded_subjects` defines the QC-based H2 exclusion set used by alpha-gamma inference and alpha-gamma EMG sensitivity diagnostics. Raw feature extraction can still compute all rows for auditability.
- Secondary H3 P3b is estimated only from the dedicated `p3b_erp` branch (step 10).

## EMG and Gamma Quality Control

| Area | Current setting | Why it is used | Publication note |
|:-----|:----------------|:---------------|:-----------------|
| EMG channel strategy | temporalis bipolar pairs plus posterior neck channels | Gives direct physiological access to likely muscle contamination rather than inferring it only indirectly from scalp EEG. | Important design feature in the pipeline. |
| EMG summary metric | PCA on derived EMG signals, retain PC1 | Produces a compact global muscle-tension covariate for later QC. | Defensible and interpretable. |
| PAC gamma-amplitude contamination handling | no automatic threshold-based EMG exclusion is applied to confirmatory PAC outputs | Keeps EMG handling as layered sensitivity evidence rather than a hard inclusion gate. | Manuscript wording should describe EMG analyses as sensitivity/diagnostic controls unless a thresholded exclusion rule is explicitly reintroduced. |

`python eeg_pipeline/run_pipeline.py --mode full` runs the main preprocessing, confirmatory analyses, and integrated QC workflows. EMG follow-up steps remain manual sensitivity diagnostics: `13_emg_pca_covariates.py`, then `08b_pac_emg_corrected.py` / `09b_alpha_gamma_pac_emg_corrected.py`, then `14_emg_pac_correlation.py` / `15_emg_alpha_gamma_correlation.py`.

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
| `pac_phase_qc.classification` | configurable residual-support thresholds | Separates clear, weak, and indeterminate phase-band support without Gaussian peak fitting. | Thresholds are descriptive QC criteria, not inferential exclusions. |
| `p3b_qc.classification` | configurable mean-amplitude and missing-channel flags | Makes participant-level P3b dashboards easy to triage. | Status labels should guide review, not automatically rewrite confirmatory results. |

## Determinism Choices

The pipeline also makes deliberate engineering choices that support reproducibility:

- `random_state = 42` across stochastic steps where possible
- `PYTHONHASHSEED = 0`
- single-thread BLAS / LAPACK / OpenMP execution
- per-step QC JSON logs with hashes
- run-level provenance manifests
