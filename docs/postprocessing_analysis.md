# EEG Post-Processing & Analysis Plan

---

## Step 1: ERP Analysis (P3b)

**Rationale:** The P3b is a classic, well-understood brain signal. Fatigue reduces its amplitude. Including this analysis acts as a familiar and robust "anchor" for our more novel PAC findings, making our story stronger for reviewers and readers.

**Method:**
- Focus on target trials at Pz or CPz electrode
- Use standard pre-stimulus baseline
- Measure P3b mean amplitude (300–500 ms) and peak latency for each block
- Calculate change scores: **ΔP3b_amp** and **ΔP3b_lat**

**Use in Model:**
1. Run t-tests to confirm expected decrease in P3b amplitude with fatigue
2. Add ΔP3b_amp as predictor in main regression model

---

## Step 2: Individual Alpha Frequency (IAF)

**Rationale:** Peak alpha frequency is a stable, trait-like marker of general cortical processing speed. IAF slowing is a well-established sign of slower cognitive processing—a global measure of fatigue's effect on brain state.

**Method:**
- Calculate IAF from posterior electrode Oz during resting-state periods (beginning and end of experiment)
- Calculate change score: **ΔIAF = IAF_Block5 − IAF_Block1**

**Use in Model:**
1. Test for group-level slowing with t-test
2. Add ΔIAF to regression model to see if global "cortical slowing" predicts performance decline

---

## Step 3: Task-Related Theta Frequency

**Rationale:** Frontal theta rhythm speed during memory maintenance acts as scaffolding. Subtle slowing of task-related frequency could be a direct neural marker of fatigue (per theta paradox paper).

**Method:**
- Focus on Central Frontal node during 1-second delay window
- Calculate instantaneous theta frequency for each trial
- Final value per block = median of trial-by-trial frequencies
- Gives us **fθ_Block1** and **fθ_Block5**

**Use in Model:**
1. T-test for theta frequency slowing
2. Add change score (**Δfθ**) to regression model

---

## Step 3B: Low Gamma Power (35–48 Hz)

**Rationale:** Low gamma (35–48 Hz) reflects local cortical processing and attentional focus. Unlike high gamma used in PAC, low gamma power indexes sensory binding and early perceptual attention. In fatigue states, we hypothesize **reduced low gamma power** over parietal-occipital regions, indicating impaired sensory integration.

**Hypothesis:**
- Low gamma power will **decrease** with fatigue (unlike theta, which increases with compensatory effort)
- This decrease will correlate with **P3b reduction** (both reflect reduced capacity)
- Low gamma and theta moving in **opposite directions** supports the effort-capacity dissociation

**Method:**

| Parameter | Specification |
|:----------|:--------------|
| **Frequency Band** | 35–48 Hz (below 50 Hz line noise) |
| **Primary Sites** | Parietal-occipital: Pz, POz, O1, O2 (sensory processing) |
| **Secondary Sites** | Frontal: Fz, FCz (attentional control) |
| **Analysis Window** | Same delay period as PAC (800–1800 ms) |
| **Measure** | Absolute and relative power (normalized to total power) |

**Use in Model:**
1. T-test for power decrease with fatigue
2. Correlate Δγ_low with ΔP3b (expected: positive correlation)
3. Add **Δγ_low** as independent predictor alongside high gamma PAC

---

## Step 4: Nodal PAC Signal

### 4.1 Get Clean PAC Signal

| Parameter | Specification |
|:----------|:--------------|
| **Nodes** | 9-node structure: LF, CF, RF, LC, C, RC, LP, CP, RP |
| **Analysis Window** | 800–1800 ms post-stimulus (delay period, avoids motor response at ~600 ms) |
| **Frequency Bands** | Theta-Gamma PAC: **Gamma 55–85 Hz** (avoids task-related beta confound) |
| **Line Noise** | 50 Hz notch filter applied during preprocessing |
| **PAC Normalisation** | Z-score against surrogate data per trial |
| **Signal Aggregation** | 10% trimmed mean of z-scores from electrodes within each node |

#### Why 10% Trimmed Mean?

**Problem:** Fatigue increases jaw/neck muscle tension → EMG artefacts → inflated gamma power → spurious PAC.

**Why not regular mean?** One contaminated electrode inflates the entire node average.

**Why not PCA?** Muscle artefacts have high variance, so they dominate PC1 in gamma band.

**Solution:** 10% trimmed mean drops the highest and lowest 10% of electrode values, then averages the rest. This is **robust to outliers** without requiring manual rejection.

> **Example:** If a node has 10 electrodes and one has muscle contamination (PAC = 5.2 when others are 0.3–0.9), the trimmed mean ignores it automatically.

### 4.2 Calculate Change Score

For each participant and each of 9 nodes:

**ΔPAC = PAC_Block5 − PAC_Block1**

---

### 4.3 Frontal-Parietal PAC Connectivity

**Rationale:** Working memory critically depends on communication between frontal (executive control) and parietal (storage/attention) regions. In fatigue states, this long-range coordination may break down before local PAC changes are detectable. This analysis tests whether **between-region** coupling predicts performance decline.

**Method:**

| Parameter | Specification |
|:----------|:--------------|
| **Region Pairs** | CF↔CP (Central Frontal ↔ Central Parietal), LF↔LP, RF↔RP |
| **Measure** | Phase-Locking Value (PLV) or coherence at theta (4–7 Hz) |
| **Cross-frequency** | Frontal theta phase → Parietal gamma amplitude |
| **Analysis Window** | Same delay period (800–1800 ms post-stimulus) |

**Region Pairs (using same 9-node structure):**

| Pair | Justification | Priority |
|:-----|:--------------|:---------|
| **RF → RP** | Visuospatial WM is right-lateralised | **Primary** |
| CF → CP | General executive control | Secondary |
| LF → LP | Verbal/phonological (lower priority for visual task) | Exploratory |

**Connectivity Metrics:**

1. **ΔPLV_θ (RF-RP):** Change in theta-band phase synchrony (right pathway)
2. **ΔPAC_cross (RF→RP):** Change in right frontal theta phase driving right parietal gamma amplitude

**Use in Model:**
- Add ΔPAC_cross (RF→RP) as predictor alongside nodal ΔPAC
- Test whether **cross-region breakdown** predicts performance decline above local nodal changes

---

## Step 5: Describe Group Network

Before predictive modelling, create descriptive picture of group-level changes:

1. Create 9×9 correlation matrix from ΔPAC values (which nodes change together)
2. Run Louvain community detection on matrix
3. Generate network graph showing main communities of change

### Louvain Community Detection — Rationale

**What it tests:** Which brain regions "fatigue together"?

**Expected outcome for visuospatial WM:**
- Right fronto-parietal nodes (RF, RC, RP) should form a community
- This community should show **coordinated decline** under fatigue
- Left hemisphere (verbal areas) may remain independent or unchanged

### Using Community Membership in Predictor Model

After Louvain identifies communities:

| Metric | How to Calculate | Use in Model |
|:-------|:-----------------|:-------------|
| **Community mean ΔPAC** | Average ΔPAC within detected community | Predictor |
| **Within-community strength** | Mean correlation of edges within community | Predictor |

**Model comparison:**
- Nodal model: 9 individual ΔPAC predictors
- Community model: Community means as predictors
- Test whether **community coordination** predicts Δd′ above individual nodes

---

## Step 6: Predictive Model for Performance Decline

Multiple regression model predicting individual **Δd′** from individual **ΔPAC** values.

### Model Structure

| Component | Description |
|:----------|:------------|
| **Covariates** | d′ from Block 1 (controls for starting ability) |
| **Nodal Model** | 9 ΔPAC nodal predictors |
| **Interaction Model** | Add interaction term (e.g., ΔPAC_RF × ΔPAC_RP) |
| **Centring** | Mean-centre predictors before creating interaction terms |
| **Validation** | Adjusted R² + k-fold cross-validation |

---

## Step 7: Integrated Modelling — Hierarchical Model Comparison

**Rationale:** Test whether neural markers predict performance decline, and compare which level of analysis provides the best prediction.

### Model Comparison (Hierarchical)

| Model | Predictors | Tests |
|:------|:-----------|:------|
| **M1: Base** | d′_Block1 + Δθ + ΔP3b + Δθ×ΔP3b | Effort-capacity dissociation |
| **M2: + Local PAC** | M1 + ΔPAC_RF, ΔPAC_RP, ΔPAC_CF, ΔPAC_CP | Does local nodal PAC add explained variance? |
| **M3: + Community** | M1 + ΔPAC_community (Louvain mean) | Does network-level PAC add variance over individual nodes? |
| **M4: + Cross-region** | M1 + ΔPAC_cross(RF→RP) | Does frontal→parietal coordination add variance? |
| **M5: Full** | M1 + ΔPAC_community + ΔPAC_cross(RF→RP) | Best combined model |

**Model comparison:** ΔR², AIC, BIC between nested models

### Parsimonious Primary Hypothesis Test

**Core model (theory-driven):**

| Predictor | Rationale |
|:----------|:----------|
| **ΔθPower** | Compensatory effort |
| **ΔP3b** | Resource allocation capacity |
| **Δθ × ΔP3b** | Effort-capacity dissociation |
| **ΔPAC_cross(RF→RP)** | Right frontoparietal coordination |

This tests the core theory: fatigue = effort up, capacity down, coordination broken.

### Interpretation of Patterns

| Pattern | Indicators | Interpretation |
|:--------|:-----------|:---------------|
| **Sustained Effort** | Δθ↑, ΔP3b stable, ΔPAC stable | Resilient system |
| **Fatigue Breakdown** | Δθ↑, ΔP3b↓, ΔPAC_cross↓ | System failing despite effort |
| **Disengagement** | Δθ↓, ΔP3b↓ | Gave up |
