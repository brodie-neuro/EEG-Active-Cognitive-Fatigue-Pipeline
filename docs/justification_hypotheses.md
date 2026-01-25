# Hypothesis Justification Document
## EEG Study: Active Cognitive Fatigue and Working Memory

**Target Journal:** Nature Human Behaviour  
**Theoretical Foundation:** Mangan & Kourtis (JCN) - "The Missing Link: Bridging Cognitive Fatigue with Working Memory"

---

## Core Theory

Active cognitive fatigue degrades working memory performance through disruption of theta-gamma cross-frequency coupling (CFC) — the neural mechanism that coordinates information maintenance (Lisman & Jensen, 2013; Miller et al., 2018).

---

## Primary Hypotheses

### H1: Effort-Capacity Dissociation

**Prediction:** Theta power increases (compensatory effort) while P3b amplitude decreases (capacity depletion) under fatigue.

| Marker | Expected Change | Mechanism |
|:-------|:----------------|:----------|
| θ Power (Fz) | ↑ Increase | ACC/MFC compensatory effort |
| P3b (Pz) | ↓ Decrease | Reduced attentional resource allocation |

**Test:** Δθ × ΔP3b interaction predicting Δd′; interaction should be **negative** (when both move in opposite directions, performance suffers most).

**Justification:** Distinguishes "trying harder" from "system failing" — central to the JCN theory.

---

### H2: Cross-Frequency Coupling Breakdown

**Prediction:** Theta-gamma PAC decreases with fatigue, reflecting breakdown of the neural code for WM maintenance.

| Measure | Expected Change | Mechanism |
|:--------|:----------------|:----------|
| ΔPAC_cross (RF→RP) | ↓ Decrease | Frontal theta no longer coordinating parietal gamma |

**Test:** ΔPAC_cross(RF→RP) predicts Δd′ beyond θ and P3b alone.

**Justification:** Right frontoparietal pathway is primary for visuospatial WM (D'Esposito & Postle, 2015). This tests whether the theta-gamma code breaks down, as proposed in JCN.

---

## Secondary Hypotheses

### H3: Processing Speed Slowing

**Prediction:** Individual alpha frequency (IAF) decreases with fatigue.

| Measure | Expected Change | Interpretation |
|:--------|:----------------|:---------------|
| ΔIAF | ↓ Decrease | Global cortical processing speed slowing |

**Test:** ΔIAF predicts Δd′.

**Justification:** IAF is a stable trait marker of processing speed (Klimesch, 1999). Slowing indicates fatigue-induced cortical state change.

---

### H4: Theta Frequency Mistuning

**Prediction:** Task-related theta frequency (fθ) slows, indicating the theta scaffold is no longer optimally tuned.

| Measure | Expected Change | Interpretation |
|:--------|:----------------|:---------------|
| Δfθ | ↓ Decrease | Theta rhythm mistuned for task demands |

**Test:** Δfθ predicts Δd′ beyond θ power.

**Justification:** Novel measure — extends the "theta paradox" (θ power up, but θ frequency slowing suggests coordination impairment).

---

### H5: Network-Level Coordination

**Prediction:** PAC changes are coordinated across cortical regions (not isolated).

| Measure | Expected Outcome | Interpretation |
|:--------|:-----------------|:---------------|
| Louvain communities | Right frontoparietal cluster | Visuospatial nodes fatigue together |

**Test:** Community mean ΔPAC predicts Δd′ better than individual nodes.

**Justification:** Tests network-level vs local breakdown — if fatigue affects distributed networks, community approach captures this.

---

## Exploratory Analyses

| Measure | Rationale |
|:--------|:----------|
| Low gamma power (35-48 Hz) | Attention/local processing; may decline with P3b |
| PLV_θ (RF-RP) | Same-frequency theta synchrony; complements cross-freq |
| Individual nodal ΔPAC (9 nodes) | Spatial specificity of breakdown |

---

## Model Comparison Strategy

| Model | Predictors | Purpose |
|:------|:-----------|:--------|
| **M1** | d′_Block1 + Δθ + ΔP3b + Δθ×ΔP3b | Effort-capacity (H1) |
| **M2** | M1 + ΔPAC_cross(RF→RP) | + Cross-region PAC (H2) |
| **M3** | M1 + ΔPAC_community | + Network PAC (H5) |
| **M4** | M1 + ΔIAF + Δfθ | + Frequency measures (H3, H4) |
| **M5** | Best combination from above | Final parsimonious model |

**Comparison:** ΔR², AIC, BIC, cross-validation

---

## Justification for Each Analysis Element

### Why Nodal PAC (Step 13)?
- **Problem:** Single electrodes are noisy; scalp EEG has poor spatial resolution.
- **Solution:** Aggregate electrodes into 9 anatomically-defined nodes; use trimmed mean to reject outlier electrodes.
- **Benefit:** Robust regional estimates comparable across subjects.

### Why Z-Score PAC?
- **Problem:** Raw PAC is biased by 1/f noise and electrode impedance.
- **Solution:** Surrogate-based Z-scoring; compare observed PAC to shuffled distribution.
- **Benefit:** Standardised values; can compare frontal vs parietal, subjects vs subjects.

### Why Cross-Region PAC (RF→RP)?
- **Problem:** Local PAC tests coupling within a region; doesn't capture inter-regional coordination.
- **Solution:** Compute PAC with frontal theta phase → parietal gamma amplitude.
- **Benefit:** Tests top-down control hypothesis — frontal regions coordinating posterior processing.

### Why RF→RP specifically?
- **Problem:** Visuospatial WM is right-lateralised.
- **Solution:** Prioritise right frontoparietal pathway.
- **Benefit:** Hypothesis-driven, not fishing.

### Why Louvain Community Detection?
- **Problem:** 9 nodes = 9 predictors (overfitting risk).
- **Solution:** Data-driven grouping of co-varying nodes.
- **Benefit:** Parsimonious predictor (community mean); tests network-level hypothesis.

### Why Θ × P3b Interaction?
- **Problem:** Theta power alone is ambiguous ("theta paradox" — effort or sleep pressure?).
- **Solution:** Combine with P3b to disambiguate.
- **Benefit:** Effort (θ↑) + capacity intact (P3b stable) = resilient. Effort (θ↑) + capacity depleted (P3b↓) = fatigue breakdown.

---

## Summary: What This Study Tests

| Gap from JCN | This Study Addresses |
|:-------------|:---------------------|
| No PAC studies under active fatigue with WM task | ✓ WAND paradigm + theta-gamma PAC |
| No separation of low vs high gamma | ✓ 35-48 Hz vs 55-85 Hz |
| No network-level PAC analysis | ✓ Nodal aggregation + Louvain |
| No link between CFC and performance | ✓ ΔPAC predicting Δd′ |
| Theta paradox unresolved | ✓ Θ × P3b interaction + fθ |
