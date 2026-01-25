# Hypothesis Justification Document
## Does Cognitive Fatigue Disrupt the Theta-Gamma Neural Code?

**Target Journal:** Nature Human Behaviour  
**Theoretical Foundation:** Mangan & Kourtis (JCN) - "The Missing Link: Bridging Cognitive Fatigue with Working Memory"

---

## Central Question

Active cognitive fatigue degrades working memory performance. We propose this occurs through disruption of cross-frequency coupling — the brain's mechanism for coordinating information maintenance.

---

## Primary Hypothesis

### H1: Fatigue Disrupts Frontoparietal Cross-Frequency Coupling

**In plain words:** Under fatigue, the frontal brain regions lose their ability to coordinate activity in parietal regions. Specifically, the timing of slow frontal rhythms no longer organises the fast bursts of activity in parietal cortex that represent memory items.

**What we measure:** Cross-frequency coupling between right frontal and right parietal regions (RF→RP).

**What we expect:** Coupling strength decreases from Block 1 to Block 5, and this decrease predicts performance decline.

**Why right hemisphere?** The task uses visual-spatial stimuli, which preferentially engage right-lateralised brain networks (D'Esposito & Postle, 2015).

**Core justification:** The theta-gamma neural code (Lisman & Jensen, 2013) is fundamental to working memory — theta provides the temporal scaffold, gamma represents individual items. If fatigue disrupts this coordination, working memory breaks down.

---

## Secondary Hypothesis

### H2: Theta Rhythm Becomes Mistuned Under Fatigue

**In plain words:** The speed of the brain's "scaffolding rhythm" (theta) slows down or becomes irregular with fatigue. When this rhythm is no longer at the right speed, information can't be properly organised.

**What we measure:** Individual theta frequency during the memory delay period.

**What we expect:** Theta frequency decreases from Block 1 to Block 5.

**Core justification:** Theta power increases with cognitive demand (the "theta paradox"). But power alone doesn't tell us if the rhythm is functioning well. Frequency slowing suggests the scaffolding is degraded, not just working harder.

---

## Exploratory Analyses

### Network-Level Patterns (Louvain Communities)

**In plain words:** Rather than analysing each brain region separately, we ask: do regions that serve similar functions fatigue together as coordinated networks?

**What we measure:** Correlation patterns across 9 brain regions; identify clusters that change together.

**What we expect:** Right frontoparietal regions (RF, RC, RP) form a cluster that declines together.

**Core justification:** If fatigue affects distributed networks rather than isolated spots, this approach captures coordinated breakdown.

---

### Individual Alpha Frequency (IAF)

**In plain words:** The brain's "idling rhythm" speed is a trait marker of cortical processing speed. Does it slow down with fatigue?

**What we measure:** Peak alpha frequency at posterior sites.

**What we expect:** IAF decreases from start to end of task.

**Core justification:** Well-established marker (Klimesch, 1999). Provides a global measure of cortical state change.

---

## On P3b: To Include or Not?

### The Issue
P3b amplitude decreases with both:
1. **Higher working memory load** (e.g., 2-back vs 3-back)
2. **Time-on-task/fatigue** (at constant load)

Hopstaken et al. (2015) showed P3b decreases across all N-back levels (1, 2, 3-back) with time-on-task, using multilevel analysis. The effect was present regardless of difficulty level.

### The Confound in Your Study
Your participants are at different N-back levels (some at 2-back, some at 3-back due to adaptive difficulty). This means P3b differences could reflect:
- Fatigue (what you want)
- Different cognitive demands between levels (confound)

### Recommendation
**Option A: Exclude P3b from primary analysis.** Focus on cross-frequency coupling which is the novel contribution. P3b is well-trodden ground.

**Option B: Include P3b but control for N-back level.** Add level as covariate. Risk: reduced power.

**Option C: Analyse only participants at same level.** Subset analysis. Risk: very small N.

**For Nature portfolio:** Parsimony favours Option A. Cross-frequency coupling is the novel story. P3b can be supplementary.

---

## Model Comparison Strategy

### In Plain Words

We build a series of models, each adding one new idea. We ask: does each new idea help predict who shows the biggest performance decline?

| Model | What It Includes | Plain English Question |
|:------|:-----------------|:-----------------------|
| **M0** | Starting performance | Baseline ability only |
| **M1** | + Frontal effort (theta power) | Does trying harder predict decline? |
| **M2** | + Scaffold speed (theta frequency) | Does rhythm mistuning predict decline? |
| **M3** | + Frontoparietal coordination (cross-region coupling) | Does coordination breakdown predict decline? |

### How We Compare
- **Change in R²:** How much more variance explained?
- **Information criteria (AIC, BIC):** Is the improvement worth the complexity?
- **Cross-validation:** Does it hold up on new data?

### What Goes Where
- **Introduction:** State hypotheses (H1, H2) in plain terms
- **Methods:** Full model specifications
- **Results:** Report ΔR² between nested models

---

## Title Options

1. "Does Cognitive Fatigue Disrupt the Theta-Gamma Neural Code?"
2. "The Missing Link: Fatigue, Cross-Frequency Coupling, and Working Memory Breakdown"
3. "Frontoparietal Coordination Breakdown Under Cognitive Fatigue"

---

## Summary: What This Study Tests

| Gap from JCN Paper | How This Study Addresses It |
|:-------------------|:----------------------------|
| No cross-frequency coupling studies under active fatigue | ✓ Theta-gamma coupling during WAND task |
| No network-level analysis | ✓ Nodal aggregation + Louvain communities |
| No link between coupling and performance | ✓ Coupling change predicts d′ change |
| Theta paradox unresolved | ✓ Theta frequency as separate measure |
