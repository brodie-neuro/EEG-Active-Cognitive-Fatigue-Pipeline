# EEG Pre-Processing Pipeline

> **System:** NeurOsan Amplifier + Brain Products actiCAP (64 channels)
> **Impedance Target:** <15 kΩ general; **<10 kΩ** for critical sites (Fz, Pz, Oz, T7, T8) to ensure clean gamma band signal.

---

## Phase 1: Foundational Cleaning & Data Conditioning

This phase focuses on loading data, establishing a clean reference, and removing major non-biological artefacts.

### Step 1: Load Data & Define Metadata

**What:** Load the raw data file (e.g., the 5-minute `.bdf` or `.eeg` file). Define electrode locations by loading a montage/channel locations file and confirm the sampling rate.

**Why:** This gives the data its spatial context. The software knows that Fz is on the forehead and Oz is at the back of the head—essential for all subsequent spatial filtering and plotting.

---

### Step 2: Downsample the Data

**What:** Reduce the sampling rate from the high recording rate (e.g., 1000 Hz) to a manageable rate (e.g., 500 Hz).

**Why:** The highest frequency of interest is 85 Hz for gamma. Sampling at 500 Hz is more than five times that frequency—sufficient to capture the signal perfectly (Nyquist theorem). This makes data files smaller and processing faster without loss of important information.

---

### Step 3: Robust Referencing (PyPREP)

**What:** Run the PyPREP pipeline on continuous data. This intelligent algorithm will:

1. Identify consistently "bad" channels (flat, disconnected, or extremely noisy)
2. Calculate the average signal using only good channels
3. Subtract this "clean average" from all good channels (re-referencing)
4. Repair bad channels by interpolating from clean neighbours

**Why:** This is the most critical step for maximising signal-to-noise ratio. Unlike simple average reference, this ensures noise from a few bad electrodes is not spread to all clean ones.

---

### Step 4: Advanced Line Noise Removal (Zapline-plus)

**What:** Apply advanced line noise filtering to remove 50 Hz electrical hum and harmonics (100 Hz, 150 Hz).

**Why:** Simple notch filters can distort brain signals at nearby frequencies. Zapline-plus removes line noise with surgical precision, leaving surrounding neural data untouched. Critical for PAC analysis that examines specific frequencies.

---

## Phase 2: Biological & High-Amplitude Artefact Removal

This phase uses algorithms to find and remove artefacts from the participant's body and movements.

### Step 5: Band-Pass Filtering

**What:** Apply a digital band-pass filter keeping only 1–100 Hz.

**Why:** 
- **1 Hz high-pass:** Removes slow signal drifts from sweat and electrode chemistry that confuse ICA
- **100 Hz low-pass:** Removes high-frequency noise outside the range of interest (non-neural)

---

### Step 6: Automated Artefact Repair (ASR)

**What:** Run Artifact Subspace Reconstruction (ASR). This finds short, high-amplitude bursts of noise (muscle twitches, head movements, electrode jiggles) and repairs them.

**Why:** Large, transient artefacts can dominate the signal and aren't easily removed by other methods. ASR cleans these segments without discarding entire trials, preserving valuable data. **Particularly effective at cleaning transient muscle noise that can contaminate the gamma band.**

---

### Step 7: Independent Component Analysis (ICA + ICLabel)

**What:** Run ICA to "un-mix" 64 channels into 64 independent components. Then use **ICLabel** to automatically classify each component:

> "Component 1: Brain (90%), Component 2: Eye Movement (99%), Component 3: Muscle (95%)"

**Why:** This is the primary tool for removing stereotyped, repetitive biological artefacts. Blinks, heartbeats, and eye movements have consistent electrical signatures. ICA isolates these into single components. **ICLabel makes the process objective and reproducible**, giving defensible, data-driven reasons for removing specific components.

**Process:** Review ICLabel's suggestions, then remove components classified with high confidence as artefacts.

---

## Phase 3: Segmentation & Final Quality Control

This phase segments clean data into analysis windows and performs final checks.

### Step 8: Epoching (Creating Trials)

**What:** Segment the continuous clean data into epochs based on event markers. Create two separate sets:

| Epoch Set | Time-Lock | Window | Purpose |
|:----------|:----------|:-------|:--------|
| **P3b Analysis** | Stimulus markers | −200 to +800 ms | ERP analysis |
| **PAC/Theta Analysis** | Response markers | +200 to +1200 ms | Delay period (avoids stimulus and motor potential) |

**Why:** Precisely isolates specific time windows relevant to different research questions.

---

### Step 9: Final Epoch-Level Rejection/Repair (Autoreject)

**What:** Run Autoreject on the newly created epoch sets.

**Why:** Final quality control performing two crucial tasks:

1. **Rejection:** Finds optimal voltage threshold to reject epochs that are too noisy
2. **Repair:** If an epoch is mostly good but has brief artefact on one or two channels, repairs those channels for that epoch only

**Outcome:** Two sets of clean, high-quality epochs with maximum possible trials retained for statistical analysis.
