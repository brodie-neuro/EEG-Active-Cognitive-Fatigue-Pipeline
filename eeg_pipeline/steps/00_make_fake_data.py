# steps/00_make_fake_data.py
from pathlib import Path
import numpy as np
import mne
from mne.channels import make_standard_montage
from mne.export import export_raw

# ---------------- Tunables ----------------
SUBJ = "TEST01"
BLOCK = 1
SFREQ = 500.0             # samples per second
DURATION_S = 5 * 60       # 5 minutes
RNG = np.random.default_rng(42)
# ------------------------------------------

out_dir = Path(__file__).resolve().parents[1] / "raw"
out_dir.mkdir(parents=True, exist_ok=True)
vhdr_path = out_dir / f"sub-{SUBJ}_block{BLOCK}-task.vhdr"

# 64 EEG based on 10-20 names
mont = make_standard_montage("standard_1020")
# Pick a common 64 set from the montage names
all10_20 = mont.ch_names
# A typical 64 selection (includes mastoids TP9, TP10)
sel64 = [
 "Fp1","Fp2","AF3","AF4","F7","F3","Fz","F4","F8",
 "FC5","FC1","FC2","FC6",
 "T7","C3","Cz","C4","T8",
 "CP5","CP1","CP2","CP6",
 "P7","P3","Pz","P4","P8",
 "PO7","PO3","POz","PO4","PO8","O1","Oz","O2",
 "F1","F2","C1","C2","CP3","CPz","CP4","P1","P2",
 "AF7","AF8","FT7","FT8","FC3","FC4","TP7","TP8",
 "PO1","PO2","FT9","FT10","TP9","TP10","PO5","PO6"
]
eeg_names = [ch for ch in sel64 if ch in all10_20]

aux_names = ["VEOG","HEOG","EMG_L","EMG_R"]  # VEOG = vertical eye, HEOG = horizontal eye, EMG = jaw muscle
ch_names = eeg_names + aux_names
ch_types = ["eeg"] * len(eeg_names) + ["eog","eog","emg","emg"]

n_ch = len(ch_names)
n_samp = int(DURATION_S * SFREQ)
times = np.arange(n_samp) / SFREQ

info = mne.create_info(ch_names=ch_names, sfreq=SFREQ, ch_types=ch_types)
info.set_montage(mont, match_case=False)

def sine(f, amp=1.0, phase=0.0):
    return amp * np.sin(2 * np.pi * f * times + phase)

def band_noise(f_lo, f_hi, amp=1.0):
    x = RNG.normal(0, 1, n_samp)
    return mne.filter.filter_data(x, SFREQ, f_lo, f_hi, phase="zero") * amp

# Base EEG
data = np.zeros((n_ch, n_samp), dtype=float)
for i in range(len(eeg_names)):
    lp = float(RNG.uniform(20, 40))
    data[i] = mne.filter.filter_data(RNG.normal(0, 1, n_samp), SFREQ, 0.1, lp, phase="zero") * 3e-6

# Posterior alpha 10 Hz
for ch in ["Pz","POz","PO3","PO4","O1","Oz","O2","PO7","PO8"]:
    if ch in eeg_names:
        data[ch_names.index(ch)] += sine(10.0, amp=8e-6)

# Fronto-midline theta 5.5 Hz
for ch in ["Fz","FCz","Cz"]:
    if ch in eeg_names:
        data[ch_names.index(ch)] += sine(5.5, amp=3e-6)

# Gamma 70 Hz bursts during the second half of a 1.8 s cycle
cycle = 1.8
burst_gate = (np.mod(times, cycle) > 0.9).astype(float)
burst_gate = mne.filter.filter_data(burst_gate, SFREQ, 0.5, 4.0, phase="zero")
for ch in ["POz","Oz","O1","O2"]:
    if ch in eeg_names:
        data[ch_names.index(ch)] += sine(70.0, amp=1.5e-6) * burst_gate

# Line hum at 50 and 100 Hz on EEG
for hz in [50.0, 100.0]:
    data[:len(eeg_names)] += sine(hz, amp=2e-7)

# Eye channels with blinks and saccades
veog = np.zeros(n_samp)
heog = band_noise(1.0, 8.0, amp=30e-6)
# Put some blinks during the stimulus part of each cycle
blink_prob = 0.01
blink_on = (np.mod(times, cycle) < 0.4) & (RNG.random(n_samp) < blink_prob)
blink_shape = mne.filter.filter_data((RNG.normal(0,1,int(0.25*SFREQ))>1.5).astype(float), SFREQ, 1.0, 8.0, phase="zero")
for i in np.where(blink_on)[0]:
    j1 = min(i + len(blink_shape), n_samp)
    veog[i:j1] += 120e-6 * blink_shape[:j1-i]  # about 120 microvolts

data[ch_names.index("VEOG")] = veog
data[ch_names.index("HEOG")] = heog

# Leak VEOG into frontal EEG so it looks real
for ch in ["Fp1","Fp2","AF3","AF4","Fz","F1","F2"]:
    if ch in eeg_names:
        data[ch_names.index(ch)] += 0.12 * veog

# Jaw EMG 60–100 Hz with bursts
def emg_like():
    env = mne.filter.filter_data(RNG.random(n_samp), SFREQ, 0.3, 3.0, phase="zero")
    env = (env - env.min()) / (env.max() - env.min())
    carrier = band_noise(60.0, 100.0, amp=1.0)
    return (0.2 + 0.8 * env) * carrier * 8e-6

emg_l = emg_like()
emg_r = emg_like()
data[ch_names.index("EMG_L")] = emg_l
data[ch_names.index("EMG_R")] = emg_r

# Leak a bit of EMG into frontal EEG
for ch in ["Fp1","Fp2","F7","F8","F3","F4"]:
    if ch in eeg_names:
        data[ch_names.index(ch)] += 0.03 * (emg_l + emg_r)

raw = mne.io.RawArray(data, info, verbose="ERROR")

# Add events as annotations: "stim" every 1.8 s, "resp" 0.6 s later
onsets_stim = np.arange(0.0, DURATION_S, cycle)
onsets_resp = onsets_stim + 0.6
ann = mne.Annotations(onsets_stim, [0.001]*len(onsets_stim), ["stim"]*len(onsets_stim))
ann += mne.Annotations(onsets_resp, [0.001]*len(onsets_resp), ["resp"]*len(onsets_resp))
raw.set_annotations(ann)

# Export as BrainVision triplet .vhdr + .vmrk + .eeg
export_raw(vhdr_path, raw, fmt="brainvision", overwrite=True)
print(f"Wrote {vhdr_path}")
