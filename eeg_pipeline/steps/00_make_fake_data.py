# steps/00_make_fake_data.py
from pathlib import Path
import numpy as np
import mne
from mne.channels import make_standard_montage
from mne.export import export_raw

# ---------------- Tunables ----------------
SUBJ = "TEST01"
BLOCK = 1
SFREQ = 500.0
DURATION_S = 5 * 60

# "Seated WM task" timing
TRIAL_ISI_MIN = 2.0          # seconds between stims
TRIAL_ISI_MAX = 3.0
STIM_DUR = 0.20              # seconds (only used for annotations)
TARGET_P = 0.25              # proportion of target trials

# Behaviour model
MISS_P = 0.04                # no response
ERR_P = 0.08                 # incorrect response when not missed
RT_MEAN = 0.65               # seconds
RT_SD = 0.15                 # seconds
RT_MIN = 0.25
RT_MAX = 1.20

# Artefact rates
BLINKS_PER_MIN = 15          # typical range 10–20 seated
BLINK_LOCKOUT = 0.45         # seconds after stim where blinking is suppressed

RNG = np.random.default_rng(42)
# ------------------------------------------

out_dir = Path(__file__).resolve().parents[1] / "raw"
out_dir.mkdir(parents=True, exist_ok=True)
vhdr_path = out_dir / f"sub-{SUBJ}_block{BLOCK}-task.vhdr"

# Montage and channel set
mont = make_standard_montage("standard_1020")
all10_20 = mont.ch_names

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

aux_names = ["VEOG", "HEOG", "EMG_L", "EMG_R"]
ch_names = eeg_names + aux_names
ch_types = ["eeg"] * len(eeg_names) + ["eog", "eog", "emg", "emg"]

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

def make_trial_onsets(duration_s, isi_min, isi_max, rng):
    t = 0.5  # start after 0.5 s
    onsets = []
    while t < duration_s - 0.5:
        onsets.append(t)
        t += float(rng.uniform(isi_min, isi_max))
    return np.array(onsets, dtype=float)

def gauss_kernel(center_s, width_s, amp, tvec):
    return amp * np.exp(-0.5 * ((tvec - center_s) / width_s) ** 2)

def add_erp(data_1d, onset_s, kernel, sfreq):
    i0 = int(onset_s * sfreq)
    klen = len(kernel)
    i1 = i0 + klen
    if i0 < 0 or i0 >= len(data_1d):
        return
    if i1 > len(data_1d):
        klen = len(data_1d) - i0
        data_1d[i0:] += kernel[:klen]
    else:
        data_1d[i0:i1] += kernel

# ---------------- Base "human-ish" EEG ----------------
data = np.zeros((n_ch, n_samp), dtype=float)

def pink_noise(n_samples, rng):
    """Generate 1/f pink noise via spectral synthesis."""
    # Generate white noise in frequency domain
    freqs = np.fft.rfftfreq(n_samples)
    freqs[0] = 1  # Avoid division by zero
    # 1/f spectrum (pink noise has -3dB/octave rolloff)
    spectrum = 1 / np.sqrt(freqs)
    spectrum[0] = 0  # No DC offset
    # Random phases
    phases = rng.uniform(0, 2 * np.pi, len(spectrum))
    # Synthesize
    fft_vals = spectrum * np.exp(1j * phases)
    signal = np.fft.irfft(fft_vals, n=n_samples)
    # Normalize to unit variance
    signal = (signal - signal.mean()) / signal.std()
    return signal

# Realistic 1/f background for each EEG channel (~20-50 µV RMS typical for raw EEG)
for i in range(len(eeg_names)):
    # Pink noise as base (1/f spectrum)
    base = pink_noise(n_samp, RNG)
    # Add some broadband white noise for higher frequencies
    white = RNG.normal(0, 0.3, n_samp)
    # Combine and scale to realistic amplitude (~30 µV RMS)
    combined = base + white
    data[i] = combined * 30e-6  # 30 µV RMS

# Slow drift (seated, minimal movement)
drift = sine(0.03, amp=5.0e-6, phase=float(RNG.uniform(0, 2*np.pi)))
data[:len(eeg_names)] += drift

# Posterior alpha around 10 Hz with slight individual variation
alpha_f = float(RNG.uniform(9.5, 10.5))
for ch in ["Pz","POz","PO3","PO4","O1","Oz","O2","PO7","PO8"]:
    if ch in eeg_names:
        data[ch_names.index(ch)] += sine(alpha_f, amp=6e-6, phase=float(RNG.uniform(0, 2*np.pi)))

# Fronto-midline theta around 5–6 Hz
theta_f = float(RNG.uniform(5.0, 6.2))
for ch in ["Fz","FCz","Cz"]:
    if ch in eeg_names:
        data[ch_names.index(ch)] += sine(theta_f, amp=2.5e-6, phase=float(RNG.uniform(0, 2*np.pi)))

# Line noise (UK 50 Hz) plus harmonic, modest
for hz in [50.0, 100.0]:
    data[:len(eeg_names)] += sine(hz, amp=1.5e-7, phase=float(RNG.uniform(0, 2*np.pi)))

# ---------------- Task events: stim every 2–3 s, response after RT ----------------
onsets_stim = make_trial_onsets(DURATION_S, TRIAL_ISI_MIN, TRIAL_ISI_MAX, RNG)
n_trials = len(onsets_stim)

is_target = RNG.random(n_trials) < TARGET_P
is_miss = RNG.random(n_trials) < MISS_P
is_error = (RNG.random(n_trials) < ERR_P) & (~is_miss)

# RTs only for non-missed trials
rts = np.full(n_trials, np.nan, dtype=float)
rt_raw = RNG.normal(RT_MEAN, RT_SD, size=n_trials)
rt_raw = np.clip(rt_raw, RT_MIN, RT_MAX)
rts[~is_miss] = rt_raw[~is_miss]

onsets_resp = onsets_stim + np.nan_to_num(rts, nan=0.0)
has_resp = ~is_miss

# ---------------- Add stimulus-locked ERPs ----------------
# Kernel window, 0 to 0.8 s
erp_t = np.arange(int(0.8 * SFREQ)) / SFREQ

# Small sensory ERP for all stims: N1 then P2 then small P3-ish
sensory_kernel = (
    gauss_kernel(0.10, 0.020, -1.2e-6, erp_t) +
    gauss_kernel(0.18, 0.030,  1.0e-6, erp_t) +
    gauss_kernel(0.32, 0.060,  0.8e-6, erp_t)
)

# Extra P3b for targets (bigger at Pz/CPz/POz)
p3_kernel = gauss_kernel(0.38, 0.080, 2.8e-6, erp_t)

posterior_erp_chs = [c for c in ["Pz","CPz","POz","P3","P4"] if c in eeg_names]
frontal_erp_chs = [c for c in ["Fz","FCz"] if c in eeg_names]

for tr, t0 in enumerate(onsets_stim):
    # All trials: sensory ERP mostly posterior, smaller frontal
    for ch in posterior_erp_chs:
        add_erp(data[ch_names.index(ch)], t0, sensory_kernel, SFREQ)
    for ch in frontal_erp_chs:
        add_erp(data[ch_names.index(ch)], t0, 0.45 * sensory_kernel, SFREQ)

    # Targets: extra P3b posterior
    if is_target[tr]:
        for ch in posterior_erp_chs:
            add_erp(data[ch_names.index(ch)], t0, p3_kernel, SFREQ)

# ---------------- EOG: blinks mostly between trials, not during stim ----------------
veog = np.zeros(n_samp, dtype=float)
heog = band_noise(0.2, 6.0, amp=18e-6)

# Blink times from a Poisson process with lockout after each stim
blink_rate_hz = BLINKS_PER_MIN / 60.0
n_blinks_expected = int(DURATION_S * blink_rate_hz * 1.3)
blink_times = RNG.uniform(0.0, DURATION_S, size=n_blinks_expected)
blink_times.sort()

# Reject blinks that occur within lockout window after any stimulus
stim_lock_windows = np.column_stack([onsets_stim, onsets_stim + BLINK_LOCKOUT])

def in_any_lockout(t, windows):
    # windows is (n,2), sorted by start
    # fast-ish check by scanning a bit
    for w0, w1 in windows:
        if t < w0:
            return False
        if w0 <= t <= w1:
            return True
    return False

blink_times_ok = [t for t in blink_times if not in_any_lockout(t, stim_lock_windows)]

# Blink shape: smooth bump ~250 ms, 100–150 µV peak
blink_len = int(0.30 * SFREQ)
bt = np.arange(blink_len) / SFREQ
blink_shape = (
    gauss_kernel(0.10, 0.035, 1.0, bt) +
    gauss_kernel(0.18, 0.050, 0.6, bt)
)
blink_shape = blink_shape / np.max(blink_shape)

for t in blink_times_ok:
    i0 = int(t * SFREQ)
    i1 = min(i0 + blink_len, n_samp)
    seg_len = i1 - i0
    amp = float(RNG.uniform(90e-6, 150e-6))
    veog[i0:i1] += amp * blink_shape[:seg_len]

data[ch_names.index("VEOG")] = veog
data[ch_names.index("HEOG")] = heog

# Leak VEOG into frontal EEG (realistic)
for ch in ["Fp1","Fp2","AF3","AF4","Fz","F1","F2"]:
    if ch in eeg_names:
        data[ch_names.index(ch)] += 0.10 * veog

# ---------------- EMG: mostly quiet, response-locked small bursts ----------------
emg_l = band_noise(40.0, 110.0, amp=2.0e-6)
emg_r = band_noise(40.0, 110.0, amp=2.0e-6)

# Add short EMG bursts around button presses
burst_len = int(0.12 * SFREQ)
bt2 = np.arange(burst_len) / SFREQ
burst_env = gauss_kernel(0.05, 0.020, 1.0, bt2)
burst_env = burst_env / np.max(burst_env)

for tr in range(n_trials):
    if not has_resp[tr]:
        continue
    t_resp = float(onsets_resp[tr])
    i0 = int((t_resp - 0.02) * SFREQ)
    if i0 < 0:
        continue
    i1 = min(i0 + burst_len, n_samp)
    seg_len = i1 - i0
    burst_carrier_l = band_noise(60.0, 120.0, amp=1.0)[:seg_len]
    burst_carrier_r = band_noise(60.0, 120.0, amp=1.0)[:seg_len]
    amp = float(RNG.uniform(4e-6, 9e-6))
    emg_l[i0:i1] += amp * burst_env[:seg_len] * burst_carrier_l
    emg_r[i0:i1] += amp * burst_env[:seg_len] * burst_carrier_r

data[ch_names.index("EMG_L")] = emg_l
data[ch_names.index("EMG_R")] = emg_r

# Small EMG leak into central channels (finger press vicinity)
for ch in ["C3","C4","FC3","FC4"]:
    if ch in eeg_names:
        data[ch_names.index(ch)] += 0.01 * (emg_l + emg_r)

raw = mne.io.RawArray(data, info, verbose="ERROR")

# ---------------- Annotations: richer labels ----------------
stim_desc = np.where(is_target, "stim/target", "stim/nontarget")
ann = mne.Annotations(onsets_stim, [STIM_DUR] * n_trials, list(stim_desc))

# Response annotations only when present
resp_onsets = onsets_resp[has_resp]
resp_dur = [0.01] * len(resp_onsets)
resp_desc = []
idxs = np.where(has_resp)[0]
for tr in idxs:
    if is_error[tr]:
        resp_desc.append("resp/error")
    else:
        resp_desc.append("resp/correct")
ann += mne.Annotations(resp_onsets, resp_dur, resp_desc)

# Miss markers (useful for later)
miss_onsets = onsets_stim[is_miss] + 0.8
if len(miss_onsets) > 0:
    ann += mne.Annotations(miss_onsets, [0.01] * len(miss_onsets), ["resp/miss"] * len(miss_onsets))

raw.set_annotations(ann)

# Export BrainVision
export_raw(vhdr_path, raw, fmt="brainvision", overwrite=True)
print(f"Wrote {vhdr_path}")
print(f"Trials: {n_trials} | targets: {int(is_target.sum())} | misses: {int(is_miss.sum())} | errors: {int(is_error.sum())}")
print(f"EEG channels: {len(eeg_names)} | total channels incl aux: {len(ch_names)}")
