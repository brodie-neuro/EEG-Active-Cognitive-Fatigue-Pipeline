# steps/00_make_fake_data.py
"""
Generate synthetic EEG data for pipeline testing.
Produces Block 1 (baseline) and Block 5 (fatigued) with pronounced,
deterministic fatigue effects across ERP, oscillatory, and PAC measures.
"""
from pathlib import Path
import numpy as np
import mne
from mne.channels import make_standard_montage
from mne.export import export_raw

# ---------------- Tunables ----------------
SUBJ = "TEST01"
SFREQ = 500.0
DURATION_S = 5 * 60  # 5 minutes per block

# "Seated WM task" timing
TRIAL_ISI_MIN = 2.0
TRIAL_ISI_MAX = 3.0
STIM_DUR = 0.20
MAINTENANCE_OFFSET = 0.80  # Trigger for maintenance-window onset
TARGET_P = 0.25

# Behaviour model (baseline)
MISS_P = 0.04
ERR_P = 0.08
RT_MEAN = 0.65
RT_SD = 0.15
RT_MIN = 0.25
RT_MAX = 1.20

# Artefact rates
BLINKS_PER_MIN = 15
BLINK_LOCKOUT = 0.45

# Fatigue effects (Block 5 modifiers, all used in generate_block)
FATIGUE = {
    'p3b_amp_scale': 0.45,           # Reduced P3b amplitude
    'p3b_latency_shift_s': 0.08,     # Delayed P3b peak
    'theta_freq_shift_hz': -2.2,     # Slower theta
    'theta_amp_scale': 1.55,         # Stronger low-theta power
    'alpha_freq_shift_hz': -1.0,     # Slower alpha/IAF
    'alpha_amp_scale': 0.55,         # Reduced alpha power
    'gamma_tonic_scale': 0.70,       # Slight gamma reduction
    'pac_amp_scale': 0.10,           # Marked PAC reduction
    'miss_p_increase': 0.10,         # More misses
    'err_p_increase': 0.10,          # More commission errors
    'rt_mean_increase_s': 0.16,      # Slower responses
    'rt_sd_increase_s': 0.05,        # More variable RT
    'blinks_increase_per_min': 8,    # More blinks
}

RNG = np.random.default_rng(42)
# ------------------------------------------

out_dir = Path(__file__).resolve().parents[1] / "raw"
out_dir.mkdir(parents=True, exist_ok=True)

# Montage and channel set
mont = make_standard_montage("standard_1020")
all10_20 = mont.ch_names

sel64 = [
    "Fp1", "Fp2", "AF3", "AF4", "AFz", "F7", "F3", "Fz", "F4", "F8",
    "FC5", "FC1", "FC2", "FC6",
    "FCz",
    "T7", "C3", "Cz", "C4", "T8",
    "CP5", "CP1", "CP2", "CP6",
    "P7", "P5", "P3", "Pz", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2",
    "F1", "F2", "C1", "C2", "CP3", "CPz", "CP4", "P1", "P2",
    "AF7", "AF8", "FT7", "FT8", "FC3", "FC4", "TP7", "TP8",
    "PO1", "PO2", "FT9", "FT10", "TP9", "TP10", "PO5", "PO6"
]
eeg_names = [ch for ch in sel64 if ch in all10_20]

aux_names = ["VEOG", "HEOG", "EMG_L", "EMG_R"]
ch_names = eeg_names + aux_names
ch_types = ["eeg"] * len(eeg_names) + ["eog", "eog", "emg", "emg"]

n_ch = len(ch_names)
n_samp = int(DURATION_S * SFREQ)
times = np.arange(n_samp) / SFREQ


def sine(f, amp=1.0, phase=0.0):
    return amp * np.sin(2 * np.pi * f * times + phase)


def band_noise(f_lo, f_hi, amp=1.0):
    x = RNG.normal(0, 1, n_samp)
    return mne.filter.filter_data(x, SFREQ, f_lo, f_hi, phase="zero") * amp


def pink_noise(n_samples, rng):
    """Generate 1/f pink noise via spectral synthesis."""
    freqs = np.fft.rfftfreq(n_samples)
    freqs[0] = 1
    spectrum = 1 / np.sqrt(freqs)
    spectrum[0] = 0
    phases = rng.uniform(0, 2 * np.pi, len(spectrum))
    fft_vals = spectrum * np.exp(1j * phases)
    signal = np.fft.irfft(fft_vals, n=n_samples)
    signal = (signal - signal.mean()) / signal.std()
    return signal


def make_trial_onsets(duration_s, isi_min, isi_max, rng):
    t = 0.5
    onsets = []
    while t < duration_s - 0.5:
        onsets.append(t)
        t += float(rng.uniform(isi_min, isi_max))
    return np.array(onsets, dtype=float)


def fixed_rate_mask(n, p, rng):
    """
    Create a boolean mask with an exact count close to n * p.
    Reduces block-to-block stochastic drift for synthetic condition effects.
    """
    p = float(np.clip(p, 0.0, 1.0))
    n_true = int(round(n * p))
    n_true = min(max(n_true, 0), n)
    mask = np.zeros(n, dtype=bool)
    if n_true > 0:
        idx = rng.choice(n, size=n_true, replace=False)
        mask[idx] = True
    return mask


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


def in_any_lockout(t, windows):
    for w0, w1 in windows:
        if t < w0:
            return False
        if w0 <= t <= w1:
            return True
    return False


def generate_block(block_num, rng):
    """
    Generate a single block of synthetic EEG.

    Parameters
    ----------
    block_num : int
        Block number (1 = baseline, 5 = fatigued).
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    mne.io.RawArray
        Raw EEG data with annotations.
    """
    is_fatigued = (block_num == 5)

    # Fatigue adjustments
    p3b_scale = FATIGUE['p3b_amp_scale'] if is_fatigued else 1.0
    p3b_latency_shift_s = FATIGUE['p3b_latency_shift_s'] if is_fatigued else 0.0
    theta_shift = FATIGUE['theta_freq_shift_hz'] if is_fatigued else 0.0
    theta_amp_scale = FATIGUE['theta_amp_scale'] if is_fatigued else 1.0
    alpha_shift = FATIGUE['alpha_freq_shift_hz'] if is_fatigued else 0.0
    alpha_amp_scale = FATIGUE['alpha_amp_scale'] if is_fatigued else 1.0
    gamma_tonic_scale = FATIGUE['gamma_tonic_scale'] if is_fatigued else 1.0
    pac_scale = FATIGUE['pac_amp_scale'] if is_fatigued else 1.0
    miss_p = MISS_P + (FATIGUE['miss_p_increase'] if is_fatigued else 0.0)
    err_p = ERR_P + (FATIGUE['err_p_increase'] if is_fatigued else 0.0)
    rt_mean = RT_MEAN + (FATIGUE['rt_mean_increase_s'] if is_fatigued else 0.0)
    rt_sd = RT_SD + (FATIGUE['rt_sd_increase_s'] if is_fatigued else 0.0)
    blinks_per_min = BLINKS_PER_MIN + (FATIGUE['blinks_increase_per_min'] if is_fatigued else 0)

    info = mne.create_info(ch_names=ch_names, sfreq=SFREQ, ch_types=ch_types)
    info.set_montage(mont, match_case=False)

    # ---- Base EEG (Lower noise for clean ERPs) ----
    data = np.zeros((n_ch, n_samp), dtype=float)

    for i in range(len(eeg_names)):
        base = pink_noise(n_samp, rng)
        white = rng.normal(0, 0.1, n_samp)
        # Very low noise background so ERPs are pristine
        data[i] = (base + white) * 5e-6 

    # Slow drift
    drift = sine(0.03, amp=5.0e-6, phase=float(rng.uniform(0, 2 * np.pi)))
    data[:len(eeg_names)] += drift

    # Posterior alpha (fatigue: slower + weaker)
    alpha_f = float(rng.uniform(9.5, 10.5)) + alpha_shift
    for ch in ["Pz", "POz", "PO3", "PO4", "O1", "Oz", "O2", "PO7", "PO8"]:
        if ch in eeg_names:
            data[ch_names.index(ch)] += sine(alpha_f, amp=6.5e-6 * alpha_amp_scale,
                                             phase=float(rng.uniform(0, 2 * np.pi)))

    # Fronto-midline theta (fatigue: slower + stronger)
    theta_f = float(rng.uniform(6.4, 7.2)) + theta_shift
    for ch in ["Fz", "FCz", "Cz"]:
        if ch in eeg_names:
            data[ch_names.index(ch)] += sine(theta_f, amp=2.8e-6 * theta_amp_scale,
                                             phase=float(rng.uniform(0, 2 * np.pi)))

    # Tonic high gamma (broadband, low amplitude)
    # Present across cortex but strongest parietal -- provides baseline
    # for PAC amplitude extraction and realistic PSD shape
    gamma_scale = gamma_tonic_scale
    for ch in eeg_names:
        gamma_f = float(rng.uniform(65, 75))
        data[ch_names.index(ch)] += sine(gamma_f, amp=0.15e-6 * gamma_scale,
                                         phase=float(rng.uniform(0, 2 * np.pi)))
    # Stronger tonic gamma at parietal sites
    for ch in ["P3", "Pz", "P4", "P8", "PO8", "PO4"]:
        if ch in eeg_names:
            gamma_f = float(rng.uniform(65, 75))
            data[ch_names.index(ch)] += sine(gamma_f, amp=0.3e-6 * gamma_scale,
                                             phase=float(rng.uniform(0, 2 * np.pi)))

    # Line noise
    for hz in [50.0, 100.0]:
        data[:len(eeg_names)] += sine(hz, amp=1.5e-7,
                                      phase=float(rng.uniform(0, 2 * np.pi)))

    # ---- Task events ----
    onsets_stim = make_trial_onsets(DURATION_S, TRIAL_ISI_MIN, TRIAL_ISI_MAX, rng)
    n_trials = len(onsets_stim)

    # Exact-rate masks reduce accidental condition inversions in single-subject mocks
    is_target = fixed_rate_mask(n_trials, TARGET_P, rng)
    is_miss = fixed_rate_mask(n_trials, miss_p, rng)
    is_error = np.zeros(n_trials, dtype=bool)
    non_miss_idx = np.where(~is_miss)[0]
    n_error = int(round(len(non_miss_idx) * max(0.0, min(err_p, 1.0))))
    if n_error > 0 and len(non_miss_idx) > 0:
        error_idx = rng.choice(non_miss_idx, size=min(n_error, len(non_miss_idx)),
                               replace=False)
        is_error[error_idx] = True

    rts = np.full(n_trials, np.nan, dtype=float)
    rt_raw = rng.normal(rt_mean, rt_sd, size=n_trials)
    rt_raw = np.clip(rt_raw, RT_MIN, RT_MAX)
    rts[~is_miss] = rt_raw[~is_miss]

    onsets_resp = onsets_stim + np.nan_to_num(rts, nan=0.0)
    has_resp = ~is_miss

    # Maintenance-onset triggers (Trigger 2, at +800 ms post-stimulus)
    onsets_maintenance = onsets_stim + MAINTENANCE_OFFSET

    # ---- ERPs (Textbook P3b) ----
    # 800ms window
    t_erp = np.linspace(0, 0.8, int(0.8 * SFREQ))
    
    # N1 (negative @ 100ms), P2 (positive @ 200ms) - small sensory components
    n1 = -3.0e-6 * np.exp(-0.5 * ((t_erp - 0.100) / 0.020) ** 2)
    p2 =  2.0e-6 * np.exp(-0.5 * ((t_erp - 0.200) / 0.040) ** 2)
    sensory_erp = n1 + p2

    # P3b (positive ~400 ms baseline; delayed in fatigue)
    p3b_base_amp = 25.0e-6 * p3b_scale
    p3b_center = 0.400 + p3b_latency_shift_s
    p3b_erp = p3b_base_amp * np.exp(-0.5 * ((t_erp - p3b_center) / 0.140) ** 2)

    posterior_erp_chs = [c for c in ["Pz", "CPz", "POz", "P3", "P4", "P1", "P2"] if c in eeg_names]
    frontal_erp_chs = [c for c in ["Fz", "FCz", "Fz", "AFz"] if c in eeg_names]

    for tr, t0 in enumerate(onsets_stim):
        # Sensory ERP on all trials
        for ch in posterior_erp_chs:
            add_erp(data[ch_names.index(ch)], t0, sensory_erp, SFREQ)
        for ch in frontal_erp_chs:
            add_erp(data[ch_names.index(ch)], t0, 0.5 * sensory_erp, SFREQ)
            
        # P3b on targets only
        if is_target[tr]:
            for ch in posterior_erp_chs:
                add_erp(data[ch_names.index(ch)], t0, p3b_erp, SFREQ)

    # ---- PAC: inject directed RF(theta phase) -> RP(gamma amplitude) coupling ----
    rf_chs = [c for c in ["F8", "F4", "FC6"] if c in eeg_names]
    rp_chs = [c for c in ["P8", "P4", "PO8"] if c in eeg_names]

    for tr, t0 in enumerate(onsets_stim):
        # Maintenance window: 800-1800 ms post-stimulus
        maint_start = t0 + MAINTENANCE_OFFSET
        maint_end = t0 + 1.8
        i_start = int(maint_start * SFREQ)
        i_end = min(int(maint_end * SFREQ), n_samp)
        if i_start >= n_samp or i_end <= i_start:
            continue

        seg_len = i_end - i_start
        seg_t = np.arange(seg_len) / SFREQ

        # RF theta phase source
        theta_phase = 2 * np.pi * theta_f * seg_t
        rf_theta = np.sin(theta_phase) * (2.2e-6 if not is_fatigued else 1.2e-6)

        # RP high-gamma carrier amplitude-modulated by RF theta phase
        gamma_f = 70.0
        gamma_carrier = np.sin(2 * np.pi * gamma_f * seg_t)
        pac_envelope = (1.0 + pac_scale * np.cos(theta_phase)) / 2.0
        rp_pac = pac_envelope * gamma_carrier * 1.6e-6

        for ch in rf_chs:
            data[ch_names.index(ch), i_start:i_end] += rf_theta
        for ch in rp_chs:
            data[ch_names.index(ch), i_start:i_end] += rp_pac

    # ---- EOG ----
    veog = np.zeros(n_samp, dtype=float)
    heog = band_noise(0.2, 6.0, amp=18e-6)

    blink_rate_hz = blinks_per_min / 60.0
    n_blinks_expected = int(DURATION_S * blink_rate_hz * 1.3)
    blink_times = rng.uniform(0.0, DURATION_S, size=n_blinks_expected)
    blink_times.sort()

    stim_lock_windows = np.column_stack([onsets_stim, onsets_stim + BLINK_LOCKOUT])
    blink_times_ok = [t for t in blink_times if not in_any_lockout(t, stim_lock_windows)]

    blink_len = int(0.30 * SFREQ)
    bt = np.arange(blink_len) / SFREQ
    blink_shape = gauss_kernel(0.10, 0.035, 1.0, bt) + gauss_kernel(0.18, 0.050, 0.6, bt)
    blink_shape = blink_shape / np.max(blink_shape)

    for t in blink_times_ok:
        i0 = int(t * SFREQ)
        i1 = min(i0 + blink_len, n_samp)
        seg_len = i1 - i0
        amp = float(rng.uniform(90e-6, 150e-6))
        veog[i0:i1] += amp * blink_shape[:seg_len]

    data[ch_names.index("VEOG")] = veog
    data[ch_names.index("HEOG")] = heog

    for ch in ["Fp1", "Fp2", "AF3", "AF4", "Fz", "F1", "F2"]:
        if ch in eeg_names:
            data[ch_names.index(ch)] += 0.10 * veog

    # ---- EMG ----
    emg_l = band_noise(40.0, 110.0, amp=2.0e-6)
    emg_r = band_noise(40.0, 110.0, amp=2.0e-6)

    burst_len = int(0.12 * SFREQ)
    bt2 = np.arange(burst_len) / SFREQ
    burst_env = gauss_kernel(0.05, 0.020, 1.0, bt2)
    burst_env = burst_env / np.max(burst_env)

    # Precompute high-gamma carriers once (faster than per-trial filtering)
    burst_carrier_l_full = band_noise(60.0, 120.0, amp=1.0)
    burst_carrier_r_full = band_noise(60.0, 120.0, amp=1.0)

    for tr in range(n_trials):
        if not has_resp[tr]:
            continue
        t_resp = float(onsets_resp[tr])
        i0 = int((t_resp - 0.02) * SFREQ)
        if i0 < 0:
            continue
        i1 = min(i0 + burst_len, n_samp)
        seg_len = i1 - i0
        if seg_len <= 0:
            continue
        burst_carrier_l = burst_carrier_l_full[i0:i1]
        burst_carrier_r = burst_carrier_r_full[i0:i1]
        amp = float(rng.uniform(4e-6, 9e-6))
        emg_l[i0:i1] += amp * burst_env[:seg_len] * burst_carrier_l
        emg_r[i0:i1] += amp * burst_env[:seg_len] * burst_carrier_r

    data[ch_names.index("EMG_L")] = emg_l
    data[ch_names.index("EMG_R")] = emg_r

    for ch in ["C3", "C4", "FC3", "FC4"]:
        if ch in eeg_names:
            data[ch_names.index(ch)] += 0.01 * (emg_l + emg_r)

    raw = mne.io.RawArray(data, info, verbose="ERROR")

    # ---- Annotations ----
    # Stimulus onset (Trigger 1)
    stim_desc = np.where(is_target, "stim/target", "stim/nontarget")
    ann = mne.Annotations(onsets_stim, [STIM_DUR] * n_trials, list(stim_desc))

    # Maintenance onset (Trigger 2) -- marks start of maintenance window
    valid_offset = onsets_maintenance < DURATION_S
    ann += mne.Annotations(
        onsets_maintenance[valid_offset],
        [0.01] * int(valid_offset.sum()),
        ["stim/offset"] * int(valid_offset.sum())
    )

    # Response annotations
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

    # Miss markers
    miss_onsets = onsets_stim[is_miss] + MAINTENANCE_OFFSET
    if len(miss_onsets) > 0:
        ann += mne.Annotations(miss_onsets, [0.01] * len(miss_onsets),
                               ["resp/miss"] * len(miss_onsets))

    raw.set_annotations(ann)
    return raw, n_trials, is_target, is_miss, is_error


# ---- Generate both blocks ----
if __name__ == "__main__":
    for block_num in [1, 5]:
        print(f"\n{'='*60}")
        print(f"  Generating Block {block_num} {'(BASELINE)' if block_num == 1 else '(FATIGUED)'}")
        print(f"{'='*60}")

        # Use different RNG seeds per block for variation
        block_rng = np.random.default_rng(42 + block_num)
        raw, n_trials, is_target, is_miss, is_error = generate_block(block_num, block_rng)

        vhdr_path = out_dir / f"sub-{SUBJ}_block{block_num}-task.vhdr"
        export_raw(vhdr_path, raw, fmt="brainvision", overwrite=True)

        print(f"Wrote {vhdr_path}")
        print(f"Trials: {n_trials} | targets: {int(is_target.sum())} "
              f"| misses: {int(is_miss.sum())} | errors: {int(is_error.sum())}")
        print(f"EEG channels: {len(eeg_names)} | total channels incl aux: {len(ch_names)}")

    print(f"\nDone -- Block 1 and Block 5 synthetic data written to {out_dir}")
