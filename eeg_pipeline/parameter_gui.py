# eeg_pipeline/parameter_gui.py
"""
Premium Parameter Editor GUI for the EEG Pipeline.

Dark marble theme with glassmorphism panels. Reads and writes
parameters.json with live validation and section-based navigation.

Features:
  - Rich tooltips (Info + Technical for every parameter)
  - Smart widgets: tag editors for lists, dropdowns for enums
  - Preset save/load system
  - Auto-backup on every save
  - Parameter hash for reproducibility tracking

Usage:
    python parameter_gui.py
"""
import json
import sys
import hashlib
import math
import subprocess
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox, font as tkfont, filedialog

PIPELINE_DIR = Path(__file__).resolve().parent
PARAMS_PATH = PIPELINE_DIR / "config" / "parameters.json"
PRESETS_DIR = PIPELINE_DIR / "config" / "presets"
PRESETS_DIR.mkdir(exist_ok=True)

# ============================================================
# COLOUR PALETTE -- Dark Marble + Glass
# ============================================================
C = {
    'bg_deep':        '#060708',
    'bg_dark':        '#0a0c0d',
    'bg_panel':       '#171c1f',
    'bg_panel_alt':   '#1e2529',
    'bg_entry':       '#222b30',
    'bg_entry_focus': '#2b373d',
    'border_glass':   '#38434b',
    'border_accent':  '#4d6d78',
    'text_primary':   '#f4f8fb',
    'text_secondary': '#d7e0e5',
    'text_dim':       '#b4c0c6',
    'accent_blue':    '#8ecae0',
    'accent_cyan':    '#8fe0c4',
    'accent_purple':  '#8ecae0',
    'accent_green':   '#8fe0c4',
    'accent_red':     '#8ecae0',
    'accent_gold':    '#8fe0c4',
    'marble_vein':    '#1e2529',
    'marble_vein_soft': '#171d20',
    'marble_sheen':   '#0c0f11',
    'scrollbar_bg':   '#101214',
    'scrollbar_fg':   '#2a3136',
    'tag_bg':         '#20272b',
    'tag_border':     '#374047',
    'nav_hover_bg':   '#131a1e',
    'nav_active_bg':  '#1b252b',
    'nav_indicator_hover': '#2f3c44',
}

# ============================================================
# SECTION METADATA
# ============================================================
SECTION_META = {
    'filtering': {
        'label': 'Filtering',
        'color': C['accent_blue'],
        'info': 'Bandpass and notch filter settings applied during raw data import.',
        'technical': 'FIR filters via MNE. High-pass removes DC drift, low-pass prevents aliasing. Notch removes powerline interference.',
    },
    'zapline': {
        'label': 'ZapLine',
        'color': C['accent_cyan'],
        'info': 'Spatial filtering to remove line noise (50/60 Hz) and harmonics.',
        'technical': 'Uses DSS (Denoising Source Separation) from meegkit. Iteratively removes fundamental + N harmonics below Nyquist.',
    },
    'ica': {
        'label': 'ICA & ICLabel',
        'color': C['accent_purple'],
        'info': 'Independent Component Analysis for artifact removal, with automated classification via ICLabel.',
        'technical': 'MNE ICA (infomax/fastica). ICLabel classifies each IC as brain/eye/heart/muscle/noise. Components exceeding per-category thresholds are excluded.',
    },
    'asr': {
        'label': 'ASR',
        'color': C['accent_green'],
        'info': 'Artifact Subspace Reconstruction for cleaning transient high-amplitude bursts.',
        'technical': 'meegkit ASR. Cutoff controls aggressiveness in SD units. Lower cutoff = more aggressive cleaning. 15-30 typical range.',
    },
    'autoreject': {
        'label': 'Autoreject',
        'color': C['accent_gold'],
        'info': 'Automated epoch-level artifact rejection and channel interpolation.',
        'technical': 'Bayesian optimisation over n_interpolate and consensus grids. Cross-validated to find optimal rejection threshold per channel.',
    },
    'epoching': {
        'label': 'Epoching',
        'color': C['accent_blue'],
        'info': 'Time windows for cutting continuous data into stimulus-locked epochs.',
        'technical': 'P3b epochs: stimulus-locked for ERP. PAC epochs: offset by stimulus duration for maintenance-period coupling analysis.',
    },
    'p3b': {
        'label': 'P3b Analysis',
        'color': C['accent_cyan'],
        'info': 'P3b ERP component extraction: mean amplitude and peak latency.',
        'technical': 'Measured at centroparietal electrodes (default: Pz). Peak search window 300-500 ms post-stimulus. Polich (2007) canonical site.',
    },
    'specparam': {
        'label': 'Specparam',
        'color': C['accent_purple'],
        'info': 'Spectral parameterisation: separates periodic peaks from aperiodic (1/f) background.',
        'technical': 'Fits Gaussian peaks over a linear background in log-log space. Used to extract IAF and ITF. Replaces fixed canonical bands.',
    },
    'band_power': {
        'label': 'Band Power',
        'color': C['accent_green'],
        'info': 'Frequency band power analysis with individualised or default band definitions.',
        'technical': 'Log-transformed PSD integrated over individualised theta/alpha bands. Falls back to canonical bands if specparam finds no peak.',
    },
    'pac': {
        'label': 'PAC',
        'color': C['accent_red'],
        'info': 'Phase-Amplitude Coupling between theta phase and gamma amplitude.',
        'technical': 'Modulation Index via tensorpac with surrogate z-scoring (N permutations). Tests theta-gamma neural code hypothesis.',
    },
    'iaf': {
        'label': 'IAF',
        'color': C['accent_gold'],
        'info': 'Individual Alpha Frequency from resting-state posterior electrodes.',
        'technical': 'Extracted via specparam from eyes-closed rest. Search range [6-14] Hz. Used to define individualised alpha band = IAF +/- bandwidth.',
    },
    'itf': {
        'label': 'ITF',
        'color': C['accent_blue'],
        'info': 'Individual Theta Frequency from task-related frontal midline activity.',
        'technical': 'Extracted during maintenance epochs at Fz. Search range [3-9] Hz. Theta band capped at IAF-1 Hz to prevent alpha leakage.',
    },
    'qc': {
        'label': 'QC Thresholds',
        'color': C['accent_red'],
        'info': 'Quality control thresholds for automated pass/fail assessment.',
        'technical': 'Each preprocessing step checks metrics against these thresholds. Exceeding a threshold flags WARNING (within 20%) or FAIL in the assessment report.',
    },
}

# ============================================================
# FIELD-LEVEL HELP (info + technical)
# ============================================================
FIELD_HELP = {
    'hp_freq': {
        'info': 'High-pass filter cutoff frequency.',
        'technical': 'Removes slow drifts and DC offset. 0.1 Hz is standard for ERP; use 1.0 Hz for connectivity analyses.',
        'unit': 'Hz',
    },
    'lp_freq': {
        'info': 'Low-pass filter cutoff frequency.',
        'technical': 'Anti-aliasing filter. Set to preserve gamma: 120 Hz for 500+ Hz sampling rate.',
        'unit': 'Hz',
    },
    'notch_freq': {
        'info': 'Powerline frequency for notch/ZapLine removal.',
        'technical': '50 Hz (UK/EU/AU) or 60 Hz (US/Canada). ZapLine uses this as fundamental; harmonics are removed automatically.',
        'unit': 'Hz',
    },
    'notch_width': {
        'info': 'Width of the notch filter.',
        'technical': 'Only used as fallback when ZapLine fails. Wider = more aggressive but may distort nearby frequencies.',
        'unit': 'Hz',
    },
    'n_harmonics': {
        'info': 'Number of line noise harmonics to remove.',
        'technical': '4 removes 50, 100, 150, 200 Hz. Increase if harmonics visible in PSD after ZapLine.',
        'unit': 'count',
    },
    'chunk_length': {
        'info': 'ZapLine processing chunk length.',
        'technical': 'Data is processed in chunks of this length. Shorter chunks adapt better to non-stationary noise.',
        'unit': 'seconds',
    },
    'n_components': {
        'info': 'Number of ICA components to extract.',
        'technical': '25 is typical for 64-channel data. Reducing this speeds up ICA but may miss artifacts. Must be <= number of good channels.',
        'unit': 'count',
    },
    'method': {
        'info': 'ICA decomposition algorithm.',
        'technical': 'infomax: robust, widely used. fastica: faster but may be less stable. picard: modern, fast convergence.',
        'choices': ['infomax', 'fastica', 'picard'],
    },
    'random_state': {
        'info': 'Random seed for reproducibility.',
        'technical': 'Ensures identical ICA decomposition across runs. Change only to test robustness of results.',
        'unit': 'integer',
    },
    'eye': {
        'info': 'Probability threshold for excluding eye artifact ICs.',
        'technical': 'ICLabel probability > this value = exclude. Lower = more aggressive (catches more blinks but risks removing brain signal).',
        'unit': '0-1',
    },
    'heart': {
        'info': 'Probability threshold for excluding heart artifact ICs.',
        'technical': 'ECG artifacts. Usually clear in topography (dipolar, frontal). 0.80 is conservative.',
        'unit': '0-1',
    },
    'muscle': {
        'info': 'Probability threshold for excluding muscle artifact ICs.',
        'technical': 'EMG contamination. Set higher (0.90+) to avoid over-rejecting temporal/frontal brain ICs that overlap with muscle topography.',
        'unit': '0-1',
    },
    'channel_noise': {
        'info': 'Probability threshold for excluding channel-noise ICs.',
        'technical': 'Single-channel noise. High threshold (0.95) since these are rare and distinctive.',
        'unit': '0-1',
    },
    'line_noise': {
        'info': 'Probability threshold for excluding line-noise ICs.',
        'technical': 'Residual 50/60 Hz. Usually handled by ZapLine, so threshold can be high.',
        'unit': '0-1',
    },
    'other': {
        'info': 'Probability threshold for excluding "other" artifact ICs.',
        'technical': 'Catch-all category. Set very high (0.99) to avoid false positives.',
        'unit': '0-1',
    },
    'cutoff': {
        'info': 'ASR aggressiveness (in standard deviations).',
        'technical': 'Lower = more aggressive. 20 is moderate. Range: 5 (very aggressive, use for noisy data) to 50+ (very conservative).',
        'unit': 'SD',
    },
    'win_len': {
        'info': 'ASR sliding window length.',
        'technical': 'Shorter windows track faster changes. 0.5s is standard.',
        'unit': 'seconds',
    },
    'n_interpolate': {
        'info': 'Number of channels to try interpolating per bad epoch.',
        'technical': 'Autoreject tests each value and selects optimal via cross-validation. Wider range = better fit but slower.',
        'unit': 'list of integers',
    },
    'consensus': {
        'info': 'Fraction of channels that must agree an epoch is bad.',
        'technical': 'Lower = stricter rejection. [0.1, 0.5, 1.0] tests a range from strict to lenient.',
        'unit': 'list of fractions',
    },
    'cv': {
        'info': 'Cross-validation folds for autoreject optimisation.',
        'technical': 'Higher = more robust estimate but slower. 5 is standard; reduce to 3 for large datasets.',
        'unit': 'integer',
    },
    'tmin': {
        'info': 'Epoch start time relative to stimulus onset.',
        'technical': 'Negative values include pre-stimulus baseline. -0.2s provides 200ms baseline for ERP.',
        'unit': 'seconds',
    },
    'tmax': {
        'info': 'Epoch end time relative to stimulus onset.',
        'technical': 'Must capture the component of interest. 0.8s captures P3b (300-500ms) with margin.',
        'unit': 'seconds',
    },
    'baseline': {
        'info': 'Time window for baseline correction.',
        'technical': '[-0.2, 0.0] subtracts mean of 200ms pre-stimulus. Set to null to skip baseline correction.',
        'unit': '[start, end] seconds',
    },
    'stim_duration_offset': {
        'info': 'Stimulus duration offset for PAC maintenance windows.',
        'technical': 'PAC epochs start at this offset after stimulus onset, capturing the maintenance period after encoding.',
        'unit': 'seconds',
    },
    'channels': {
        'info': 'Electrode channels for this analysis.',
        'technical': 'Select specific scalp sites. Pz is canonical for P3b (Polich 2007). Posterior channels for alpha.',
        'available': ['Fp1','Fp2','F7','F3','Fz','F4','F8','FC5','FC1','FCz','FC2','FC6',
                      'T7','C3','Cz','C4','T8','CP5','CP1','CPz','CP2','CP6',
                      'P7','P3','Pz','P4','P8','PO7','PO3','POz','PO4','PO8',
                      'O1','Oz','O2'],
    },
    'channel': {
        'info': 'Single electrode channel for this analysis.',
        'technical': 'Fz is canonical for frontal midline theta.',
        'choices': ['Fz','FCz','Cz','F3','F4','FC1','FC2'],
    },
    'tmin_peak': {
        'info': 'P3b peak search window start.',
        'technical': 'Peak is searched within [tmin_peak, tmax_peak]. Standard P3b range: 300-500ms.',
        'unit': 'seconds',
    },
    'tmax_peak': {
        'info': 'P3b peak search window end.',
        'technical': 'Extend to 600ms for clinical/elderly populations with slower P3b.',
        'unit': 'seconds',
    },
    'freq_range': {
        'info': 'Frequency range for specparam model fitting.',
        'technical': '[1, 30] Hz captures theta through alpha. Extend to 50+ to include beta/gamma peaks.',
        'unit': '[low, high] Hz',
    },
    'peak_width_limits': {
        'info': 'Allowed peak widths in specparam.',
        'technical': 'Prevents fitting noise spikes (too narrow) or aperiodic features (too wide). [1, 8] Hz is standard.',
        'unit': '[min, max] Hz',
    },
    'max_n_peaks': {
        'info': 'Maximum peaks specparam can fit.',
        'technical': '6 allows theta + alpha + beta + harmonics. Reduce to force only dominant peaks.',
        'unit': 'count',
    },
    'min_peak_height': {
        'info': 'Minimum power for a peak to be detected.',
        'technical': 'In log power units above aperiodic background. 0.1 is sensitive; increase to 0.2+ to ignore weak peaks.',
        'unit': 'log power',
    },
    'aperiodic_mode': {
        'info': 'Aperiodic background model shape.',
        'technical': 'fixed: simple 1/f. knee: allows a bend at low frequencies (better for long recordings or resting state).',
        'choices': ['fixed', 'knee'],
    },
    'phase_band': {
        'info': 'Phase frequency band for PAC analysis.',
        'technical': 'Theta [4, 8] Hz is standard. Will be individualised to ITF +/- 2 Hz when available.',
        'unit': '[low, high] Hz',
    },
    'amp_band': {
        'info': 'Amplitude frequency band for PAC analysis.',
        'technical': 'High gamma [55, 85] Hz. Captures gamma bursts phase-locked to theta troughs.',
        'unit': '[low, high] Hz',
    },
    'n_surrogates': {
        'info': 'Number of surrogate permutations for PAC statistics.',
        'technical': 'More surrogates = more precise null distribution. 200 is minimum for publication; 500+ preferred.',
        'unit': 'count',
    },
    'trim': {
        'info': 'Trim proportion for surrogate distribution.',
        'technical': 'Removes extreme surrogate values before z-scoring. 0.10 = trim top/bottom 10%.',
        'unit': 'proportion',
    },
    'search_range': {
        'info': 'Frequency range to search for the spectral peak.',
        'technical': 'IAF: [6, 14] Hz. ITF: [3, 9] Hz. Specparam finds the strongest peak within this range.',
        'unit': '[low, high] Hz',
    },
    'welch_win': {
        'info': 'Welch PSD window length.',
        'technical': 'Longer windows give better frequency resolution. 4s gives 0.25 Hz resolution.',
        'unit': 'seconds',
    },
    'theta_default': {
        'info': 'Default theta band when individualised peak unavailable.',
        'technical': 'Canonical theta: [4, 8] Hz. Used as fallback when specparam cannot detect a theta peak.',
        'unit': '[low, high] Hz',
    },
    'alpha_default': {
        'info': 'Default alpha band when individualised peak unavailable.',
        'technical': 'Canonical alpha: [8, 13] Hz. Used as fallback when specparam cannot detect an alpha peak.',
        'unit': '[low, high] Hz',
    },
    'individualized_bandwidth': {
        'info': 'Half-width for individualised frequency bands.',
        'technical': 'Band = peak +/- this value. 2.0 Hz gives 4 Hz wide bands centered on individual peak.',
        'unit': 'Hz',
    },
    'theta_cf_channels': {
        'info': 'Channels for frontal midline theta at CF node.',
        'technical': 'Fz, FCz, Cz capture FM-theta. Used for band power analysis and ITF extraction.',
        'available': ['Fz','FCz','Cz','F3','F4','FC1','FC2','FC5','FC6'],
    },
    'alpha_posterior_channels': {
        'info': 'Channels for posterior alpha analysis.',
        'technical': 'Oz, O1, O2, POz, PO3, PO4, PO7, PO8. Captures dominant posterior alpha rhythm.',
        'available': ['Oz','O1','O2','POz','PO3','PO4','PO7','PO8','Pz','P3','P4','P7','P8'],
    },
    'max_bad_channels_pct': {
        'info': 'Maximum percentage of bad channels allowed.',
        'technical': 'If >10% of EEG channels are bad, data quality may be too poor for reliable analysis.',
        'unit': '%',
    },
    'max_ica_components_rejected': {
        'info': 'Maximum ICA components that should be rejected.',
        'technical': 'Rejecting >8 components suggests noisy data or over-sensitive thresholds.',
        'unit': 'count',
    },
    'min_brain_ics_remaining': {
        'info': 'Minimum brain ICs that must remain after rejection.',
        'technical': 'If <15 brain ICs remain, too much signal may have been removed.',
        'unit': 'count',
    },
    'max_asr_modified_pct': {
        'info': 'Maximum data modified by ASR.',
        'technical': '>30% suggests very noisy data or too aggressive cutoff. Consider raising cutoff.',
        'unit': '%',
    },
    'max_epoch_rejection_pct': {
        'info': 'Maximum epoch rejection rate.',
        'technical': '>30% means too many bad epochs. Check preprocessing or widen autoreject consensus.',
        'unit': '%',
    },
    'min_epochs_per_condition': {
        'info': 'Minimum epochs needed per experimental condition.',
        'technical': '<20 epochs may produce unreliable ERPs. Consider relaxing rejection criteria.',
        'unit': 'count',
    },
    'p3b_latency_range_ms': {
        'info': 'Expected P3b peak latency range.',
        'technical': 'Healthy adults: 300-500ms. Extend to [250, 600] for clinical or elderly populations.',
        'unit': '[min, max] ms',
    },
    'p3b_min_amplitude_uv': {
        'info': 'Minimum expected P3b amplitude.',
        'technical': 'Values <0.5 uV may indicate poor signal quality or incorrect electrode selection.',
        'unit': 'uV',
    },
}


def params_hash(data):
    """Compute short hash of parameters dict."""
    raw = json.dumps(data, sort_keys=True).encode()
    return hashlib.md5(raw).hexdigest()[:8]


# ============================================================
# TOOLTIP (rich version)
# ============================================================
class RichToolTip:
    """Two-part tooltip: Info line + Technical detail."""
    SHOW_DELAY_MS = 300
    HIDE_DELAY_MS = 80
    OFFSET_X = 18
    OFFSET_Y = 24
    SCREEN_PAD = 12

    def __init__(self, widget, info, technical='', unit=''):
        self.widget = widget
        self.info = info
        self.technical = technical
        self.unit = unit
        self.tw = None
        self._show_job = None
        self._hide_job = None
        self._pointer_x = 0
        self._pointer_y = 0

        # Use add='+' so we do not overwrite widget-specific hover handlers.
        widget.bind('<Enter>', self._on_enter, add='+')
        widget.bind('<Leave>', self._on_leave, add='+')
        widget.bind('<Motion>', self._on_motion, add='+')
        widget.bind('<ButtonPress>', self._hide_now, add='+')
        widget.bind('<Destroy>', self._on_destroy, add='+')

    @staticmethod
    def _is_descendant(child, ancestor):
        while child is not None:
            if child == ancestor:
                return True
            child = getattr(child, 'master', None)
        return False

    def _pointer_over_target(self):
        try:
            px, py = self.widget.winfo_pointerxy()
            hovered = self.widget.winfo_containing(px, py)
        except tk.TclError:
            return False
        if hovered is None:
            return False
        if self._is_descendant(hovered, self.widget):
            return True
        if self.tw and self._is_descendant(hovered, self.tw):
            return True
        return False

    def _on_enter(self, event=None):
        self._cancel_hide()
        self._schedule_show()

    def _on_motion(self, event=None):
        if event is not None:
            self._pointer_x = event.x_root
            self._pointer_y = event.y_root
        if self.tw:
            self._position_tooltip()

    def _on_leave(self, event=None):
        self._cancel_show()
        self._schedule_hide()

    def _schedule_show(self):
        if self.tw or self._show_job is not None:
            return
        self._show_job = self.widget.after(self.SHOW_DELAY_MS, self._show_now)

    def _schedule_hide(self):
        if self._hide_job is not None:
            return
        self._hide_job = self.widget.after(self.HIDE_DELAY_MS, self._maybe_hide)

    def _cancel_show(self):
        if self._show_job is not None:
            try:
                self.widget.after_cancel(self._show_job)
            except tk.TclError:
                pass
            self._show_job = None

    def _cancel_hide(self):
        if self._hide_job is not None:
            try:
                self.widget.after_cancel(self._hide_job)
            except tk.TclError:
                pass
            self._hide_job = None

    def _show_now(self):
        self._show_job = None
        if self.tw or not self._pointer_over_target():
            return

        self.tw = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.configure(bg=C['border_glass'])
        try:
            tw.wm_attributes('-topmost', True)
        except tk.TclError:
            pass

        inner = tk.Frame(tw, bg=C['bg_panel_alt'], padx=1, pady=1)
        inner.pack(fill='both', expand=True, padx=1, pady=1)

        info_text = self.info if self.info else "No description available."
        info_lbl = tk.Label(inner, text=info_text, justify='left',
                            bg=C['bg_panel_alt'], fg=C['text_primary'],
                            font=('Segoe UI', 9, 'bold'), padx=10, pady=0,
                            wraplength=380, anchor='w')
        info_lbl.pack(fill='x', pady=(6, 2))

        if self.technical:
            tech_lbl = tk.Label(inner, text=self.technical, justify='left',
                                bg=C['bg_panel_alt'], fg=C['text_primary'],
                                font=('Segoe UI', 9), padx=10, pady=0,
                                wraplength=380, anchor='w')
            tech_lbl.pack(fill='x', pady=(0, 4))

        if self.unit:
            tk.Label(inner, text=f"Unit: {self.unit}",
                     bg=C['bg_deep'], fg=C['text_dim'],
                     font=('Segoe UI', 7), padx=8, pady=2).pack(
                         anchor='w', padx=8, pady=(0, 6))

        tw.bind('<Enter>', lambda e: self._cancel_hide(), add='+')
        tw.bind('<Leave>', lambda e: self._schedule_hide(), add='+')
        tw.bind('<ButtonPress>', self._hide_now, add='+')
        self._position_tooltip()

    def _position_tooltip(self):
        if not self.tw:
            return
        try:
            x = self._pointer_x or self.widget.winfo_rootx()
            y = self._pointer_y or self.widget.winfo_rooty()
            x += self.OFFSET_X
            y += self.OFFSET_Y

            self.tw.update_idletasks()
            width = self.tw.winfo_reqwidth()
            height = self.tw.winfo_reqheight()
            screen_w = self.tw.winfo_screenwidth()
            screen_h = self.tw.winfo_screenheight()

            if x + width + self.SCREEN_PAD > screen_w:
                x = max(self.SCREEN_PAD, screen_w - width - self.SCREEN_PAD)
            if y + height + self.SCREEN_PAD > screen_h:
                y = max(self.SCREEN_PAD, y - height - self.OFFSET_Y - 8)
            if y + height + self.SCREEN_PAD > screen_h:
                y = max(self.SCREEN_PAD, screen_h - height - self.SCREEN_PAD)

            self.tw.wm_geometry(f"+{x}+{y}")
        except tk.TclError:
            self._hide_now()

    def _maybe_hide(self):
        self._hide_job = None
        if self._pointer_over_target():
            return
        self._hide_now()

    def _hide_now(self, event=None):
        self._cancel_show()
        self._cancel_hide()
        if self.tw:
            self.tw.destroy()
            self.tw = None

    def _on_destroy(self, event=None):
        self._hide_now()


# ============================================================
# TAG EDITOR WIDGET (for list values like channels)
# ============================================================
class TagEditor(tk.Frame):
    """Visual tag editor for list values (channels, numeric lists)."""

    def __init__(self, parent, values, available=None, **kwargs):
        super().__init__(parent, bg=C['bg_entry'], highlightthickness=1,
                         highlightbackground=C['border_glass'],
                         highlightcolor=C['accent_blue'], **kwargs)
        self.values = list(values) if values else []
        self.available = available  # If set, shows dropdown
        self.on_change = None

        self.tags_frame = tk.Frame(self, bg=C['bg_entry'])
        self.tags_frame.pack(fill='x', padx=4, pady=4)

        # Add button / entry row
        bottom = tk.Frame(self, bg=C['bg_entry'])
        bottom.pack(fill='x', padx=4, pady=(0, 4))

        if available:
            self.combo_var = tk.StringVar()
            self.combo = ttk.Combobox(bottom, textvariable=self.combo_var,
                                       values=available, width=12,
                                       state='readonly',
                                       style='Dark.TCombobox')
            self.combo.pack(side='left', padx=(0, 4))
            add_btn = tk.Button(bottom, text='+', command=self._add_from_combo,
                                bg=C['accent_green'], fg=C['bg_deep'],
                                font=('Segoe UI', 8, 'bold'),
                                relief='flat', padx=6, pady=0, cursor='hand2')
            add_btn.pack(side='left')
        else:
            self.entry_var = tk.StringVar()
            self.entry = tk.Entry(bottom, textvariable=self.entry_var,
                                   bg=C['bg_entry_focus'], fg=C['text_primary'],
                                   insertbackground=C['accent_cyan'],
                                   font=('Consolas', 9), width=10,
                                   relief='flat', bd=0)
            self.entry.pack(side='left', padx=(0, 4))
            self.entry.bind('<Return>', lambda e: self._add_from_entry())
            add_btn = tk.Button(bottom, text='+', command=self._add_from_entry,
                                bg=C['accent_green'], fg=C['bg_deep'],
                                font=('Segoe UI', 8, 'bold'),
                                relief='flat', padx=6, pady=0, cursor='hand2')
            add_btn.pack(side='left')

        self._render_tags()

    def _render_tags(self):
        for w in self.tags_frame.winfo_children():
            w.destroy()

        for i, val in enumerate(self.values):
            tag = tk.Frame(self.tags_frame, bg=C['tag_bg'],
                           highlightbackground=C['tag_border'],
                           highlightthickness=1)
            tag.pack(side='left', padx=2, pady=2)

            tk.Label(tag, text=str(val), bg=C['tag_bg'],
                     fg=C['accent_cyan'], font=('Consolas', 9),
                     padx=4, pady=1).pack(side='left')

            rm_btn = tk.Button(tag, text='x', command=lambda idx=i: self._remove(idx),
                                bg=C['tag_bg'], fg=C['accent_red'],
                                font=('Segoe UI', 7, 'bold'),
                                relief='flat', padx=2, pady=0, cursor='hand2',
                                bd=0)
            rm_btn.pack(side='left')

    def _add_from_combo(self):
        val = self.combo_var.get()
        if val and val not in self.values:
            self.values.append(val)
            self._render_tags()
            self._fire_change()

    def _add_from_entry(self):
        raw = self.entry_var.get().strip()
        if not raw:
            return
        # Try to parse as number
        try:
            val = int(raw)
        except ValueError:
            try:
                val = float(raw)
            except ValueError:
                val = raw
        # Allow adding (including re-adding after deletion)
        self.values.append(val)
        self.entry_var.set('')
        self._render_tags()
        self._fire_change()

    def _remove(self, idx):
        self.values.pop(idx)
        self._render_tags()
        self._fire_change()

    def _fire_change(self):
        if self.on_change:
            self.on_change()

    def get(self):
        return list(self.values)


# ============================================================
# MAIN GUI
# ============================================================
class ParameterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("EEG Pipeline -- Parameter Editor")
        self.root.configure(bg=C['bg_deep'])
        self.root.geometry("1020x760")
        self.root.minsize(860, 600)

        # Try DPI awareness
        try:
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass

        # --- Dark-theme styling for ttk Combobox widgets ---
        style = ttk.Style()
        style.theme_use('clam')  # clam supports full colour customisation
        style.configure('Dark.TCombobox',
                        fieldbackground=C['bg_entry'],
                        background=C['bg_panel_alt'],
                        foreground=C['text_primary'],
                        arrowcolor=C['accent_cyan'],
                        bordercolor=C['border_glass'],
                        lightcolor=C['border_glass'],
                        darkcolor=C['border_glass'],
                        selectbackground=C['accent_blue'],
                        selectforeground=C['text_primary'])
        style.map('Dark.TCombobox',
                  fieldbackground=[('readonly', C['bg_entry']),
                                   ('focus', C['bg_entry_focus'])],
                  foreground=[('readonly', C['text_primary'])],
                  selectbackground=[('readonly', C['accent_blue'])],
                  selectforeground=[('readonly', C['text_primary'])])
        # Dropdown listbox colours (set via option database)
        self.root.option_add('*TCombobox*Listbox.background', C['bg_panel'])
        self.root.option_add('*TCombobox*Listbox.foreground', C['text_primary'])
        self.root.option_add('*TCombobox*Listbox.selectBackground', C['accent_blue'])
        self.root.option_add('*TCombobox*Listbox.selectForeground', C['text_primary'])

        # Load params
        self.params_path = PARAMS_PATH
        self.load_params()
        self.modified = False
        self.widgets = {}  # key -> (getter, setter, widget)
        self.nav_buttons = {}  # section key -> nav widgets/state
        self.active_section_key = None

        # Fonts
        self.font_title = tkfont.Font(family='Segoe UI', size=16, weight='bold')
        self.font_section = tkfont.Font(family='Segoe UI', size=12, weight='bold')
        self.font_label = tkfont.Font(family='Segoe UI', size=10)
        self.font_entry = tkfont.Font(family='Consolas', size=10)
        self.font_small = tkfont.Font(family='Segoe UI', size=8)
        self.font_btn = tkfont.Font(family='Segoe UI', size=10, weight='bold')

        self._build_ui()

    def load_params(self):
        with open(self.params_path, 'r', encoding='utf-8') as f:
            self.params = json.load(f)
        self.original_hash = params_hash(self.params)

    def _attach_marble_texture(self, parent, base_color, vein_count=7):
        """Add a subtle black-marble texture behind widgets in a frame."""
        texture = tk.Canvas(parent, bg=base_color, highlightthickness=0, bd=0)
        texture.place(relx=0, rely=0, relwidth=1, relheight=1)
        texture.bind(
            '<Configure>',
            lambda e, c=texture, n=vein_count: self._paint_marble_texture(c, n),
            add='+'
        )
        self._paint_marble_texture(texture, vein_count)
        return texture

    def _paint_marble_texture(self, canvas, vein_count):
        canvas.delete('marble')
        w = max(1, canvas.winfo_width())
        h = max(1, canvas.winfo_height())
        if w < 10 or h < 10:
            return

        canvas.create_rectangle(0, 0, w, h, fill=canvas.cget('bg'),
                                outline='', tags='marble')
        sheen_h = max(1, int(h * 0.22))
        canvas.create_rectangle(0, 0, w, sheen_h, fill=C['marble_sheen'],
                                outline='', tags='marble')

        for i in range(vein_count):
            amplitude = max(2, h * (0.05 + (i % 3) * 0.015))
            base_y = (h / (vein_count + 1)) * (i + 1)
            phase = i * 0.9
            points = []
            for x in range(-40, w + 41, 48):
                y = base_y + math.sin((x * 0.014) + phase) * amplitude
                points.extend((x, y))

            vein_color = C['marble_vein'] if i % 2 == 0 else C['marble_vein_soft']
            width = 1
            canvas.create_line(*points, fill=vein_color, width=width,
                               smooth=True, splinesteps=20, tags='marble')

    # ----------------------------------------------------------
    # UI CONSTRUCTION
    # ----------------------------------------------------------
    def _build_ui(self):
        # --- Title bar ---
        title_frame = tk.Frame(self.root, bg=C['bg_deep'], height=64)
        title_frame.pack(fill='x')
        title_frame.pack_propagate(False)
        self._attach_marble_texture(title_frame, C['bg_deep'], vein_count=8)

        tk.Frame(title_frame, bg=C['marble_vein'], height=2).pack(fill='x', side='bottom')

        tk.Label(title_frame, text="EEG Pipeline Parameters",
                 font=self.font_title, bg=C['bg_deep'],
                 fg=C['accent_cyan']).pack(side='left', padx=20, pady=14)

        # Hash badge (clickable for explanation)
        hash_frame = tk.Frame(title_frame, bg=C['bg_panel'],
                               highlightbackground=C['border_glass'], highlightthickness=1)
        hash_frame.pack(side='right', padx=20, pady=14)

        self.hash_label = tk.Label(hash_frame, text=f"# {self.original_hash}",
                                   font=('Consolas', 9), bg=C['bg_panel'],
                                   fg=C['accent_purple'], padx=10, pady=4,
                                   cursor='hand2')
        self.hash_label.pack()
        RichToolTip(self.hash_label,
                    info="Parameter Hash -- unique fingerprint of your current settings.",
                    technical="This MD5 hash changes whenever any parameter is modified. "
                              "It is recorded in every QC assessment report, linking results "
                              "to the exact parameter configuration used. This ensures full "
                              "reproducibility: re-running with the same hash guarantees identical processing.",
                    unit="MD5 (first 8 chars)")

        # --- Main: sidebar + scrollable content ---
        main = tk.Frame(self.root, bg=C['bg_deep'])
        main.pack(fill='both', expand=True)

        # Sidebar
        self.sidebar = tk.Frame(main, bg=C['bg_dark'], width=190)
        self.sidebar.pack(side='left', fill='y')
        self.sidebar.pack_propagate(False)
        self._attach_marble_texture(self.sidebar, C['bg_dark'], vein_count=10)

        tk.Label(self.sidebar, text="SECTIONS",
                 font=self.font_small, bg=C['bg_dark'],
                 fg=C['text_secondary']).pack(pady=(16, 8), padx=16, anchor='w')

        tk.Frame(main, bg=C['border_glass'], width=1).pack(side='left', fill='y')

        # Scrollable content
        self.canvas = tk.Canvas(main, bg=C['bg_deep'], highlightthickness=0)
        scrollbar = tk.Scrollbar(main, orient='vertical', command=self.canvas.yview)
        self.scroll_frame = tk.Frame(self.canvas, bg=C['bg_deep'])

        self.scroll_frame.bind('<Configure>',
                               lambda e: self.canvas.configure(scrollregion=self.canvas.bbox('all')))
        self.canvas.create_window((0, 0), window=self.scroll_frame, anchor='nw')
        self.canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')
        self.canvas.pack(side='left', fill='both', expand=True)

        # Mousewheel
        self.canvas.bind_all('<MouseWheel>',
                             lambda e: self.canvas.yview_scroll(-1 * (e.delta // 120), 'units'))

        # Build sections
        self.section_frames = {}
        first_section = None
        for section_key in self.params:
            if section_key.startswith('_'):
                continue
            if first_section is None:
                first_section = section_key
            self._add_section(section_key)
        if first_section:
            self._set_active_section(first_section)

        # --- Bottom bar ---
        btn_bar = tk.Frame(self.root, bg=C['bg_dark'], height=56)
        btn_bar.pack(fill='x', side='bottom')
        btn_bar.pack_propagate(False)
        self._attach_marble_texture(btn_bar, C['bg_dark'], vein_count=7)
        tk.Frame(btn_bar, bg=C['marble_vein'], height=1).pack(fill='x', side='top')

        self.status_var = tk.StringVar(value="Ready")
        self.status_label = tk.Label(btn_bar, textvariable=self.status_var,
                                     font=self.font_small, bg=C['bg_dark'],
                                     fg=C['text_secondary'])
        self.status_label.pack(side='left', padx=20, pady=16)

        # Buttons (right to left)
        self._btn(btn_bar, "Run Pipeline", self._open_runner, C['accent_cyan']).pack(side='left', padx=10, pady=12)
        self._btn(btn_bar, "Revert", self._revert, C['accent_red']).pack(side='right', padx=6, pady=12)
        self._btn(btn_bar, "Save", self._save, C['accent_green']).pack(side='right', padx=6, pady=12)
        self._btn(btn_bar, "Load Preset", self._load_preset, C['accent_blue']).pack(side='right', padx=6, pady=12)
        self._btn(btn_bar, "Save Preset", self._save_preset, C['accent_purple']).pack(side='right', padx=6, pady=12)

    def _btn(self, parent, text, command, color):
        btn = tk.Button(parent, text=text, command=command, font=self.font_btn,
                        bg=color, fg=C['bg_deep'], activebackground=color,
                        relief='flat', padx=16, pady=5, cursor='hand2', bd=0)
        btn.bind('<Enter>', lambda e: btn.configure(bg=C['text_primary']))
        btn.bind('<Leave>', lambda e: btn.configure(bg=color))
        return btn

    # ----------------------------------------------------------
    # SECTION BUILDER
    # ----------------------------------------------------------
    def _add_section(self, key):
        meta = SECTION_META.get(key, {
            'label': key.replace('_', ' ').title(),
            'color': C['accent_blue'],
            'info': '', 'technical': '',
        })

        # Sidebar nav
        nav_row = tk.Frame(self.sidebar, bg=C['bg_dark'])
        nav_row.pack(fill='x', padx=4, pady=1)

        indicator = tk.Frame(nav_row, width=4, bg=C['bg_dark'])
        indicator.pack(side='left', fill='y')

        nav = tk.Button(nav_row, text=f"  {meta['label']}",
                        font=self.font_label, bg=C['bg_dark'], fg=C['text_secondary'],
                        activebackground=C['nav_hover_bg'], activeforeground=C['text_primary'],
                        relief='flat', anchor='w', padx=14, pady=5,
                        cursor='hand2', bd=0,
                        command=lambda k=key: self._scroll_to(k))
        nav.pack(side='left', fill='x', expand=True)
        nav.bind('<Enter>', lambda e, k=key: self._set_nav_hover(k, True), add='+')
        nav.bind('<Leave>', lambda e, k=key: self._set_nav_hover(k, False), add='+')

        self.nav_buttons[key] = {
            'row': nav_row,
            'indicator': indicator,
            'button': nav,
            'hover': False,
        }
        self._refresh_nav_style(key)

        # Content panel
        section = tk.Frame(self.scroll_frame, bg=C['bg_deep'])
        section.pack(fill='x', padx=20, pady=(14, 4))
        self.section_frames[key] = section

        glass = tk.Frame(section, bg=C['bg_panel'],
                         highlightbackground=C['border_glass'],
                         highlightthickness=1)
        glass.pack(fill='x')
        self._attach_marble_texture(glass, C['bg_panel'], vein_count=6)

        # Header
        header = tk.Frame(glass, bg=C['bg_panel'])
        header.pack(fill='x')
        tk.Frame(header, bg=meta['color'], width=4, height=36).pack(side='left', fill='y')
        tk.Label(header, text=f"  {meta['label']}",
                 font=self.font_section, bg=C['bg_panel'],
                 fg=meta['color']).pack(side='left', padx=8, pady=10)

        # Section info/technical
        if meta.get('info'):
            info_frame = tk.Frame(glass, bg=C['bg_panel'])
            info_frame.pack(fill='x', padx=20, pady=(2, 0))
            tk.Label(info_frame, text=meta['info'], font=('Segoe UI', 9),
                     bg=C['bg_panel'], fg=C['text_primary'],
                     wraplength=600, justify='left', anchor='w').pack(fill='x')
        if meta.get('technical'):
            tech_frame = tk.Frame(glass, bg=C['bg_panel'])
            tech_frame.pack(fill='x', padx=20, pady=(0, 4))
            tk.Label(tech_frame, text=f"Technical: {meta['technical']}",
                     font=('Segoe UI', 9),
                     bg=C['bg_panel'], fg=C['text_primary'],
                     wraplength=600, justify='left', anchor='w').pack(fill='x')

        tk.Frame(glass, bg=C['marble_vein'], height=1).pack(fill='x', padx=16)

        # Fields
        fields = tk.Frame(glass, bg=C['bg_panel'])
        fields.pack(fill='x', padx=20, pady=(8, 16))
        self._build_fields(fields, self.params[key], prefix=key)

    # ----------------------------------------------------------
    # FIELD BUILDER (recursive)
    # ----------------------------------------------------------
    def _build_fields(self, parent, data, prefix='', depth=0):
        if not isinstance(data, dict):
            return

        row = 0
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            help_data = FIELD_HELP.get(key, {})

            if isinstance(value, dict):
                sub = tk.Label(parent, text=f"  {key.replace('_', ' ').title()}",
                               font=self.font_label, bg=C['bg_panel'],
                               fg=C['accent_purple'])
                sub.grid(row=row, column=0, columnspan=2, sticky='w',
                        padx=(depth * 16, 0), pady=(8, 2))
                if help_data:
                    RichToolTip(sub, help_data.get('info', ''),
                                help_data.get('technical', ''),
                                help_data.get('unit', ''))
                row += 1
                sf = tk.Frame(parent, bg=C['bg_panel'])
                sf.grid(row=row, column=0, columnspan=2, sticky='ew',
                       padx=(depth * 16 + 12, 0))
                self._build_fields(sf, value, prefix=full_key, depth=depth + 1)
                row += 1
                continue

            # Label
            lbl = tk.Label(parent, text=key.replace('_', ' '),
                           font=self.font_label, bg=C['bg_panel'],
                           fg=C['text_primary'])
            lbl.grid(row=row, column=0, sticky='w', padx=(depth * 16, 12), pady=4)

            # Choose widget type
            widget = self._make_widget(parent, full_key, key, value, help_data)
            widget.grid(row=row, column=1, sticky='ew', padx=(0, 8), pady=4)

            # Attach tooltip to BOTH label and widget
            if help_data:
                tip_args = (help_data.get('info', key),
                            help_data.get('technical', ''),
                            help_data.get('unit', ''))
                RichToolTip(lbl, *tip_args)
                RichToolTip(widget, *tip_args)
            row += 1

        parent.columnconfigure(1, weight=1)

    def _make_widget(self, parent, full_key, key, value, help_data):
        """Create the appropriate widget for a parameter value."""

        # --- Dropdown for enum/choices ---
        if 'choices' in help_data:
            var = tk.StringVar(value=str(value))
            combo = ttk.Combobox(parent, textvariable=var,
                                  values=help_data['choices'],
                                  state='readonly', width=20,
                                  style='Dark.TCombobox')
            combo.bind('<<ComboboxSelected>>', lambda e: self._mark_modified())
            self.widgets[full_key] = (lambda: var.get(),
                                       lambda v: var.set(str(v)),
                                       combo, type(value).__name__)
            return combo

        # --- Tag editor for lists with available options ---
        if isinstance(value, list) and 'available' in help_data:
            editor = TagEditor(parent, value, available=help_data['available'])
            editor.on_change = self._mark_modified
            self.widgets[full_key] = (lambda e=editor: e.get(),
                                       lambda v, e=editor: self._reset_tags(e, v),
                                       editor, 'list')
            return editor

        # --- Tag editor for numeric/string lists ---
        if isinstance(value, list):
            editor = TagEditor(parent, value, available=None)
            editor.on_change = self._mark_modified
            self.widgets[full_key] = (lambda e=editor: e.get(),
                                       lambda v, e=editor: self._reset_tags(e, v),
                                       editor, 'list')
            return editor

        # --- Standard text entry ---
        var = tk.StringVar(value=self._fmt(value))
        entry = tk.Entry(parent, textvariable=var, font=self.font_entry,
                         bg=C['bg_entry'], fg=C['text_primary'],
                         insertbackground=C['accent_cyan'],
                         selectbackground=C['accent_blue'],
                         relief='flat', bd=0, highlightthickness=1,
                         highlightbackground=C['border_glass'],
                         highlightcolor=C['accent_blue'], width=28)
        var.trace_add('write', lambda *a: self._mark_modified())
        entry.bind('<FocusIn>', lambda e: entry.configure(bg=C['bg_entry_focus']))
        entry.bind('<FocusOut>', lambda e: entry.configure(bg=C['bg_entry']))
        self.widgets[full_key] = (lambda: var.get(),
                                   lambda v: var.set(self._fmt(v)),
                                   entry, type(value).__name__)
        return entry

    @staticmethod
    def _reset_tags(editor, values):
        editor.values = list(values) if values else []
        editor._render_tags()

    @staticmethod
    def _fmt(value):
        if value is None:
            return "null"
        if isinstance(value, list):
            return json.dumps(value)
        if isinstance(value, bool):
            return str(value).lower()
        return str(value)

    def _parse(self, text, orig_type):
        text = text.strip()
        if text in ('null', 'None', ''):
            return None
        if orig_type == 'list':
            return json.loads(text)
        if orig_type == 'int':
            return int(float(text))
        if orig_type == 'float':
            return float(text)
        if orig_type == 'bool':
            return text.lower() in ('true', '1', 'yes')
        return text

    def _mark_modified(self):
        self.modified = True
        self.status_var.set("Modified (unsaved)")
        self.status_label.configure(fg=C['accent_gold'])

    def _set_nav_hover(self, key, is_hover):
        nav_data = self.nav_buttons.get(key)
        if not nav_data:
            return
        nav_data['hover'] = bool(is_hover)
        self._refresh_nav_style(key)

    def _refresh_nav_style(self, key):
        nav_data = self.nav_buttons.get(key)
        if not nav_data:
            return

        active = (key == self.active_section_key)
        hover = nav_data.get('hover', False)

        if active:
            row_bg = C['nav_active_bg']
            btn_bg = C['nav_active_bg']
            fg = C['text_primary']
            indicator_bg = C['accent_cyan']
        elif hover:
            row_bg = C['nav_hover_bg']
            btn_bg = C['nav_hover_bg']
            fg = C['text_primary']
            indicator_bg = C['nav_indicator_hover']
        else:
            row_bg = C['bg_dark']
            btn_bg = C['bg_dark']
            fg = C['text_secondary']
            indicator_bg = C['bg_dark']

        nav_data['row'].configure(bg=row_bg)
        nav_data['indicator'].configure(bg=indicator_bg)
        nav_data['button'].configure(bg=btn_bg, fg=fg)

    def _set_active_section(self, key):
        if key not in self.nav_buttons:
            return
        old_key = self.active_section_key
        self.active_section_key = key
        if old_key and old_key in self.nav_buttons and old_key != key:
            self._refresh_nav_style(old_key)
        self._refresh_nav_style(key)

    def _scroll_to(self, key):
        self._set_active_section(key)
        if key in self.section_frames:
            frame = self.section_frames[key]
            self.root.update_idletasks()
            y = frame.winfo_y()
            sr = self.canvas.bbox('all')
            if sr and sr[3] > self.canvas.winfo_height():
                self.canvas.yview_moveto(y / sr[3])

    # ----------------------------------------------------------
    # COLLECT / SAVE / REVERT
    # ----------------------------------------------------------
    def _collect(self):
        result = json.loads(json.dumps(self.params))
        for full_key, (getter, setter, widget, orig_type) in self.widgets.items():
            parts = full_key.split('.')
            d = result
            for p in parts[:-1]:
                d = d[p]
            try:
                raw_val = getter()
                if orig_type == 'list':
                    d[parts[-1]] = raw_val  # TagEditor returns list directly
                else:
                    d[parts[-1]] = self._parse(raw_val, orig_type)
            except Exception as e:
                raise ValueError(f"Invalid value for '{full_key}': {getter()}") from e
        return result

    def _save(self):
        try:
            new_params = self._collect()
        except ValueError as e:
            messagebox.showerror("Validation Error", str(e))
            return

        # Backup
        backup_dir = PIPELINE_DIR / "config" / "backups"
        backup_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = backup_dir / f"parameters_{ts}.json"
        with open(self.params_path, 'r', encoding='utf-8') as f:
            old = f.read()
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(old)

        # Write new
        with open(self.params_path, 'w', encoding='utf-8') as f:
            json.dump(new_params, f, indent=4, ensure_ascii=False)

        self.params = new_params
        new_hash = params_hash(new_params)
        self.original_hash = new_hash
        self.hash_label.configure(text=f"# {new_hash}")
        self.modified = False
        self.status_var.set(
            f"Saved at {datetime.now().strftime('%H:%M:%S')} | "
            f"Backup: {backup_path.name} | Hash: {new_hash}"
        )
        self.status_label.configure(fg=C['accent_green'])

    def _revert(self):
        if self.modified:
            if not messagebox.askyesno("Revert", "Discard all unsaved changes?"):
                return
        self.load_params()
        for full_key, (getter, setter, widget, orig_type) in self.widgets.items():
            parts = full_key.split('.')
            d = self.params
            try:
                for p in parts:
                    d = d[p]
                setter(d)
            except (KeyError, TypeError):
                pass
        self.modified = False
        self.hash_label.configure(text=f"# {self.original_hash}")
        self.status_var.set("Reverted to saved state")
        self.status_label.configure(fg=C['accent_cyan'])

    def _save_preset(self):
        try:
            current = self._collect()
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            return
        path = filedialog.asksaveasfilename(
            initialdir=str(PRESETS_DIR),
            title="Save Parameter Preset",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
        )
        if path:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(current, f, indent=4, ensure_ascii=False)
            self.status_var.set(f"Preset saved: {Path(path).name}")
            self.status_label.configure(fg=C['accent_purple'])

    def _load_preset(self):
        path = filedialog.askopenfilename(
            initialdir=str(PRESETS_DIR),
            title="Load Parameter Preset",
            filetypes=[("JSON files", "*.json")],
        )
        if not path:
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                preset = json.load(f)
        except Exception as e:
            messagebox.showerror("Error", f"Could not load preset:\n{e}")
            return

        # Apply preset values to widgets
        for full_key, (getter, setter, widget, orig_type) in self.widgets.items():
            parts = full_key.split('.')
            d = preset
            try:
                for p in parts:
                    d = d[p]
                setter(d)
            except (KeyError, TypeError):
                pass

        self._mark_modified()
        self.status_var.set(f"Loaded preset: {Path(path).name} (unsaved)")
        self.status_label.configure(fg=C['accent_gold'])

    def _open_runner(self):
        runner = PIPELINE_DIR / "pipeline_runner_gui.py"
        if not runner.exists():
            messagebox.showerror("Missing Runner", f"Runner not found:\n{runner}")
            return
        try:
            subprocess.Popen([sys.executable, str(runner)], cwd=str(PIPELINE_DIR.parent))
            self.status_var.set("Opened pipeline runner")
            self.status_label.configure(fg=C['accent_cyan'])
        except Exception as e:
            messagebox.showerror("Runner Error", f"Could not open runner:\n{e}")

    def run(self):
        def on_close():
            if self.modified:
                if messagebox.askyesno("Unsaved Changes",
                                        "You have unsaved changes. Quit anyway?"):
                    self.root.destroy()
            else:
                self.root.destroy()

        self.root.protocol("WM_DELETE_WINDOW", on_close)
        self.root.mainloop()


def main():
    root = tk.Tk()
    app = ParameterGUI(root)
    app.run()


if __name__ == "__main__":
    main()
