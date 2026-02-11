"""
One-click pipeline runner GUI for EEG Study 2.

Visual style intentionally matches parameter_gui.py:
- black-marble palette
- glass-style cards
- rich tooltips
"""
from __future__ import annotations

import os
import sys
import math
import threading
import subprocess
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, font as tkfont

import yaml


PIPELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_DIR.parent
STUDY_PATH = PIPELINE_DIR / "config" / "study.yml"
PRESETS_DIR = PIPELINE_DIR / "config" / "presets"


PREPROCESS_STEPS = [
    PIPELINE_DIR / "steps" / "01_import_qc.py",
    PIPELINE_DIR / "steps" / "02_clean_reference.py",
    PIPELINE_DIR / "steps" / "04_zapline.py",
    PIPELINE_DIR / "steps" / "05_ica_iclabel.py",
    PIPELINE_DIR / "steps" / "06_asr.py",
    PIPELINE_DIR / "steps" / "07_epoch.py",
    PIPELINE_DIR / "steps" / "08_autoreject.py",
    PIPELINE_DIR / "steps" / "09_features.py",
]

ANALYSIS_STEPS = [
    PIPELINE_DIR / "analysis" / "10_erp_p3b.py",
    PIPELINE_DIR / "analysis" / "11_band_power.py",
    PIPELINE_DIR / "analysis" / "12_peak_frequencies.py",
    PIPELINE_DIR / "analysis" / "13_pac_nodal.py",
    PIPELINE_DIR / "analysis" / "16_merge_features.py",
]

VIS_STEPS = [
    PIPELINE_DIR / "analysis" / "visualise_outputs.py",
    PIPELINE_DIR / "analysis" / "visualise_group.py",
]


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
}


class RichToolTip:
    """Two-part tooltip: Info + technical detail."""
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
                            wraplength=420, anchor='w')
        info_lbl.pack(fill='x', pady=(6, 2))

        if self.technical:
            tech_lbl = tk.Label(inner, text=self.technical, justify='left',
                                bg=C['bg_panel_alt'], fg=C['text_primary'],
                                font=('Segoe UI', 9), padx=10, pady=0,
                                wraplength=420, anchor='w')
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


class PipelineRunnerGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("EEG Pipeline -- Runner")
        self.root.configure(bg=C['bg_deep'])
        self.root.geometry("1120x760")
        self.root.minsize(940, 620)

        try:
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass

        style = ttk.Style()
        style.theme_use('clam')
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
        style.configure('Runner.Horizontal.TProgressbar',
                        troughcolor=C['bg_entry'],
                        background=C['accent_cyan'],
                        bordercolor=C['border_glass'],
                        lightcolor=C['accent_cyan'],
                        darkcolor=C['accent_cyan'])
        self.root.option_add('*TCombobox*Listbox.background', C['bg_panel'])
        self.root.option_add('*TCombobox*Listbox.foreground', C['text_primary'])
        self.root.option_add('*TCombobox*Listbox.selectBackground', C['accent_blue'])
        self.root.option_add('*TCombobox*Listbox.selectForeground', C['text_primary'])

        self.mode_var = tk.StringVar(value="Full (Preprocess + Analysis)")
        self.raw_root_var = tk.StringVar(value="")
        self.pattern_var = tk.StringVar(value="")
        self.format_var = tk.StringVar(value="auto")
        self.preset_var = tk.StringVar(value="(use parameters.json)")
        self.apply_preset_var = tk.BooleanVar(value=False)
        self.include_viz_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Ready")
        self.scan_var = tk.StringVar(value="Files matched: --")

        self.current_process: subprocess.Popen[str] | None = None
        self.worker_thread: threading.Thread | None = None
        self.stop_requested = False
        self.preset_map: dict[str, Path] = {}

        self.font_title = tkfont.Font(family='Segoe UI', size=15, weight='bold')
        self.font_section = tkfont.Font(family='Segoe UI', size=11, weight='bold')
        self.font_label = tkfont.Font(family='Segoe UI', size=10)
        self.font_entry = tkfont.Font(family='Consolas', size=10)
        self.font_small = tkfont.Font(family='Segoe UI', size=8)
        self.font_btn = tkfont.Font(family='Segoe UI', size=10, weight='bold')

        self._build_ui()
        self._load_study_defaults()
        self._load_presets()

    def _attach_marble_texture(self, parent, base_color, vein_count=7):
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
            canvas.create_line(*points, fill=vein_color, width=1,
                               smooth=True, splinesteps=20, tags='marble')

    def _build_card(self, parent, title, accent=None):
        accent_color = accent or C['accent_cyan']
        outer = tk.Frame(parent, bg=C['bg_panel'],
                         highlightbackground=C['border_glass'],
                         highlightthickness=1)
        self._attach_marble_texture(outer, C['bg_panel'], vein_count=6)

        header = tk.Frame(outer, bg=C['bg_panel'])
        header.pack(fill='x')
        tk.Frame(header, bg=accent_color, width=4, height=30).pack(side='left', fill='y')
        tk.Label(header, text=f"  {title}", font=self.font_section,
                 bg=C['bg_panel'], fg=accent_color).pack(side='left', padx=8, pady=8)
        tk.Frame(outer, bg=C['marble_vein'], height=1).pack(fill='x', padx=12)

        body = tk.Frame(outer, bg=C['bg_panel'])
        body.pack(fill='both', expand=True, padx=12, pady=10)
        return outer, body

    def _entry(self, parent, var):
        e = tk.Entry(parent, textvariable=var, font=self.font_entry,
                     bg=C['bg_entry'], fg=C['text_primary'],
                     insertbackground=C['accent_cyan'],
                     selectbackground=C['accent_blue'],
                     relief='flat', bd=0, highlightthickness=1,
                     highlightbackground=C['border_glass'],
                     highlightcolor=C['accent_blue'])
        e.bind('<FocusIn>', lambda ev, w=e: w.configure(bg=C['bg_entry_focus']))
        e.bind('<FocusOut>', lambda ev, w=e: w.configure(bg=C['bg_entry']))
        return e

    def _btn(self, parent, text, command, color):
        btn = tk.Button(parent, text=text, command=command, font=self.font_btn,
                        bg=color, fg=C['bg_deep'], activebackground=color,
                        relief='flat', padx=16, pady=5, cursor='hand2', bd=0)
        btn.bind('<Enter>', lambda e, b=btn: b.configure(bg=C['text_primary']))
        btn.bind('<Leave>', lambda e, b=btn, c=color: b.configure(bg=c))
        return btn

    def _build_ui(self):
        title_frame = tk.Frame(self.root, bg=C['bg_deep'], height=64)
        title_frame.pack(fill='x')
        title_frame.pack_propagate(False)
        self._attach_marble_texture(title_frame, C['bg_deep'], vein_count=8)
        tk.Frame(title_frame, bg=C['marble_vein'], height=2).pack(fill='x', side='bottom')

        tk.Label(title_frame, text="EEG Pipeline Runner",
                 font=self.font_title, bg=C['bg_deep'],
                 fg=C['accent_cyan']).pack(side='left', padx=20, pady=14)
        tk.Label(title_frame, text="Front-to-back preprocessing + analysis",
                 font=self.font_small, bg=C['bg_deep'],
                 fg=C['text_secondary']).pack(side='left', padx=(6, 0), pady=(18, 10))
        self.open_settings_btn = self._btn(
            title_frame, "Open Settings", self._open_parameter_editor, C['accent_blue']
        )
        self.open_settings_btn.pack(side='right', padx=20, pady=14)
        RichToolTip(self.open_settings_btn,
                    info="Open parameter settings GUI.",
                    technical="Launches parameter_gui.py in a separate window so you can edit or save presets.",
                    unit="action")

        main = tk.Frame(self.root, bg=C['bg_deep'])
        main.pack(fill='both', expand=True)

        cfg_card, cfg = self._build_card(main, "Run Configuration", accent=C['accent_blue'])
        cfg_card.pack(fill='x', padx=14, pady=(12, 8))
        cfg.columnconfigure(1, weight=1)
        cfg.columnconfigure(3, weight=1)

        tk.Label(cfg, text="Mode", font=self.font_label,
                 bg=C['bg_panel'], fg=C['text_primary']).grid(
                    row=0, column=0, sticky='w', padx=(0, 8), pady=4)
        self.mode_combo = ttk.Combobox(
            cfg, textvariable=self.mode_var, state='readonly', style='Dark.TCombobox',
            values=["Preprocessing Only", "Analysis Only", "Full (Preprocess + Analysis)"],
            width=31
        )
        self.mode_combo.grid(row=0, column=1, sticky='w', pady=4)
        RichToolTip(self.mode_combo,
                    info="Select run scope.",
                    technical="Preprocessing executes steps 01-09; Analysis executes 10/11/12/13/16; "
                              "Full runs both in order.",
                    unit="mode")

        self.include_viz_chk = tk.Checkbutton(
            cfg, text="Include visualisation scripts", variable=self.include_viz_var,
            bg=C['bg_panel'], fg=C['text_primary'], activebackground=C['bg_panel'],
            activeforeground=C['text_primary'], selectcolor=C['bg_entry'],
            highlightthickness=0, bd=0, font=self.font_label, cursor='hand2'
        )
        self.include_viz_chk.grid(row=0, column=3, sticky='w', padx=(16, 0), pady=4)
        RichToolTip(self.include_viz_chk,
                    info="Append post-run plotting scripts.",
                    technical="Adds visualise_outputs.py and visualise_group.py for figure generation.",
                    unit="boolean")

        tk.Label(cfg, text="Raw Data Folder", font=self.font_label,
                 bg=C['bg_panel'], fg=C['text_primary']).grid(
                    row=1, column=0, sticky='w', padx=(0, 8), pady=4)
        self.raw_root_entry = self._entry(cfg, self.raw_root_var)
        self.raw_root_entry.grid(row=1, column=1, columnspan=2, sticky='ew', pady=4)
        RichToolTip(self.raw_root_entry,
                    info="Root folder containing raw files.",
                    technical="Used as data.root override for this run only; does not edit study.yml.",
                    unit="path")

        self.browse_btn = self._btn(cfg, "Browse", self._browse_raw_root, C['accent_blue'])
        self.browse_btn.grid(row=1, column=3, sticky='e', pady=4)
        RichToolTip(self.browse_btn,
                    info="Choose raw EEG folder.",
                    technical="Opens directory picker and fills Raw Data Folder.",
                    unit="action")

        tk.Label(cfg, text="File Pattern", font=self.font_label,
                 bg=C['bg_panel'], fg=C['text_primary']).grid(
                    row=2, column=0, sticky='w', padx=(0, 8), pady=4)
        self.pattern_entry = self._entry(cfg, self.pattern_var)
        self.pattern_entry.grid(row=2, column=1, sticky='ew', pady=4)
        RichToolTip(self.pattern_entry,
                    info="Glob pattern for input files.",
                    technical="Example: sub-*_block*-task.vhdr. Used with the selected raw root folder.",
                    unit="glob pattern")

        tk.Label(cfg, text="Format", font=self.font_label,
                 bg=C['bg_panel'], fg=C['text_primary']).grid(
                    row=2, column=2, sticky='e', padx=(12, 8), pady=4)
        self.format_combo = ttk.Combobox(
            cfg, textvariable=self.format_var, state='readonly',
            style='Dark.TCombobox', width=14,
            values=["auto", "brainvision", "edf", "fif", "bdf", "egi", "cnt", "eeglab"]
        )
        self.format_combo.grid(row=2, column=3, sticky='w', pady=4)
        RichToolTip(self.format_combo,
                    info="Force file reader format or use auto-detect.",
                    technical="Auto infers reader from extension. Override if extension is inconsistent.",
                    unit="format")

        scan_row = tk.Frame(cfg, bg=C['bg_panel'])
        scan_row.grid(row=3, column=1, columnspan=3, sticky='w', pady=(4, 2))
        self.scan_btn = self._btn(scan_row, "Scan Raw Files", self._scan_raw_files, C['accent_green'])
        self.scan_btn.pack(side='left')
        RichToolTip(self.scan_btn,
                    info="Count files matching root + pattern.",
                    technical="Performs local glob scan and previews first matches in the run log.",
                    unit="action")

        self.scan_label = tk.Label(scan_row, textvariable=self.scan_var,
                                   font=self.font_label, bg=C['bg_panel'],
                                   fg=C['text_secondary'])
        self.scan_label.pack(side='left', padx=(12, 0))

        preset_card, preset = self._build_card(main, "Preset Selection", accent=C['accent_cyan'])
        preset_card.pack(fill='x', padx=14, pady=(0, 8))
        preset.columnconfigure(1, weight=1)

        tk.Label(preset, text="Preset", font=self.font_label,
                 bg=C['bg_panel'], fg=C['text_primary']).grid(
                    row=0, column=0, sticky='w', padx=(0, 8), pady=4)
        self.preset_combo = ttk.Combobox(
            preset, textvariable=self.preset_var, state='readonly',
            style='Dark.TCombobox', width=46
        )
        self.preset_combo.grid(row=0, column=1, sticky='w', pady=4)
        RichToolTip(self.preset_combo,
                    info="Select a parameters preset JSON.",
                    technical="Only applied when 'Use selected preset for this run' is enabled.",
                    unit="json preset")

        self.apply_preset_chk = tk.Checkbutton(
            preset, text="Use selected preset for this run", variable=self.apply_preset_var,
            bg=C['bg_panel'], fg=C['text_primary'], activebackground=C['bg_panel'],
            activeforeground=C['text_primary'], selectcolor=C['bg_entry'],
            highlightthickness=0, bd=0, font=self.font_label, cursor='hand2'
        )
        self.apply_preset_chk.grid(row=0, column=2, sticky='w', padx=(16, 0), pady=4)
        RichToolTip(self.apply_preset_chk,
                    info="Run with the selected preset without overwriting parameters.json.",
                    technical="Sets EEG_PARAMETERS_PATH for subprocesses; default run uses config/parameters.json.",
                    unit="boolean")

        action_card, actions = self._build_card(main, "Execution", accent=C['accent_gold'])
        action_card.pack(fill='x', padx=14, pady=(0, 8))
        self.run_btn = self._btn(actions, "Run Pipeline", self._start_run, C['accent_green'])
        self.run_btn.pack(side='left')
        RichToolTip(self.run_btn,
                    info="Start selected pipeline steps.",
                    technical="Runs each script sequentially with live log capture and progress updates.",
                    unit="action")

        self.stop_btn = self._btn(actions, "Stop", self._stop_run, C['accent_red'])
        self.stop_btn.pack(side='left', padx=(8, 0))
        self.stop_btn.configure(state='disabled')
        RichToolTip(self.stop_btn,
                    info="Request graceful stop.",
                    technical="Terminates current subprocess and ends remaining queue.",
                    unit="action")

        self.progress = ttk.Progressbar(
            actions, mode='determinate', length=380, style='Runner.Horizontal.TProgressbar'
        )
        self.progress.pack(side='left', padx=(16, 0))
        RichToolTip(self.progress,
                    info="Pipeline progress across scripts.",
                    technical="Value increments after each script completes successfully.",
                    unit="steps completed")

        self.status_label = tk.Label(actions, textvariable=self.status_var,
                                     font=self.font_label, bg=C['bg_panel'],
                                     fg=C['text_secondary'])
        self.status_label.pack(side='left', padx=(16, 0))

        log_card = tk.Frame(main, bg=C['bg_panel'],
                            highlightbackground=C['border_glass'],
                            highlightthickness=1)
        log_card.pack(fill='both', expand=True, padx=14, pady=(0, 12))
        self._attach_marble_texture(log_card, C['bg_panel'], vein_count=6)
        header = tk.Frame(log_card, bg=C['bg_panel'])
        header.pack(fill='x')
        tk.Frame(header, bg=C['accent_blue'], width=4, height=30).pack(side='left', fill='y')
        tk.Label(header, text="  Run Log", font=self.font_section,
                 bg=C['bg_panel'], fg=C['accent_blue']).pack(side='left', padx=8, pady=8)
        tk.Frame(log_card, bg=C['marble_vein'], height=1).pack(fill='x', padx=12)

        log_wrap = tk.Frame(log_card, bg=C['bg_panel'])
        log_wrap.pack(fill='both', expand=True, padx=12, pady=10)
        log_wrap.columnconfigure(0, weight=1)
        log_wrap.rowconfigure(0, weight=1)

        self.log_text = tk.Text(log_wrap, wrap='word', state='disabled', font=self.font_entry,
                                bg=C['bg_panel_alt'], fg=C['text_primary'],
                                insertbackground=C['accent_cyan'],
                                selectbackground=C['accent_blue'],
                                relief='flat', bd=0, highlightthickness=1,
                                highlightbackground=C['border_glass'],
                                highlightcolor=C['accent_blue'])
        self.log_text.grid(row=0, column=0, sticky='nsew')
        log_scroll = tk.Scrollbar(log_wrap, orient='vertical', command=self.log_text.yview,
                                  bg=C['scrollbar_bg'], troughcolor=C['scrollbar_fg'])
        log_scroll.grid(row=0, column=1, sticky='ns')
        self.log_text.configure(yscrollcommand=log_scroll.set)
        RichToolTip(self.log_text,
                    info="Live stdout/stderr from each pipeline script.",
                    technical="If a step fails, the last lines here are the primary debugging source.",
                    unit="log stream")

    def _load_study_defaults(self):
        try:
            with open(STUDY_PATH, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f) or {}
        except Exception as exc:
            self._append_log(f"Failed to read study.yml: {exc}")
            return

        data_cfg = cfg.get('data', {})
        self.raw_root_var.set(str(data_cfg.get('root', '')))
        self.pattern_var.set(str(data_cfg.get('pattern', 'sub-*_block*-task.vhdr')))
        self.format_var.set(str(data_cfg.get('format', 'auto')))

    def _open_parameter_editor(self):
        editor = PIPELINE_DIR / "parameter_gui.py"
        if not editor.exists():
            messagebox.showerror("Missing Settings GUI", f"Settings GUI not found:\n{editor}")
            return
        try:
            subprocess.Popen([sys.executable, str(editor)], cwd=str(REPO_ROOT))
            self.status_var.set("Opened parameter settings GUI")
            self.status_label.configure(fg=C['accent_cyan'])
        except Exception as e:
            messagebox.showerror("Open Error", f"Could not open settings GUI:\n{e}")

    def _load_presets(self):
        self.preset_map.clear()
        options = ["(use parameters.json)"]
        if PRESETS_DIR.exists():
            for path in sorted(PRESETS_DIR.glob('*.json')):
                self.preset_map[path.name] = path
                options.append(path.name)
        self.preset_combo['values'] = options
        if self.preset_var.get() not in options:
            self.preset_var.set(options[0])

    def _browse_raw_root(self):
        initial_dir = self.raw_root_var.get().strip() or str(PIPELINE_DIR / "raw")
        selected = filedialog.askdirectory(initialdir=initial_dir, title="Select Raw Data Folder")
        if selected:
            self.raw_root_var.set(selected)

    def _scan_raw_files(self):
        root = self.raw_root_var.get().strip()
        pattern = self.pattern_var.get().strip() or "sub-*_block*-task.vhdr"
        if not root:
            self.scan_var.set("Files matched: 0")
            self.scan_label.configure(fg=C['accent_red'])
            messagebox.showwarning("Missing Path", "Choose a raw data folder first.")
            return

        root_path = Path(root)
        if not root_path.exists():
            self.scan_var.set("Files matched: 0")
            self.scan_label.configure(fg=C['accent_red'])
            messagebox.showerror("Invalid Path", f"Folder does not exist:\n{root_path}")
            return

        files = sorted(root_path.glob(pattern))
        self.scan_var.set(f"Files matched: {len(files)}")
        self.scan_label.configure(fg=C['accent_green'] if files else C['accent_gold'])
        self._append_log(f"Scan: {len(files)} files matched pattern '{pattern}' in '{root_path}'.")
        for sample in files[:5]:
            self._append_log(f"  - {sample.name}")
        if len(files) > 5:
            self._append_log(f"  ... and {len(files) - 5} more")

    def _build_run_env(self) -> dict[str, str]:
        env = os.environ.copy()
        raw_root = self.raw_root_var.get().strip()
        pattern = self.pattern_var.get().strip()
        data_format = self.format_var.get().strip()

        if raw_root:
            env["EEG_DATA_ROOT"] = raw_root
        if pattern:
            env["EEG_DATA_PATTERN"] = pattern
        if data_format:
            env["EEG_DATA_FORMAT"] = data_format

        if self.apply_preset_var.get():
            selected = self.preset_var.get()
            if selected in self.preset_map:
                env["EEG_PARAMETERS_PATH"] = str(self.preset_map[selected])

        return env

    def _get_step_list(self) -> list[Path]:
        mode = self.mode_var.get()
        if mode == "Preprocessing Only":
            steps = list(PREPROCESS_STEPS)
        elif mode == "Analysis Only":
            steps = list(ANALYSIS_STEPS)
        else:
            steps = list(PREPROCESS_STEPS) + list(ANALYSIS_STEPS)

        if self.include_viz_var.get() and mode in {"Analysis Only", "Full (Preprocess + Analysis)"}:
            steps.extend(VIS_STEPS)
        return steps

    def _set_running(self, running: bool):
        state = 'disabled' if running else 'normal'
        self.mode_combo.configure(state='disabled' if running else 'readonly')
        self.raw_root_entry.configure(state=state)
        self.pattern_entry.configure(state=state)
        self.format_combo.configure(state='disabled' if running else 'readonly')
        self.preset_combo.configure(state='disabled' if running else 'readonly')
        self.apply_preset_chk.configure(state=state)
        self.include_viz_chk.configure(state=state)
        self.browse_btn.configure(state=state)
        self.scan_btn.configure(state=state)
        self.run_btn.configure(state='disabled' if running else 'normal')
        self.stop_btn.configure(state='normal' if running else 'disabled')

    def _start_run(self):
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo("Pipeline Running", "A run is already in progress.")
            return

        steps = self._get_step_list()
        if not steps:
            messagebox.showerror("No Steps", "No scripts were selected to run.")
            return

        mode = self.mode_var.get()
        if mode in {"Preprocessing Only", "Full (Preprocess + Analysis)"}:
            root = self.raw_root_var.get().strip()
            if not root:
                messagebox.showerror("Missing Raw Path", "Set the raw data folder before preprocessing.")
                return
            if not Path(root).exists():
                messagebox.showerror("Invalid Raw Path", f"Folder does not exist:\n{root}")
                return

        if self.apply_preset_var.get() and self.preset_var.get() not in self.preset_map:
            messagebox.showerror("Preset Missing", "Select a preset JSON or untick preset use.")
            return

        env = self._build_run_env()
        self.stop_requested = False
        self.progress.configure(maximum=len(steps), value=0)
        self.status_var.set("Starting...")
        self.status_label.configure(fg=C['accent_gold'])

        self._append_log("")
        self._append_log("=" * 78)
        self._append_log(f"Run started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._append_log(f"Mode: {mode}")
        self._append_log(f"Steps: {len(steps)}")
        if env.get("EEG_PARAMETERS_PATH"):
            self._append_log(f"Preset: {Path(env['EEG_PARAMETERS_PATH']).name}")
        else:
            self._append_log("Preset: (using config/parameters.json)")
        self._append_log("=" * 78)

        self._set_running(True)
        self.worker_thread = threading.Thread(
            target=self._run_steps_worker,
            args=(steps, env),
            daemon=True,
        )
        self.worker_thread.start()

    def _run_steps_worker(self, steps: list[Path], env: dict[str, str]):
        try:
            total = len(steps)
            for i, script in enumerate(steps, start=1):
                if self.stop_requested:
                    break
                if not script.exists():
                    raise FileNotFoundError(f"Script not found: {script}")

                self._ui_status(f"[{i}/{total}] Running {script.name}...")
                self._ui_status_color(C['accent_gold'])
                self._ui_log("")
                self._ui_log(f">>> [{i}/{total}] {script}")

                cmd = [sys.executable, str(script)]
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(REPO_ROOT),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                self.current_process = proc

                assert proc.stdout is not None
                for line in proc.stdout:
                    self._ui_log(line.rstrip())
                    if self.stop_requested and proc.poll() is None:
                        proc.terminate()

                rc = proc.wait()
                self.current_process = None

                if self.stop_requested:
                    self._ui_log("Stop requested. Ending run.")
                    break
                if rc != 0:
                    raise RuntimeError(f"{script.name} failed with exit code {rc}")

                self._ui_progress(i)

            if self.stop_requested:
                self._ui_status("Stopped by user.")
                self._ui_status_color(C['accent_red'])
            else:
                self._ui_status("Completed.")
                self._ui_status_color(C['accent_green'])
                self._ui_log("")
                self._ui_log(f"Run completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        except Exception as exc:
            self._ui_status("Failed.")
            self._ui_status_color(C['accent_red'])
            self._ui_log("")
            self._ui_log(f"ERROR: {exc}")
        finally:
            self.current_process = None
            self.root.after(0, lambda: self._set_running(False))

    def _stop_run(self):
        if not (self.worker_thread and self.worker_thread.is_alive()):
            return
        self.stop_requested = True
        proc = self.current_process
        if proc and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass
        self._append_log("Stop requested...")
        self.status_var.set("Stopping...")
        self.status_label.configure(fg=C['accent_red'])

    def _ui_log(self, text: str):
        self.root.after(0, self._append_log, text)

    def _ui_status(self, text: str):
        self.root.after(0, self.status_var.set, text)

    def _ui_status_color(self, color: str):
        self.root.after(0, lambda: self.status_label.configure(fg=color))

    def _ui_progress(self, value: int):
        self.root.after(0, lambda: self.progress.configure(value=value))

    def _append_log(self, text: str):
        self.log_text.configure(state='normal')
        self.log_text.insert('end', text + "\n")
        self.log_text.see('end')
        self.log_text.configure(state='disabled')


def main():
    root = tk.Tk()
    app = PipelineRunnerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
