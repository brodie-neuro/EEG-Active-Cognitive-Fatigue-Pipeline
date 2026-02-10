# eeg_pipeline/src/utils_report.py
"""
Quality Control (QC) report and figure generation.

Generates:
  1. Per-step QC figures (PSD overlays, IC topographies, drop logs, etc.)
  2. Per-subject assessment reports (markdown)

Usage
-----
    from src.utils_report import QCReport

    qc = QCReport('sub-001', block=1)
    qc.log_step('01_import', status='PASS', metrics={...})
    qc.add_figure('01_import_psd', fig)
    qc.save_report()
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

pipeline_dir = Path(__file__).resolve().parents[1]


class QCReport:
    """Accumulates QC data across pipeline steps and generates a report."""

    def __init__(self, subject, block, output_root=None):
        self.subject = subject
        self.block = block
        self.timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.steps = []

        if output_root is None:
            output_root = pipeline_dir / "outputs"

        self.qc_dir = output_root / "qc" / subject / f"block{block}"
        self.qc_dir.mkdir(parents=True, exist_ok=True)

        self.report_dir = output_root / "reports"
        self.report_dir.mkdir(parents=True, exist_ok=True)

        # Load QC thresholds from parameters.json
        try:
            from src.utils_config import get_param
            self.thresholds = get_param('qc') or {}
        except Exception:
            self.thresholds = {}

    def log_step(self, step_name, status='PASS', metrics=None, notes=None,
                 params_used=None, input_file=None, output_file=None):
        """Record a step's QC result.

        Parameters
        ----------
        step_name : str
            Step identifier (e.g. '01_import', '05_ica').
        status : str
            'PASS', 'WARNING', or 'FAIL'.
        metrics : dict, optional
            Key metrics (e.g. {'n_channels': 64, 'n_events': 156}).
        notes : str, optional
            Free-text notes or warnings.
        params_used : dict, optional
            Parameters used for this step (from parameters.json).
        input_file : str, optional
            Path to input file.
        output_file : str, optional
            Path to output file.
        """
        entry = {
            'step': step_name,
            'status': status,
            'metrics': metrics or {},
            'notes': notes or '',
            'params_used': params_used or {},
            'input_file': str(input_file) if input_file else '',
            'output_file': str(output_file) if output_file else '',
            'timestamp': datetime.now().strftime('%H:%M:%S'),
        }
        self.steps.append(entry)

    def add_figure(self, name, fig, dpi=150):
        """Save a matplotlib figure as a QC image.

        Parameters
        ----------
        name : str
            Figure name (without extension).
        fig : matplotlib.figure.Figure
            The figure to save.
        dpi : int
            Resolution.
        """
        fpath = self.qc_dir / f"{name}.png"
        fig.savefig(fpath, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        return fpath

    def assess_metric(self, metric_name, value, threshold_key=None,
                      comparison='<='):
        """Check a metric against a QC threshold.

        Parameters
        ----------
        metric_name : str
            Human-readable metric name.
        value : float
            Observed value.
        threshold_key : str, optional
            Key in parameters.json qc section.
        comparison : str
            '<=' or '>=' for threshold comparison direction.

        Returns
        -------
        str
            'PASS', 'WARNING', or 'FAIL'.
        """
        if threshold_key is None or threshold_key not in self.thresholds:
            return 'PASS'  # No threshold defined

        threshold = self.thresholds[threshold_key]

        if comparison == '<=':
            if value <= threshold:
                return 'PASS'
            elif value <= threshold * 1.2:  # 20% margin = WARNING
                return 'WARNING'
            else:
                return 'FAIL'
        elif comparison == '>=':
            if value >= threshold:
                return 'PASS'
            elif value >= threshold * 0.8:
                return 'WARNING'
            else:
                return 'FAIL'
        return 'PASS'

    def save_report(self):
        """Generate and save the assessment report as markdown."""
        # Count statuses
        n_pass = sum(1 for s in self.steps if s['status'] == 'PASS')
        n_warn = sum(1 for s in self.steps if s['status'] == 'WARNING')
        n_fail = sum(1 for s in self.steps if s['status'] == 'FAIL')
        total = len(self.steps)

        overall = 'PASS'
        if n_fail > 0:
            overall = 'FAIL'
        elif n_warn > 0:
            overall = 'WARNING'

        # Build markdown
        lines = [
            f"# Subject Assessment: {self.subject}, Block {self.block}\n",
            f"## Summary",
            f"- **Overall**: {overall} ({n_pass}/{total} passed"
            f"{f', {n_warn} warnings' if n_warn else ''}"
            f"{f', {n_fail} failures' if n_fail else ''})",
            f"- **Date processed**: {self.timestamp}",
        ]

        # Add parameters hash if available
        try:
            from src.utils_config import parameters_hash
            lines.append(f"- **Parameters hash**: {parameters_hash()}")
        except Exception:
            pass

        lines.append(f"- **QC figures**: `{self.qc_dir}`\n")
        lines.append("## Step-by-Step Assessment\n")

        for entry in self.steps:
            status_icon = {'PASS': 'PASS', 'WARNING': 'WARN', 'FAIL': 'FAIL'}.get(
                entry['status'], '?')
            lines.append(f"### {entry['step']}")
            lines.append(f"- **Status**: [{status_icon}] {entry['status']}")
            lines.append(f"- **Time**: {entry['timestamp']}")

            if entry['metrics']:
                for k, v in entry['metrics'].items():
                    lines.append(f"- **{k}**: {v}")

            if entry['notes']:
                lines.append(f"- **Notes**: {entry['notes']}")

            if entry['params_used']:
                lines.append(f"- **Parameters**: `{json.dumps(entry['params_used'], default=str)}`")

            if entry['input_file']:
                lines.append(f"- **Input**: `{entry['input_file']}`")
            if entry['output_file']:
                lines.append(f"- **Output**: `{entry['output_file']}`")

            lines.append("")

        # Write report
        report_path = self.report_dir / f"{self.subject}_block{self.block}_assessment.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        print(f"Assessment report saved: {report_path}")
        return report_path

    def save_json(self):
        """Save raw QC data as JSON for programmatic access."""
        data = {
            'subject': self.subject,
            'block': self.block,
            'timestamp': self.timestamp,
            'steps': self.steps,
        }
        json_path = self.qc_dir / "qc_data.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        return json_path


# --- Convenience QC figure generators ---

def qc_psd_overlay(raw_before, raw_after, title, fmin=0.5, fmax=50):
    """Generate a before/after PSD overlay figure.

    Parameters
    ----------
    raw_before : mne.io.Raw
        Raw data before processing step.
    raw_after : mne.io.Raw
        Raw data after processing step.
    title : str
        Plot title (e.g. 'PSD: Before/After ZapLine').

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    for raw, label, color, alpha in [
        (raw_before, 'Before', '#F44336', 0.6),
        (raw_after, 'After', '#2196F3', 0.8),
    ]:
        psd = raw.compute_psd(fmin=fmin, fmax=fmax, verbose=False)
        freqs = psd.freqs
        psds_mean = psd.get_data().mean(axis=0) * 1e12  # uV^2/Hz
        ax.semilogy(freqs, psds_mean, color=color, linewidth=1.5,
                    label=label, alpha=alpha)

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (uV^2/Hz)')
    ax.set_title(title, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def qc_epoch_summary(epochs, title='Epoch Summary'):
    """Generate an epoch quality summary figure.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs object (after autoreject if applicable).

    Returns
    -------
    fig : matplotlib.figure.Figure
    metrics : dict
        Summary metrics.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Grand average ERP at Pz
    pz_picks = [ch for ch in ['Pz'] if ch in epochs.ch_names]
    if not pz_picks:
        pz_picks = epochs.ch_names[:1]

    evoked = epochs.copy().pick(pz_picks).average()
    times = evoked.times * 1000  # ms
    data = evoked.data.mean(axis=0) * 1e6  # uV

    axes[0].plot(times, data, color='#2196F3', linewidth=2)
    axes[0].axhline(0, color='grey', linewidth=0.5)
    axes[0].axvline(0, color='grey', linewidth=0.5, linestyle='--')
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Amplitude (uV)')
    axes[0].set_title(f'Grand Average ERP ({pz_picks[0]})')
    axes[0].grid(True, alpha=0.3)

    # Right: Epoch variance distribution
    data_all = epochs.get_data()  # (n_epochs, n_channels, n_times)
    epoch_var = np.var(data_all, axis=(1, 2)) * 1e12  # uV^2
    axes[1].hist(epoch_var, bins=30, color='#4CAF50', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Epoch Variance (uV^2)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Epoch Variance Distribution')
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout()

    metrics = {
        'n_epochs': len(epochs),
        'mean_variance_uv2': float(np.mean(epoch_var)),
        'median_variance_uv2': float(np.median(epoch_var)),
    }
    return fig, metrics
