# docs/generate_specparam_diagrams.py
"""
Generate publication-quality SVG diagrams explaining how specparam
(spectral parameterisation) works.

Creates 4-panel figure:
  Panel 1: Raw PSD with 1/f slope visible
  Panel 2: Aperiodic (1/f) fit overlaid
  Panel 3: Flattened spectrum (aperiodic removed) showing periodic peaks
  Panel 4: Gaussian peak fitting with residual

Output: docs/figures/specparam_explanation.svg
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

# --- Ensure output directory exists ---
OUTPUT_DIR = Path(__file__).resolve().parent / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Style ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.edgecolor': 'white',
})

# Colours
C_RAW = '#1a1a2e'        # Dark navy for raw PSD
C_APERIODIC = '#e63946'  # Red for 1/f fit
C_PERIODIC = '#457b9d'   # Steel blue for flattened
C_THETA = '#2a9d8f'      # Teal for theta peak
C_ALPHA = '#e76f51'      # Orange-red for alpha peak
C_GAMMA = '#6c5ce7'      # Purple for gamma
C_RESIDUAL = '#adb5bd'   # Grey for residual
C_FILL = '#a8dadc'        # Light teal fill


def generate_synthetic_psd():
    """Create a realistic synthetic PSD with known components."""
    freqs = np.linspace(1, 40, 1000)

    # 1/f aperiodic component: P = b - log10(f^chi)
    # offset=1.5, exponent=1.5
    aperiodic = 1.5 - 1.5 * np.log10(freqs)

    # Periodic peaks (Gaussians)
    theta_peak = 0.45 * np.exp(-0.5 * ((freqs - 5.8) / 1.2) ** 2)
    alpha_peak = 1.2 * np.exp(-0.5 * ((freqs - 10.2) / 1.5) ** 2)
    beta_peak = 0.15 * np.exp(-0.5 * ((freqs - 22.0) / 2.5) ** 2)

    # Combined (in log space)
    psd_log = aperiodic + theta_peak + alpha_peak + beta_peak

    # Add subtle noise
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.02, len(freqs))
    psd_log_noisy = psd_log + noise

    return freqs, psd_log_noisy, aperiodic, theta_peak, alpha_peak, beta_peak


def main():
    freqs, psd, aperiodic, theta, alpha, beta = generate_synthetic_psd()
    flattened = psd - aperiodic

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ================================================================
    # PANEL 1: Raw PSD
    # ================================================================
    ax1 = axes[0, 0]
    ax1.plot(freqs, psd, color=C_RAW, linewidth=2)
    ax1.fill_between(freqs, psd.min() - 0.3, psd, alpha=0.08, color=C_RAW)

    # Annotate peaks
    ax1.annotate('θ peak\n(~6 Hz)', xy=(5.8, psd[np.argmin(np.abs(freqs - 5.8))]),
                 xytext=(3, psd.max() - 0.3),
                 arrowprops=dict(arrowstyle='->', color=C_THETA, lw=1.5),
                 fontsize=9, color=C_THETA, fontweight='bold')
    ax1.annotate('α peak\n(~10 Hz)', xy=(10.2, psd[np.argmin(np.abs(freqs - 10.2))]),
                 xytext=(15, psd.max() - 0.1),
                 arrowprops=dict(arrowstyle='->', color=C_ALPHA, lw=1.5),
                 fontsize=9, color=C_ALPHA, fontweight='bold')

    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Log Power (a.u.)')
    ax1.set_title('Stage 1: Raw Power Spectral Density', fontsize=13, fontweight='bold',
                  pad=12)
    ax1.text(0.02, 0.05, 'Oscillatory peaks sit ON TOP\nof the 1/f aperiodic slope',
             transform=ax1.transAxes, fontsize=8, fontstyle='italic',
             color='#555', verticalalignment='bottom')
    ax1.grid(True, alpha=0.2)

    # ================================================================
    # PANEL 2: Aperiodic fit
    # ================================================================
    ax2 = axes[0, 1]
    ax2.plot(freqs, psd, color=C_RAW, linewidth=1.5, alpha=0.5, label='Raw PSD')
    ax2.plot(freqs, aperiodic, color=C_APERIODIC, linewidth=2.5,
             linestyle='--', label='Aperiodic fit (1/f)')

    # Shade the periodic components (difference)
    ax2.fill_between(freqs, aperiodic, psd, where=(psd > aperiodic),
                     alpha=0.25, color=C_FILL, label='Periodic component')

    ax2.annotate('Aperiodic (1/f)\nnon-oscillatory\nbackground',
                 xy=(25, aperiodic[np.argmin(np.abs(freqs - 25))]),
                 xytext=(28, aperiodic[np.argmin(np.abs(freqs - 25))] + 0.6),
                 arrowprops=dict(arrowstyle='->', color=C_APERIODIC, lw=1.5),
                 fontsize=8, color=C_APERIODIC, fontweight='bold')

    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Log Power (a.u.)')
    ax2.set_title('Stage 2: Aperiodic (1/f) Fit & Separation', fontsize=13,
                  fontweight='bold', pad=12)
    ax2.legend(fontsize=8, loc='upper right')
    ax2.text(0.02, 0.05,
             'The 1/f slope represents\nnon-oscillatory neural activity\n(broadband, scale-free)',
             transform=ax2.transAxes, fontsize=8, fontstyle='italic',
             color='#555', verticalalignment='bottom')
    ax2.grid(True, alpha=0.2)

    # ================================================================
    # PANEL 3: Flattened spectrum
    # ================================================================
    ax3 = axes[1, 0]
    ax3.plot(freqs, flattened, color=C_PERIODIC, linewidth=2)
    ax3.axhline(0, color='gray', linewidth=0.8, linestyle='-', alpha=0.5)
    ax3.fill_between(freqs, 0, flattened, where=(flattened > 0),
                     alpha=0.15, color=C_PERIODIC)

    # Mark peaks
    theta_idx = np.argmin(np.abs(freqs - 5.8))
    alpha_idx = np.argmin(np.abs(freqs - 10.2))

    ax3.annotate('θ', xy=(5.8, flattened[theta_idx]),
                 xytext=(5.8, flattened[theta_idx] + 0.15),
                 fontsize=14, color=C_THETA, fontweight='bold', ha='center')
    ax3.annotate('α', xy=(10.2, flattened[alpha_idx]),
                 xytext=(10.2, flattened[alpha_idx] + 0.15),
                 fontsize=14, color=C_ALPHA, fontweight='bold', ha='center')

    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Power above 1/f (a.u.)')
    ax3.set_title('Stage 3: Flattened Spectrum (Periodic Only)', fontsize=13,
                  fontweight='bold', pad=12)
    ax3.text(0.02, 0.05,
             'After removing the 1/f slope,\nonly oscillatory peaks remain.\nThese are the TRUE brain rhythms.',
             transform=ax3.transAxes, fontsize=8, fontstyle='italic',
             color='#555', verticalalignment='bottom')
    ax3.grid(True, alpha=0.2)

    # ================================================================
    # PANEL 4: Gaussian peak fitting + residual
    # ================================================================
    ax4 = axes[1, 1]

    # Show flattened spectrum
    ax4.plot(freqs, flattened, color=C_PERIODIC, linewidth=1.5, alpha=0.4,
             label='Flattened spectrum')

    # Individual Gaussian fits
    ax4.plot(freqs, theta, color=C_THETA, linewidth=2.5,
             label=f'θ Gaussian (f₀=5.8, σ=1.2)')
    ax4.fill_between(freqs, 0, theta, alpha=0.15, color=C_THETA)

    ax4.plot(freqs, alpha, color=C_ALPHA, linewidth=2.5,
             label=f'α Gaussian (f₀=10.2, σ=1.5)')
    ax4.fill_between(freqs, 0, alpha, alpha=0.15, color=C_ALPHA)

    ax4.plot(freqs, beta, color=C_GAMMA, linewidth=2,
             label=f'β Gaussian (f₀=22, σ=2.5)', linestyle='--')

    # Residual
    residual = flattened - theta - alpha - beta
    ax4.plot(freqs, residual, color=C_RESIDUAL, linewidth=1,
             label='Residual (noise)', alpha=0.6)
    ax4.axhline(0, color='gray', linewidth=0.8, linestyle='-', alpha=0.5)

    # Annotate the fitting process
    ax4.annotate('1. Fit tallest\n   peak first (α)',
                 xy=(10.2, alpha[np.argmin(np.abs(freqs - 10.2))]),
                 xytext=(18, 0.95),
                 arrowprops=dict(arrowstyle='->', color=C_ALPHA, lw=1.5),
                 fontsize=8, color=C_ALPHA, fontweight='bold', ha='left')

    ax4.annotate('2. Subtract α,\n   fit θ in residual',
                 xy=(5.8, theta[np.argmin(np.abs(freqs - 5.8))]),
                 xytext=(1.5, 0.75),
                 arrowprops=dict(arrowstyle='->', color=C_THETA, lw=1.5),
                 fontsize=8, color=C_THETA, fontweight='bold')

    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('Power above 1/f (a.u.)')
    ax4.set_title('Stage 4: Iterative Gaussian Peak Fitting', fontsize=13,
                  fontweight='bold', pad=12)
    ax4.legend(fontsize=7, loc='upper right', framealpha=0.9)
    ax4.text(0.02, 0.05,
             'Each peak is fit as a Gaussian.\n'
             'After fitting, it is SUBTRACTED →\n'
             'the RESIDUAL reveals smaller peaks.',
             transform=ax4.transAxes, fontsize=8, fontstyle='italic',
             color='#555', verticalalignment='bottom')
    ax4.grid(True, alpha=0.2)

    # --- Global title ---
    fig.suptitle('How Specparam (FOOOF) Works: Spectral Parameterisation',
                 fontsize=16, fontweight='bold', y=1.02)

    fig.tight_layout()

    # Save as SVG (vector) and PNG (raster backup)
    svg_path = OUTPUT_DIR / "specparam_explanation.svg"
    png_path = OUTPUT_DIR / "specparam_explanation.png"
    fig.savefig(svg_path, format='svg', bbox_inches='tight', dpi=300)
    fig.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
    print(f"Saved specparam diagrams:")
    print(f"  SVG: {svg_path}")
    print(f"  PNG: {png_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
