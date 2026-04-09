# steps/01_import_qc.py
import argparse
import json
import os
import sys
from pathlib import Path
# Deterministic BLAS/LAPACK thread limits: set before scientific imports.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MNE_DONTWRITE_HOME"] = "true"
os.environ.setdefault("_MNE_FAKE_HOME_DIR", os.path.dirname(os.path.dirname(__file__)))
import mne
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Point to 'eeg_pipeline' folder
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils_io import (
    load_config,
    glob_subjects,
    read_raw,
    subject_id_from_path,
    save_clean_raw,
    parse_subject_filter,
    subject_matches,
)
from src.utils_qc import basic_filters, find_bad_channels
from src.utils_config import get_param
from src.utils_determinism import file_sha256, save_step_qc, stable_json_hash
from src.utils_report import QCReport
from src.utils_logging import setup_pipeline_logger, log_subject_start


def save_qc_figures(raw: mne.io.BaseRaw, out_dir: Path, subj: str) -> None:
    """Save a PSD and a short trace panel for fast visual QC."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Power spectrum
    fig_psd = raw.compute_psd(fmin=1, fmax=120).plot(show=False)
    fig_psd.savefig(out_dir / f"{subj}_psd.png", dpi=160)
    plt.close(fig_psd)

    # 2. Trace panel (Corrected)
    # Note: We create the plot, THEN resize it using matplotlib commands
    fig_tr = raw.plot(
        n_channels=len(raw.ch_names),  # Show ALL channels
        duration=10.0,
        scalings=dict(eeg=20e-6, eog=200e-6, emg=100e-6),
        show=False
    )
    # FIX: Use set_size_inches to make it tall enough for 64 channels
    fig_tr.set_size_inches(12, 20)

    fig_tr.savefig(out_dir / f"{subj}_traces.png", dpi=160)
    plt.close(fig_tr)


def main():
    parser = argparse.ArgumentParser(description="Step 01: import + basic QC")
    parser.add_argument(
        "--subject",
        type=str,
        default="",
        help="Optional subject filter (e.g. sub-dario or sub-a,sub-b).",
    )
    args = parser.parse_args()
    subject_filter = parse_subject_filter(args.subject)

    cfg = load_config()
    logger = setup_pipeline_logger('01_import_qc')

    # 1) Find files
    paths = glob_subjects(cfg["data"]["root"], cfg["data"]["pattern"])
    if not paths:
        raise FileNotFoundError(
            "No raw files matched your pattern. Check study.yml"
        )

    if subject_filter:
        paths = [p for p in paths if subject_matches(subject_id_from_path(p), subject_filter)]
        if not paths:
            print(f"No raw files matched --subject filter: {sorted(subject_filter)}")
            return

    # 2) Prepare output folders
    pipeline_root = Path(__file__).resolve().parents[1]
    qc_json_dir = pipeline_root / "outputs" / "qc_reports"
    qc_png_dir = pipeline_root / "outputs" / "qc_figs"
    out_clean_dir = pipeline_root / "outputs" / "derivatives" / "cleaned_raw"
    out_erp_clean_dir = pipeline_root / "outputs" / "derivatives" / "erp_cleaned_raw"

    for d in [qc_json_dir, qc_png_dir, out_clean_dir, out_erp_clean_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 3) Loop over files
    for p in paths:
        subj = subject_id_from_path(p)
        print(f"Processing {subj}...")

        # 3a) Load raw
        raw = read_raw(p, cfg["data"]["format"], cfg["montage"])

        # --- CRITICAL FIX FOR EMG CHANNELS ---
        # We must tell MNE which channels are NOT brain channels.
        # Otherwise, they get flagged as "Bad EEG" and deleted, or worse,
        # they contaminate the robust average reference.
        aux_map = {}

        # CNT files use non-standard names for some channels.
        # Rename to match standard_1020 montage before any type mapping.
        rename_map = {}
        if 'FP1' in raw.ch_names:
            rename_map['FP1'] = 'Fp1'
        if 'LHEOG' in raw.ch_names and 'F7' not in raw.ch_names:
            rename_map['LHEOG'] = 'F7'   # F7 electrode wired through HEOG jack
        if 'RHEOG' in raw.ch_names and 'F8' not in raw.ch_names:
            rename_map['RHEOG'] = 'F8'   # F8 electrode wired through HEOG jack
        if rename_map:
            raw.rename_channels(rename_map)
            print(f"Renamed channels: {rename_map}")

        # Explicit assignments from montage_channel_plan.md
        # Temporalis = bipolar: F7/FT7 (left), F8/FT8 (right)
        # Posterior neck = monopolar: TP7 (left), TP8 (right)
        emg_channels = ["F7", "FT7", "F8", "FT8", "TP7", "TP8"]
        # Mastoid references and unused aux inputs have no montage position.
        misc_channels = ["M1", "M2", "BVEOG", "TVEOG"]
        # AF7/AF8 are scalp EEG — some raw FIFs have them pre-typed as eog.
        # Force them to eeg for consistency across all participants.
        eeg_override = ["AF7", "AF8"]
        # No dedicated EOG electrodes in this study — ICLabel handles ocular artefact.

        for ch in raw.ch_names:
            up = ch.upper()
            if ch in emg_channels or "EMG" in up:
                aux_map[ch] = "emg"
            elif ch in misc_channels:
                aux_map[ch] = "misc"
            elif "ECG" in up:
                aux_map[ch] = "ecg"

        if aux_map:
            print(f"Setting channel types: {aux_map}")
            raw.set_channel_types(aux_map)

        # Force AF7/AF8 to EEG (some FIFs have them pre-typed as eog)
        eeg_fix = {ch: "eeg" for ch in eeg_override if ch in raw.ch_names
                   and raw.get_channel_types(picks=[ch])[0] != "eeg"}
        if eeg_fix:
            print(f"Overriding to EEG: {list(eeg_fix.keys())}")
            raw.set_channel_types(eeg_fix)

        # Re-apply montage AFTER all channel type changes.
        # The initial read_raw() applies it when some channels may still be
        # eog/misc, so MNE skips their positions. Now that types are correct,
        # re-apply so every EEG channel gets coordinates.
        from mne.channels import make_standard_montage
        raw.set_montage(make_standard_montage(cfg['montage']), on_missing='ignore')
        # -------------------------------------

        raw.info["bads"] = []

        # 3b) Shared preprocessing bandpass (from parameters.json)
        filt_params = get_param('filtering', default={})
        hp = float(filt_params.get('hp_freq', 1.0))
        lp = float(filt_params.get('lp_freq', 100.0))
        notch = float(filt_params.get('notch_freq', 50.0))
        erp_cfg = get_param('erp_branch', default={}) or {}
        erp_enabled = bool(erp_cfg.get('enabled', False))
        erp_hp = float(erp_cfg.get('hp_freq', 0.1))

        logger.info(
            "Filtering: HP=%.1f, LP=%.1f, Notch=OFF (Step 03 notch filter handles line-noise removal)",
            hp, lp
        )
        raw_main = basic_filters(raw.copy(), hp, lp, None)
        raw_erp = None
        if erp_enabled:
            logger.info(
                "ERP branch filtering: HP=%.1f, LP=%.1f, Notch=OFF (dedicated P3b branch only)",
                erp_hp, lp
            )
            raw_erp = basic_filters(raw.copy(), erp_hp, lp, None)

        # 3c) Automatic bad channel suggestion
        # Note: This function only looks at 'eeg' channels now.
        # Since we changed EMG_L to type 'emg' above, it will be IGNORED here.
        bads = find_bad_channels(raw_main)
        raw_main.info["bads"] = bads
        if raw_erp is not None:
            raw_erp.info["bads"] = list(bads)
        logger.info("Bad EEG channels found: %s", bads)

        # 3d) Save QC figures (will show EMG at bottom)
        save_qc_figures(raw_main, qc_png_dir, subj)

        # 3e) Save JSON Report
        rep = {
            "subject": subj,
            "n_channels": len(raw.ch_names),
            "bads": bads,
            "file": str(p),
        }
        with open(qc_json_dir / f"{subj}_qc.json", "w") as f:
            json.dump(rep, f, indent=2)

        # 3f) QC Report
        block_str = ''.join(c for c in subj if c.isdigit())
        block_num = int(block_str[-1]) if block_str else 1
        qc = QCReport(subj, block_num)

        n_eeg = len(mne.pick_types(raw_main.info, eeg=True))
        bad_pct = 100 * len(bads) / n_eeg if n_eeg > 0 else 0
        status = qc.assess_metric('Bad channels %', bad_pct,
                                  'max_bad_channels_pct', '<=')
        try:
            events, _ = mne.events_from_annotations(raw_main, verbose=False)
            n_events = len(events)
        except Exception:
            n_events = 0

        out_file = save_clean_raw(raw_main, out_clean_dir, subj, "cleaned")
        erp_out_file = None
        if raw_erp is not None:
            erp_out_file = save_clean_raw(raw_erp, out_erp_clean_dir, subj, "cleaned")

        qc.log_step('01_import', status=status,
                     metrics={
                         'n_channels': len(raw_main.ch_names),
                         'n_eeg_channels': n_eeg,
                         'n_bad_channels': len(bads),
                         'bad_channels': str(bads),
                         'bad_pct': round(bad_pct, 1),
                         'n_events': n_events,
                         'duration_s': round(raw_main.times[-1], 1),
                         'sfreq': raw_main.info['sfreq'],
                     },
                     params_used={
                         'hp_main': hp,
                         'hp_erp': erp_hp if erp_enabled else None,
                         'lp': lp,
                         'notch_requested_hz': notch,
                     },
                     input_file=str(p),
                     output_file=out_file)
        qc.save_report()

        step_qc_path = save_step_qc(
            "01_import",
            subj,
            block_num,
            {
                "status": status,
                "input_file": str(p),
                "input_hash": file_sha256(p),
                "output_file": str(out_file),
                "output_hash": file_sha256(out_file),
                "parameters_used": {
                    'hp_main': hp,
                    'hp_erp': erp_hp if erp_enabled else None,
                    'lp': lp,
                    'notch_requested_hz': notch,
                },
                "step_specific": {
                    "channel_count": len(raw_main.ch_names),
                    "channel_types": {
                        kind: len(mne.pick_types(raw_main.info, **{kind: True}))
                        for kind in ("eeg", "eog", "emg", "ecg")
                    },
                    "sfreq": float(raw_main.info["sfreq"]),
                    "duration_s": float(raw_main.times[-1]),
                    "montage": str(cfg["montage"]),
                    "bad_channels": bads,
                    "bad_channels_hash": stable_json_hash(sorted(bads)),
                    "n_events": int(n_events),
                    "erp_branch": {
                        "enabled": erp_enabled,
                        "hp_freq": erp_hp if erp_enabled else None,
                        "output_file": erp_out_file,
                        "output_hash": file_sha256(erp_out_file) if erp_out_file else None,
                    },
                },
            },
        )
        logger.info("Saved step QC log: %s", step_qc_path)

        # 3g) Save
        logger.info("Saved %s cleaned data.", subj)

    logger.info("Step 01 complete. Check %s for QC.", qc_png_dir)


if __name__ == "__main__":
    main()
