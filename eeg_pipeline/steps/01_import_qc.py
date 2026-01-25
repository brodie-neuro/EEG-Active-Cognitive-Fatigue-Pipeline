# steps/01_import_qc.py
import json
import sys
from pathlib import Path
import mne
import matplotlib.pyplot as plt

# Point to 'eeg_pipeline' folder
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils_io import (
    load_config,
    glob_subjects,
    read_raw,
    subject_id_from_path,
    save_clean_raw,
)
from src.utils_qc import basic_filters, find_bad_channels


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
    cfg = load_config()

    # 1) Find files
    paths = glob_subjects(cfg["data"]["root"], cfg["data"]["pattern"])
    if not paths:
        raise FileNotFoundError(
            "No raw files matched your pattern. Check study.yml"
        )

    # 2) Prepare output folders
    pipeline_root = Path(__file__).resolve().parents[1]
    qc_json_dir = pipeline_root / "outputs" / "qc_reports"
    qc_png_dir = pipeline_root / "outputs" / "qc_figs"
    out_clean_dir = pipeline_root / "outputs" / "derivatives" / "cleaned_raw"

    for d in [qc_json_dir, qc_png_dir, out_clean_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 3) Loop over files
    for p in paths:
        subj = subject_id_from_path(p)
        print(f"Processing {subj}...")

        # 3a) Load raw
        raw = read_raw(p, cfg["data"]["format"], cfg["montage"])

        # --- CRITICAL FIX FOR EMG CHANNELS ---
        # We must tell MNE which channels are NOT brain channels.
        # Otherwise, they get flagged as "Bad EEG" and deleted.
        # We check if these names exist in the data before setting them.
        aux_map = {}
        for ch in ["EMG_L", "EMG_R", "EMG", "ECG", "VEOG", "HEOG"]:
            if ch in raw.ch_names:
                # Map EMG_L to 'emg', VEOG to 'eog', etc.
                if "EMG" in ch: aux_map[ch] = "emg"
                if "EOG" in ch: aux_map[ch] = "eog"
                if "ECG" in ch: aux_map[ch] = "ecg"

        if aux_map:
            print(f"Setting channel types: {aux_map}")
            raw.set_channel_types(aux_map)
        # -------------------------------------

        raw.info["bads"] = []

        # 3b) Light filtering
        raw = basic_filters(
            raw,
            cfg["filters"]["hp"],
            cfg["filters"]["lp"],
            cfg["filters"]["notch"],
        )

        # 3c) Automatic bad channel suggestion
        # Note: This function only looks at 'eeg' channels now.
        # Since we changed EMG_L to type 'emg' above, it will be IGNORED here.
        bads = find_bad_channels(raw)
        raw.info["bads"] = bads
        print(f"Bad EEG channels found: {bads}")

        # 3d) Save QC figures (will show EMG at bottom)
        save_qc_figures(raw, qc_png_dir, subj)

        # 3e) Save JSON Report
        rep = {
            "subject": subj,
            "n_channels": len(raw.ch_names),
            "bads": bads,
            "file": str(p),
        }
        with open(qc_json_dir / f"{subj}_qc.json", "w") as f:
            json.dump(rep, f, indent=2)

        # 3f) Save
        save_clean_raw(raw, out_clean_dir, subj, "cleaned")

    print(f"Done. Check {qc_png_dir} for the tall image!")


if __name__ == "__main__":
    main()