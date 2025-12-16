# steps/01_import_qc.py
import json
import sys
from pathlib import Path
import mne
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils_io import (  # <--- Note the 'src.' prefix
    load_config,
    glob_subjects,
    read_raw,
    subject_id_from_path,
    save_clean_raw,
)
from src.utils_qc import basic_filters, find_bad_channels # <--- Note the 'src.' prefix

def save_qc_figures(raw: mne.io.BaseRaw, out_dir: Path, subj: str) -> None:
    """Save a PSD and a short trace panel for fast visual QC."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Power spectrum
    fig_psd = raw.plot_psd(fmin=1, fmax=120, show=False)
    fig_psd.savefig(out_dir / f"{subj}_psd.png", dpi=160)
    mne.viz.utils.plt.close(fig_psd)

    # 10 s trace panel
    fig_tr = raw.plot(
        n_channels=min(32, len(raw.ch_names)),
        duration=10.0,
        scalings=dict(eeg=20e-6, eog=200e-6, emg=100e-6),
        show=False,
    )
    fig_tr.savefig(out_dir / f"{subj}_traces.png", dpi=160)
    mne.viz.utils.plt.close(fig_tr)


def main():
    cfg = load_config()  # robust path inside utils_io

    # 1) Find files
    paths = glob_subjects(cfg["data"]["root"], cfg["data"]["pattern"])
    if not paths:
        raise FileNotFoundError(
            "No raw files matched your pattern. "
            "Check config.data.root and config.data.pattern in study.yml."
        )

    # 2) Prepare output folders
    qc_json_dir = Path("outputs/qc_reports")
    qc_png_dir = Path("outputs/qc_figs")
    out_clean_dir = Path("outputs/derivatives/cleaned_raw")
    qc_json_dir.mkdir(parents=True, exist_ok=True)
    qc_png_dir.mkdir(parents=True, exist_ok=True)
    out_clean_dir.mkdir(parents=True, exist_ok=True)

    # 3) Loop over files
    for p in paths:
        subj = subject_id_from_path(p)

        # 3a) Load raw and set montage
        raw = read_raw(p, cfg["data"]["format"], cfg["montage"])
        raw.info["bads"] = []

        # 3b) Light filtering for stable QC
        raw = basic_filters(
            raw,
            cfg["filters"]["hp"],
            cfg["filters"]["lp"],
            cfg["filters"]["notch"],
        )

        # 3c) Automatic bad channel suggestion
        bads = find_bad_channels(raw)
        raw.info["bads"] = bads

        # 3d) Save quick QC figures
        save_qc_figures(raw, qc_png_dir, subj)

        # 3e) Write a small JSON QC report
        rep = {
            "subject": subj,
            "n_channels": len(raw.ch_names),
            "bads": bads,
            "sfreq": float(raw.info["sfreq"]),
            "montage": cfg["montage"],
            "hp": cfg["filters"]["hp"],
            "lp": cfg["filters"]["lp"],
            "notch": cfg["filters"]["notch"],
            "file": str(p),
        }
        with open(qc_json_dir / f"{subj}_qc.json", "w") as f:
            json.dump(rep, f, indent=2)

        # 3f) Save a filtered copy for the next step
        save_clean_raw(raw, out_clean_dir, subj, "cleaned")

    print(f"Done. QC JSON in {qc_json_dir}, figures in {qc_png_dir}.")


if __name__ == "__main__":
    main()
