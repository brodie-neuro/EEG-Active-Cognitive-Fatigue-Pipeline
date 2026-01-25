# steps/00_view_raw.py
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # add .../eeg_pipeline

import mne
from src.utils_io import load_config, glob_subjects  # noqa: E402

def main():
    cfg = load_config()
    paths = glob_subjects(cfg["data"]["root"], cfg["data"]["pattern"])
    if not paths:
        raise RuntimeError("No files matched your pattern in study.yml")

    vhdr = Path(paths[0])  # BrainVision header file
    raw = mne.io.read_raw_brainvision(vhdr, preload=True)

    raw.plot(
        n_channels=64, duration=8.0,
        scalings=dict(eeg=20e-6, eog=200e-6, emg=100e-6),
        block=True,
    )
    raw.plot_psd(fmin=1, fmax=120)

if __name__ == "__main__":
    main()
