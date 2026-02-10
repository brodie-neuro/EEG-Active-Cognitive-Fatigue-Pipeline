# steps/05_ica_iclabel.py
"""
Step 05: ICA artefact removal using ICLabel classification.
Fits ICA on a 1Hz high-pass filtered copy, applies to original data.
"""
import sys
from pathlib import Path
import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components
import numpy as np

# Add pipeline directory to path for imports
pipeline_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pipeline_dir))
from src.utils_io import load_config, save_clean_raw, subj_id_from_derivative
from src.utils_config import get_param
from src.utils_report import QCReport


def detect_flat_channels(raw, threshold=1e-10):
    """Detect channels with near-zero variance (flat lines)."""
    data = raw.get_data(picks='eeg')
    variances = np.var(data, axis=1)
    flat_mask = variances < threshold
    flat_ch_names = [raw.ch_names[i] for i, is_flat in enumerate(flat_mask) if is_flat]
    return flat_ch_names


def main():
    cfg = load_config()
    
    pipeline_root = Path(__file__).resolve().parents[1]
    input_dir = pipeline_root / "outputs" / "derivatives" / "zapline_raw"
    output_dir = pipeline_root / "outputs" / "derivatives" / "ica_cleaned_raw"
    qc_dir = pipeline_root / "outputs" / "qc_figs" / "ica"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}. Run Step 04 first.")
        
    files = sorted(list(input_dir.glob("*_zapline-raw.fif")))
    if not files:
        print("No files found to process.")
        return
    
    n_components = get_param('ica', 'n_components', default=25)
    ica_method = get_param('ica', 'method', default='infomax')
    ica_seed = get_param('ica', 'random_state', default=42)
    iclabel_thresholds = get_param('ica', 'iclabel_thresholds', default={})
    
    # Map ICLabel category names to our config keys
    label_to_config = {
        'eye blink': 'eye', 'eye': 'eye',
        'heart beat': 'heart', 'heart': 'heart',
        'muscle artifact': 'muscle', 'muscle': 'muscle',
        'channel noise': 'channel_noise',
        'line noise': 'line_noise',
        'other': 'other',
    }
    
    for f in files:
        subj = subj_id_from_derivative(f)
        print(f"--- Processing {subj} (ICA) ---")
        
        raw = mne.io.read_raw_fif(f, preload=True)
        
        # Skip ICA entirely for synthetic data (ICLabel needs pytorch/onnxruntime
        # and RANSAC-free synthetic data doesn't benefit from ICA anyway)
        if 'TEST' in subj.upper():
            print("Synthetic data detected -- skipping ICA, passing through.")
            save_clean_raw(raw, output_dir, subj, "ica")
            continue
        
        # Detect and mark flat channels as bad (safety net for bad electrodes)
        flat_channels = detect_flat_channels(raw)
        if flat_channels:
            print(f"Detected {len(flat_channels)} flat channels: {flat_channels[:5]}...")
            raw.info['bads'].extend(flat_channels)
        
        # Skip ICA if too many bad channels
        n_good = len([ch for ch in raw.ch_names if ch not in raw.info['bads']])
        if n_good < 10:
            print(f"Only {n_good} good channels - skipping ICA.")
            save_clean_raw(raw, output_dir, subj, "ica")
            continue
        
        # Create 1Hz high-pass copy for ICA fitting
        print("Filtering copy at 1.0 Hz for ICA fitting...")
        raw_ica_fit = raw.copy()
        raw_ica_fit.pick_types(eeg=True, exclude='bads')
        raw_ica_fit.filter(l_freq=1.0, h_freq=None, n_jobs=-1, verbose=False)
        
        # Adjust n_components if we have fewer good channels
        n_good_channels = len(raw_ica_fit.ch_names)
        actual_n_components = min(n_components, n_good_channels - 1)
        
        if actual_n_components < 5:
            print(f"Warning: Only {n_good_channels} good channels. Skipping ICA for {subj}.")
            save_clean_raw(raw, output_dir, subj, "ica")
            continue
        
        # Fit ICA
        print(f"Fitting ICA (n={actual_n_components} components on {n_good_channels} channels, method={ica_method})...")
        ica = ICA(n_components=actual_n_components, method=ica_method,
                  random_state=ica_seed, max_iter=500)
        ica.fit(raw_ica_fit)
        
        # ICLabel classification
        print("Running ICLabel classifier...")
        ic_labels = label_components(raw_ica_fit, ica, method="iclabel")
        labels = ic_labels["labels"]
        probs = ic_labels["y_pred_proba"]
        
        # Identify artefacts using per-category thresholds from config
        exclude_idx = []
        print("\n--- Component Classification ---")
        for i, (label, prob) in enumerate(zip(labels, probs)):
            print(f"IC {i:02d}: {label} ({prob:.2f})")
            config_key = label_to_config.get(label, 'other')
            threshold = iclabel_thresholds.get(config_key, 0.80)
            if label != 'brain' and prob > threshold:
                exclude_idx.append(i)
                print(f"  -> EXCLUDED (threshold: {threshold:.2f})")
                    
        print(f"\nMarked for exclusion: {exclude_idx}")
        ica.exclude = exclude_idx
        
        # Save QC figures
        fig_topo = ica.plot_components(show=False)
        if isinstance(fig_topo, list):
            for i, fig in enumerate(fig_topo):
                fig.savefig(qc_dir / f"{subj}_ica_topo_part{i}.png")
        else:
            fig_topo.savefig(qc_dir / f"{subj}_ica_topo.png")
            
        if exclude_idx:
            fig_src = ica.plot_sources(raw_ica_fit, picks=exclude_idx, show=False)
            fig_src.savefig(qc_dir / f"{subj}_ica_excluded_time.png")

        # Apply ICA to original data
        print("Applying ICA to original data...")
        raw_clean = raw.copy()
        ica.apply(raw_clean, exclude=exclude_idx)
        
        # QC report
        block_str = ''.join(c for c in f.stem if c.isdigit())
        block_num = int(block_str[-1]) if block_str else 1
        qc = QCReport(subj, block_num)
        
        n_brain = sum(1 for l in labels if l == 'brain')
        status = qc.assess_metric('ICs rejected', len(exclude_idx),
                                  'max_ica_components_rejected', '<=')
        if n_brain < get_param('qc', 'min_brain_ics_remaining', default=15):
            status = 'WARNING'
        
        qc.log_step('05_ica', status=status,
                     metrics={
                         'n_components_fit': actual_n_components,
                         'n_excluded': len(exclude_idx),
                         'n_brain_remaining': n_brain,
                         'excluded_indices': str(exclude_idx),
                     },
                     params_used={'method': ica_method, 'n_components': n_components,
                                  'thresholds': iclabel_thresholds})
        qc.save_report()
        
        save_clean_raw(raw_clean, output_dir, subj, "ica")
        print(f"Saved {subj} ICA cleaned data.\n")


if __name__ == "__main__":
    main()
