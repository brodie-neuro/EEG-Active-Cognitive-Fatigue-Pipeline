"""
Run EEG pipeline steps from CLI with optional subject filtering.

Examples:
  python run_pipeline.py --subject sub-dario
  python run_pipeline.py --mode preprocess --subject sub-pilot1,sub-dario
  python run_pipeline.py --adapter config/adapters/pilot1_cnt.json --subject sub-pilot1
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parent
PREPROCESS_DIR = "preprocessing"
POSTPROCESS_DIR = "postprocessing"

PREPROCESS = [
    ("01 Import + QC", f"{PREPROCESS_DIR}/01_import_qc.py"),
    ("02 Simple Reference", f"{PREPROCESS_DIR}/02_simple_reference.py"),
    ("03 Notch Filter", f"{PREPROCESS_DIR}/03_notch_filter.py"),
    ("04 ASR", f"{PREPROCESS_DIR}/04_asr.py"),
    ("05 ICA + ICLabel", f"{PREPROCESS_DIR}/05_ica_iclabel.py"),
    ("06 Epoch", f"{PREPROCESS_DIR}/06_epoch.py"),
    ("07 Autoreject", f"{PREPROCESS_DIR}/07_autoreject.py"),
]

ANALYSIS = [
    ("08 ERP P3b", f"{POSTPROCESS_DIR}/08_erp_p3b.py"),
    ("09 Band Power", f"{POSTPROCESS_DIR}/09_band_power.py"),
    ("10 PAC Nodal", f"{POSTPROCESS_DIR}/10_pac_nodal.py"),
    ("11 Theta wPLI", f"{POSTPROCESS_DIR}/11_theta_wpli.py"),
    ("12 Gamma Power", f"{POSTPROCESS_DIR}/12_gamma_power.py"),
    ("13 Merge Features", f"{POSTPROCESS_DIR}/13_merge_features.py"),
]

DETERMINISM_ENV = {
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
    "MNE_DONTWRITE_HOME": "true",
}
REPEAT_CAPABLE_STEPS = {"04_asr", "05_ica_iclabel", "07_autoreject"}


def _normalize_step_key(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_").replace("+", "")


def _find_step(step_ref: str, steps: list[tuple[str, str, bool]]) -> tuple[str, str, bool] | None:
    target = _normalize_step_key(step_ref)
    for name, rel_path, pass_subject in steps:
        candidates = {
            _normalize_step_key(name),
            _normalize_step_key(Path(rel_path).stem),
        }
        if target in candidates:
            return name, rel_path, pass_subject
    return None


def run_step(
    name: str,
    script_rel: str,
    env: dict[str, str],
    subject_filter: str,
    pass_subject: bool,
    extra_args: list[str] | None = None,
) -> tuple[float, bool, str]:
    script = PIPELINE_DIR / script_rel
    cmd = [sys.executable, str(script)]
    if pass_subject and subject_filter:
        cmd += ["--subject", subject_filter]
    if extra_args:
        cmd += extra_args

    print(f"\n{'=' * 70}")
    print(f"STEP: {name}")
    print(f"CMD : {' '.join(cmd)}")
    det_env = {k: env.get(k) for k in DETERMINISM_ENV}
    det_env["_MNE_FAKE_HOME_DIR"] = env.get("_MNE_FAKE_HOME_DIR")
    print(f"ENV : {json.dumps(det_env, sort_keys=True)}")
    print(f"{'=' * 70}")

    start = time.time()
    result = subprocess.run(
        cmd,
        cwd=str(PIPELINE_DIR),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = time.time() - start
    output = (result.stdout or "") + (result.stderr or "")
    success = result.returncode == 0

    # Keep console readable.
    for line in output.splitlines():
        s = line.strip()
        if s and not s.startswith(("Creating RawArray", "NOTE:", "  ")):
            try:
                print(f"  {s}")
            except UnicodeEncodeError:
                # Keep the runner alive on Windows terminals using cp1252.
                safe = s.encode("ascii", errors="replace").decode("ascii")
                print(f"  {safe}")

    print(f"[{'OK' if success else 'FAIL'}] {name} ({elapsed:.1f}s)")
    return elapsed, success, output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EEG preprocessing/analysis pipeline.")
    parser.add_argument("--mode", choices=["full", "preprocess", "analysis"], default="full")
    parser.add_argument("--subject", default="", help="Optional subject filter for steps that support --subject.")
    parser.add_argument("--adapter", default="", help="Optional adapter JSON path.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue running after failed steps.")
    parser.add_argument(
        "--repeat-step",
        default="",
        help="Repeat a single supported step with deterministic hash comparisons (e.g. 04_asr).",
    )
    parser.add_argument(
        "--repeat-count",
        type=int,
        default=0,
        help="Number of additional repeat runs to request from --repeat-step.",
    )
    args = parser.parse_args()

    # Pipeline-wide reproducibility: every child step inherits the same
    # single-threaded BLAS/LAPACK/OpenMP limits and fixed hash seed.
    for var, val in DETERMINISM_ENV.items():
        os.environ[var] = val
    os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(PIPELINE_DIR))
    env = os.environ.copy()

    # Save provenance manifest for this run.
    try:
        sys.path.insert(0, str(PIPELINE_DIR))
        from src.utils_determinism import save_provenance_manifest
        manifest_path = save_provenance_manifest(
            PIPELINE_DIR / "outputs" / "features",
            command=" ".join(sys.argv),
        )
        print(f"Provenance manifest saved: {manifest_path}")
    except Exception as e:
        print(f"WARNING: Could not save provenance manifest: {e}")

    if args.adapter.strip():
        env["EEG_ADAPTER_PATH"] = str((PIPELINE_DIR / args.adapter).resolve() if not Path(args.adapter).is_absolute() else Path(args.adapter))

    available_steps: list[tuple[str, str, bool]] = (
        [(name, path, True) for name, path in PREPROCESS]
        + [(name, path, False) for name, path in ANALYSIS]
    )
    all_steps: list[tuple[str, str, bool]] = []
    if args.mode in {"full", "preprocess"}:
        all_steps.extend([(name, path, True) for name, path in PREPROCESS])
    if args.mode in {"full", "analysis"}:
        all_steps.extend([(name, path, False) for name, path in ANALYSIS])

    if args.repeat_count and not args.repeat_step:
        raise SystemExit("--repeat-count requires --repeat-step.")

    repeat_extra_args: list[str] = []
    if args.repeat_step:
        if args.repeat_count < 1:
            raise SystemExit("--repeat-step requires --repeat-count >= 1.")
        selected = _find_step(args.repeat_step, available_steps)
        if selected is None:
            raise SystemExit(f"Unknown repeat step: {args.repeat_step}")
        step_key = Path(selected[1]).stem
        if step_key not in REPEAT_CAPABLE_STEPS:
            raise SystemExit(
                f"Step '{step_key}' does not support --repeat yet. "
                f"Supported: {sorted(REPEAT_CAPABLE_STEPS)}"
            )
        all_steps = [selected]
        repeat_extra_args = ["--repeat", str(args.repeat_count)]

    total_start = time.time()
    timings: dict[str, float] = {}
    failures: list[str] = []

    print("#" * 70)
    print(f"EEG PIPELINE RUNNER ({args.mode.upper()})")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Subject filter: {args.subject or '(none)'}")
    print(f"Adapter: {env.get('EEG_ADAPTER_PATH', '(none)')}")
    print(f"Determinism: OPENBLAS={env.get('OPENBLAS_NUM_THREADS')}, "
          f"MKL={env.get('MKL_NUM_THREADS')}, OMP={env.get('OMP_NUM_THREADS')}, "
          f"PYTHONHASHSEED={env.get('PYTHONHASHSEED')}, "
          f"_MNE_FAKE_HOME_DIR={env.get('_MNE_FAKE_HOME_DIR')}")
    print("#" * 70)

    for name, rel_path, pass_subject in all_steps:
        elapsed, success, _output = run_step(
            name,
            rel_path,
            env,
            args.subject.strip(),
            pass_subject,
            extra_args=repeat_extra_args,
        )
        timings[name] = elapsed
        if not success:
            failures.append(name)
            if not args.continue_on_error:
                break

    total = time.time() - total_start
    print(f"\n{'=' * 70}")
    print("TIMING SUMMARY")
    print(f"{'=' * 70}")
    for step_name, elapsed in timings.items():
        status = "FAIL" if step_name in failures else "OK"
        pct = (elapsed / total * 100.0) if total > 0 else 0.0
        print(f"[{status}] {step_name:<35s} {elapsed:7.1f}s ({pct:4.1f}%)")
    print(f"TOTAL: {total:.1f}s ({total/60:.1f} min)")
    if failures:
        print(f"Failures: {failures}")
        raise SystemExit(1)

    print(f"Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
