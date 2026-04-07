# src/utils_determinism.py
"""
Pipeline-wide determinism utilities.

Provides consistent seeding, thread-limiting, and provenance logging
so every preprocessing step can be made reproducible with minimal boilerplate.

Usage at the top of any step script (BEFORE scientific imports):

    import os, sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.utils_determinism import enforce_determinism
    enforce_determinism()          # sets env vars + seeds

    import numpy as np             # now safe to import
    ...
"""
import hashlib
import importlib.metadata as importlib_metadata
import json
import os
import platform
import random
import subprocess
import sys
from contextlib import ExitStack, nullcontext
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# 1.  Environment-variable thread caps
# ---------------------------------------------------------------------------
_THREAD_ENV_VARS = {
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
}

_HASH_ENV_VARS = {
    "PYTHONHASHSEED": "0",
}

_EXTRA_ENV_VARS = {
    "MNE_DONTWRITE_HOME": "true",
}


def set_determinism_env() -> None:
    """Set all thread-cap and hash-seed environment variables.

    MUST be called before importing numpy / scipy / sklearn / mne.
    """
    for key, val in {**_THREAD_ENV_VARS, **_HASH_ENV_VARS, **_EXTRA_ENV_VARS}.items():
        os.environ[key] = val
    os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# 2.  Random seeds
# ---------------------------------------------------------------------------
DEFAULT_SEED = 42


def set_random_seeds(seed: int = DEFAULT_SEED) -> None:
    """Seed Python stdlib + numpy PRNGs."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# 3.  threadpoolctl context manager
# ---------------------------------------------------------------------------
def threadpool_limit_context():
    """Return a context manager that caps BLAS/OpenMP threads at runtime.

    Falls back to a no-op context if threadpoolctl is not installed.
    """
    try:
        from threadpoolctl import threadpool_limits
        stack = ExitStack()
        activated = False
        for user_api in ("blas", "openmp"):
            try:
                stack.enter_context(threadpool_limits(limits=1, user_api=user_api))
                activated = True
            except Exception:
                continue
        return stack if activated else nullcontext()
    except ImportError:
        return nullcontext()
    except Exception:
        return nullcontext()


# ---------------------------------------------------------------------------
# 4.  Combined convenience entry-point
# ---------------------------------------------------------------------------
def enforce_determinism(seed: int = DEFAULT_SEED) -> None:
    """One-call setup: env vars + seeds.  Call before scientific imports."""
    set_determinism_env()
    set_random_seeds(seed)


# ---------------------------------------------------------------------------
# 5.  Runtime info logging
# ---------------------------------------------------------------------------
def _get_git_hash() -> str:
    """Return short git commit hash, or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _get_blas_info() -> str:
    """Return BLAS backend description if numpy is available."""
    try:
        from threadpoolctl import threadpool_info
        info = threadpool_info()
        if info:
            return json.dumps(info, sort_keys=True)
    except Exception:
        pass

    try:
        import numpy as np
        info = np.__config__.show(mode="dicts") if hasattr(np.__config__, "show") else {}
        if isinstance(info, dict):
            return json.dumps(info, sort_keys=True, default=str)
        return "see numpy.__config__.show()"
    except Exception:
        return "unavailable"


_PACKAGE_DISTRIBUTIONS = {
    "numpy": "numpy",
    "scipy": "scipy",
    "mne": "mne",
    "asrpy": "asrpy",
    "autoreject": "autoreject",
    "sklearn": "scikit-learn",
    "threadpoolctl": "threadpoolctl",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
}


def _get_package_version(dist_name: str) -> str:
    """Return a package version without importing the package itself."""
    try:
        return importlib_metadata.version(dist_name)
    except importlib_metadata.PackageNotFoundError:
        return "not_installed"
    except Exception as exc:
        return f"error:{type(exc).__name__}"


def get_runtime_info() -> dict:
    """Collect a snapshot of the runtime environment for provenance."""
    info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _get_git_hash(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
    }

    # Package versions
    for pkg_name, dist_name in _PACKAGE_DISTRIBUTIONS.items():
        info[f"{pkg_name}_version"] = _get_package_version(dist_name)

    info["blas_backend"] = _get_blas_info()

    # Active thread env vars
    info["thread_env"] = {k: os.environ.get(k, "NOT_SET") for k in _THREAD_ENV_VARS}
    info["PYTHONHASHSEED"] = os.environ.get("PYTHONHASHSEED", "NOT_SET")
    info["MNE_DONTWRITE_HOME"] = os.environ.get("MNE_DONTWRITE_HOME", "NOT_SET")
    info["_MNE_FAKE_HOME_DIR"] = os.environ.get("_MNE_FAKE_HOME_DIR", "NOT_SET")

    return info


def log_runtime_determinism_info(logger=None) -> dict:
    """Log and return runtime determinism information."""
    info = get_runtime_info()
    msg = (
        f"Determinism env: "
        f"OPENBLAS={info['thread_env'].get('OPENBLAS_NUM_THREADS')}, "
        f"MKL={info['thread_env'].get('MKL_NUM_THREADS')}, "
        f"OMP={info['thread_env'].get('OMP_NUM_THREADS')}, "
        f"PYTHONHASHSEED={info['PYTHONHASHSEED']}"
    )
    if logger:
        logger.info(msg)
    else:
        print(f"  {msg}")
    return info


# ---------------------------------------------------------------------------
# 6.  Provenance manifest
# ---------------------------------------------------------------------------
def save_provenance_manifest(output_dir: Path, command: str = "") -> Path:
    """Write a provenance manifest JSON to the given directory.

    Returns the path to the saved file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline_root = Path(__file__).resolve().parents[1]
    manifest = get_runtime_info()
    manifest["command"] = command or " ".join(sys.argv)
    manifest["pipeline_root"] = str(pipeline_root)
    manifest["pipeline_file_hashes"] = _pipeline_file_hashes(pipeline_root)

    manifest_path = output_dir / "provenance_manifest.json"
    with open(manifest_path, "w") as fp:
        json.dump(manifest, fp, indent=2)
    return manifest_path


# ---------------------------------------------------------------------------
# 7.  Stable hashing helpers (shared across steps)
# ---------------------------------------------------------------------------
def array_hash(arr, length: int = 16) -> str:
    """Deterministic short SHA-256 hash of a numpy array."""
    import numpy as np
    arr_c = np.ascontiguousarray(arr)
    return hashlib.sha256(arr_c.tobytes()).hexdigest()[:length]


def stable_json_hash(value, length: int = 16) -> str:
    """Deterministic short SHA-256 hash of JSON-serializable content."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def file_sha256(path: str | Path, length: int = 16, chunk_size: int = 1024 * 1024) -> str:
    """Deterministic short SHA-256 hash of a file's bytes."""
    digest = hashlib.sha256()
    with open(Path(path), "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()[:length]


def _pipeline_file_hashes(base_dir: Path) -> dict[str, str]:
    """Hash the active code/config surface so runs can be compared exactly."""
    candidates: list[Path] = []
    explicit = [
        base_dir / "run_pipeline.py",
        base_dir / "config" / "parameters.json",
        base_dir / "config" / "study.yml",
    ]
    candidates.extend([p for p in explicit if p.exists()])

    for subdir in ("preprocessing", "postprocessing", "src"):
        root = base_dir / subdir
        if root.exists():
            candidates.extend(sorted(root.rglob("*.py")))

    hashes: dict[str, str] = {}
    seen: set[str] = set()
    for path in candidates:
        rel = str(path.relative_to(base_dir)).replace("\\", "/")
        if rel in seen:
            continue
        seen.add(rel)
        try:
            hashes[rel] = file_sha256(path, length=64)
        except Exception as exc:
            hashes[rel] = f"error:{type(exc).__name__}"
    return hashes


def save_step_qc(
    step_name: str,
    subject: str,
    block: int,
    data: dict,
    output_root: Path | None = None,
) -> Path:
    """Save immutable per-step QC JSON with runtime metadata attached."""
    pipeline_root = Path(__file__).resolve().parents[1]
    qc_dir = (output_root or (pipeline_root / "outputs" / "qc_step_logs"))
    qc_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc)
    ts_label = timestamp.strftime("%Y%m%dT%H%M%S%fZ")
    record = {
        "step": step_name,
        "subject": subject,
        "block": int(block),
        "timestamp": timestamp.isoformat(),
        "runtime_info": get_runtime_info(),
        **data,
    }

    filename = f"{step_name}_{subject}_block{int(block)}_{ts_label}.json"
    path = qc_dir / filename
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, default=str)
    return path
