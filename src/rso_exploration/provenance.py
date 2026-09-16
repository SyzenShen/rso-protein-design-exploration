from __future__ import annotations
import datetime
import hashlib
import json
import os
import platform
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

COLABDESIGN_REPO_URL = "https://github.com/sokrypton/ColabDesign"
RSO_NOTEBOOK_URL = "https://raw.githubusercontent.com/sokrypton/ColabDesign/main/af/examples/RSO.ipynb"
FRANK2024_DOI = "10.1126/science.adq1741"
ZENODO_CODE_DOI = "10.5281/zenodo.13309081"
FIGSHARE_DATA_DOI = "10.6084/m9.figshare.27009724"
PDB_8S89 = "8S89"

OFFICIAL_RSO_NOTEBOOK_SHA256 = "4d2c9477e65127277bd1c56b90c7fb69fd2e7ed00435d95d16baad1ca0b24ffb"
RSO_NOTEBOOK_DOWNLOAD_DATE = "2026-09-16T09:19:01Z"


def utc_now_iso() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_cmd(cmd: list[str], timeout: int = 10) -> tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except Exception as e:
        return 127, "", str(e)


def get_git_commit(repo_path: Optional[str | Path] = None) -> Optional[str]:
    cwd = str(repo_path) if repo_path else os.getcwd()
    rc, out, _ = run_cmd(["git", "-C", cwd, "rev-parse", "HEAD"])
    return out if rc == 0 and out else None


def get_git_dirty(repo_path: Optional[str | Path] = None) -> Optional[bool]:
    cwd = str(repo_path) if repo_path else os.getcwd()
    rc, out, _ = run_cmd(["git", "-C", cwd, "status", "--porcelain"])
    return (len(out) > 0) if rc == 0 else None


def detect_python_version() -> str:
    return platform.python_version()


def detect_platform() -> dict[str, str]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "version": platform.version(),
    }


def detect_gpu_name() -> Optional[str]:
    rc, out, _ = run_cmd(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    if rc == 0 and out:
        return out.splitlines()[0].strip()
    return None


def detect_gpu_memory_mb() -> Optional[float]:
    rc, out, _ = run_cmd(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"])
    if rc == 0 and out:
        try:
            return float(out.splitlines()[0].strip())
        except ValueError:
            return None
    return None


def detect_colabdesign_commit() -> Optional[str]:
    try:
        import colabdesign  # type: ignore
        pkg_dir = Path(colabdesign.__file__).resolve().parent
        rc, out, _ = run_cmd(["git", "-C", str(pkg_dir.parent), "rev-parse", "HEAD"])
        if rc == 0 and out:
            return out
    except Exception:
        pass
    return None


def detect_jax_version() -> Optional[str]:
    try:
        import jax  # type: ignore
        return str(getattr(jax, "__version__", "unknown"))
    except Exception:
        return None


def detect_jax_devices() -> list[str]:
    try:
        import jax  # type: ignore
        return [str(d) for d in jax.devices()]
    except Exception:
        return []


@dataclass
class ProvenanceRecord:
    timestamp_utc: str
    colabdesign_repo_url: str = COLABDESIGN_REPO_URL
    colabdesign_commit: Optional[str] = None
    rso_notebook_url: str = RSO_NOTEBOOK_URL
    rso_notebook_download_date: str = RSO_NOTEBOOK_DOWNLOAD_DATE
    rso_notebook_sha256: str = OFFICIAL_RSO_NOTEBOOK_SHA256
    python_version: Optional[str] = None
    jax_version: Optional[str] = None
    jax_devices: list[str] = None
    cuda_version: Optional[str] = None
    gpu_name: Optional[str] = None
    gpu_memory_mb: Optional[float] = None
    alphafold_params_source: Optional[str] = None
    mpnn_weights_type: Optional[str] = None
    random_seed: Optional[int] = None
    git_commit: Optional[str] = None
    git_dirty: Optional[bool] = None
    platform_info: Optional[dict[str, str]] = None
    config_snapshot: Optional[dict[str, Any]] = None
    notebook_modified_from_official: bool = False

    def __post_init__(self):
        if self.jax_devices is None:
            self.jax_devices = []

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def collect_provenance(
    seed: Optional[int] = None,
    config_snapshot: Optional[dict[str, Any]] = None,
    mpnn_weights: Optional[str] = None,
    alphafold_params_source: str = "downloaded_during_setup",
    notebook_modified: bool = True,
) -> ProvenanceRecord:
    return ProvenanceRecord(
        timestamp_utc=utc_now_iso(),
        colabdesign_commit=detect_colabdesign_commit(),
        python_version=detect_python_version(),
        jax_version=detect_jax_version(),
        jax_devices=detect_jax_devices(),
        gpu_name=detect_gpu_name(),
        gpu_memory_mb=detect_gpu_memory_mb(),
        alphafold_params_source=alphafold_params_source,
        mpnn_weights_type=mpnn_weights,
        random_seed=seed,
        git_commit=get_git_commit(),
        git_dirty=get_git_dirty(),
        platform_info=detect_platform(),
        config_snapshot=config_snapshot,
        notebook_modified_from_official=notebook_modified,
    )


def save_provenance(prov: ProvenanceRecord, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(prov.to_dict(), f, indent=2, default=str)


def load_provenance(path: Path) -> ProvenanceRecord:
    with open(path) as f:
        data = json.load(f)
    return ProvenanceRecord(**data)
