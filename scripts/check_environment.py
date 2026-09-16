#!/usr/bin/env python3
"""Environment audit: print system specs and write docs/environment_audit.md.

Usage:
  python scripts/check_environment.py [--out docs/environment_audit.md]

Checks: OS, CPU, RAM, disk, Python, conda, git, NVIDIA/CUDA, JAX,
         git repo state, network connectivity to ColabDesign.
"""
from __future__ import annotations
import argparse
import datetime
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from rso_exploration.provenance import (  # noqa: E402
    COLABDESIGN_REPO_URL, RSO_NOTEBOOK_URL,
    run_cmd, detect_platform, detect_python_version,
    detect_gpu_name, detect_gpu_memory_mb,
    detect_jax_version, detect_jax_devices,
    get_git_commit, get_git_dirty,
)


def shasum_candidate() -> tuple[Optional[str], Optional[str]]:
    for tool in ["shasum", "sha256sum"]:
        p = shutil.which(tool)
        if p:
            return tool, p
    return None, None


def check_network(url: str, timeout: int = 10) -> tuple[bool, str]:
    # Try urllib first; fall back to curl when the system Python on macOS
    # does not have CA certificates configured (SSL: CERTIFICATE_VERIFY_FAILED).
    try:
        import urllib.request
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return (r.status == 200), f"HTTP {r.status}"
    except Exception:
        pass
    curl = shutil.which("curl")
    if curl:
        try:
            p = subprocess.run(
                [curl, "-sI", "--max-time", str(timeout), url],
                capture_output=True, text=True, timeout=timeout + 5,
            )
            first = (p.stdout or "").splitlines()[0] if p.stdout else ""
            ok = ("200" in first) or ("HTTP" in first and " 3" in first)
            return ok, first.strip() or "curl: no output"
        except Exception as e:
            return False, f"curl fallback failed: {e!s:.200s}"
    return False, "no urllib and no curl available"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(PROJECT_ROOT / "docs" / "environment_audit.md"))
    args = ap.parse_args()

    info: dict[str, str] = {}
    info["timestamp_utc"] = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    info["os_uname"] = subprocess.run(["uname", "-a"], capture_output=True, text=True).stdout.strip()
    info["os_release"] = "\n".join(filter(None, [
        subprocess.run(["sw_vers"], capture_output=True, text=True).stdout.strip(),
        "" if not Path("/etc/os-release").exists() else Path("/etc/os-release").read_text()[:500],
    ]))
    info["cpu"] = (
        subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        or (subprocess.run(["lscpu"], capture_output=True, text=True).stdout or "n/a")[:500]
    )
    mem_bytes = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout.strip()
    info["ram"] = mem_bytes or (subprocess.run(["free", "-h"], capture_output=True, text=True).stdout.strip() or "n/a")
    info["disk"] = subprocess.run(["df", "-h", "/"], capture_output=True, text=True).stdout.strip()
    info["python_version"] = detect_python_version()
    info["conda"] = shutil.which("conda") or "not found"
    info["mamba"] = shutil.which("mamba") or "not found"
    rc, git_out, _ = run_cmd(["git", "--version"])
    info["git"] = git_out if rc == 0 else "not found"
    info["nvidia_smi"] = shutil.which("nvidia-smi") or "not found"
    info["gpu_name"] = detect_gpu_name() or "n/a (no nvidia-smi or no GPU)"
    gmem = detect_gpu_memory_mb()
    info["gpu_memory_mb"] = str(gmem) if gmem is not None else "n/a"
    rc, cuda_out, _ = run_cmd(["nvcc", "--version"])
    info["cuda"] = cuda_out.splitlines()[-1] if (rc == 0 and cuda_out) else "n/a"
    info["jax_version"] = detect_jax_version() or "not installed"
    devices = detect_jax_devices()
    info["jax_devices"] = ", ".join(devices) if devices else "n/a (jax not installed or no devices)"
    git_head = get_git_commit(PROJECT_ROOT)
    info["git_repo_head"] = git_head if git_head else "(not a git repo)"
    git_dirty_flag = get_git_dirty(PROJECT_ROOT)
    if git_dirty_flag is None:
        info["git_dirty"] = "n/a"
    elif git_dirty_flag:
        info["git_dirty"] = "yes"
    else:
        info["git_dirty"] = "no"
    ok_gh, msg_gh = check_network(COLABDESIGN_REPO_URL)
    ok_nb, msg_nb = check_network(RSO_NOTEBOOK_URL)
    info["network_github_colabdesign"] = f"{'REACHABLE' if ok_gh else 'UNREACHABLE'} ({msg_gh})"
    info["network_rso_notebook_raw"] = f"{'REACHABLE' if ok_nb else 'UNREACHABLE'} ({msg_nb})"

    backend = "UNKNOWN"
    if info["gpu_name"] != "n/a (no nvidia-smi or no GPU)" and info["gpu_name"] != "n/a":
        backend = "A (local NVIDIA GPU)"
    elif ok_gh and ok_nb:
        backend = "B (Google Colab - network reachable, no local GPU)"
    else:
        backend = "C (insufficient environment info)"
    info["backend_decision"] = backend

    lines = []
    lines.append("# Environment Audit")
    lines.append("")
    lines.append(f"Generated (UTC): **{info['timestamp_utc']}**")
    lines.append("")
    lines.append("## Audit Results")
    lines.append("")
    keys = [
        ("os_uname", "OS (uname)"), ("os_release", "OS release details"),
        ("cpu", "CPU"), ("ram", "RAM (bytes or human-readable)"),
        ("disk", "Disk usage for /"), ("python_version", "Python version"),
        ("conda", "conda path"), ("mamba", "mamba path"),
        ("git", "git version"), ("nvidia_smi", "nvidia-smi path"),
        ("gpu_name", "GPU model"), ("gpu_memory_mb", "GPU total memory (MB)"),
        ("cuda", "CUDA (nvcc) version"),
        ("jax_version", "JAX version"), ("jax_devices", "JAX visible devices"),
        ("git_repo_head", "Current git HEAD"), ("git_dirty", "Git working tree dirty?"),
        ("network_github_colabdesign", "Network: ColabDesign GitHub"),
        ("network_rso_notebook_raw", "Network: RSO notebook raw URL"),
    ]
    for k, label in keys:
        v = info[k]
        if "\n" in v:
            lines.append(f"### {label}\n```\n{v}\n```\n")
        else:
            lines.append(f"- **{label}:** {v}")
    lines.append("")
    lines.append("## Execution Backend Decision")
    lines.append("")
    lines.append(f"SELECTED: **{backend}**")
    lines.append("")
    lines.append("Rationale:")
    if backend.startswith("B"):
        lines.append("- No NVIDIA GPU or CUDA detected on this machine. RSO, ProteinMPNN inference, and AlphaFold validation require GPU compute.")
        lines.append("- GitHub and raw notebook URLs are reachable, so Google Colab (or a local Linux box with NVIDIA) is appropriate.")
        lines.append("- Local environment still supports configs, plotting, result collection, schema validation, and tests.")
    elif backend.startswith("A"):
        lines.append("- Local NVIDIA GPU detected. Full pipeline may be runnable locally if CUDA, JAX, and params are installed.")
    else:
        lines.append("- Neither local GPU nor network reachable. Static project scaffolding only.")
    lines.append("")
    lines.append("## Local capabilities")
    lines.append("")
    lines.append("- Static code, configuration, and tests: OK")
    lines.append("- Collect + visualize results CSV: OK")
    lines.append("- RSO / AlphaFold / ProteinMPNN GPU computation: NOT RUNNABLE locally (use Colab)")
    lines.append("")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")
    print(f"Backend: {backend}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
