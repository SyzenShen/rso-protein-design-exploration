# Environment Audit

Generated (UTC): **2026-09-16T12:18:01Z**

## Audit Results

- **OS (uname):** Darwin bio-mac 24.6.0 Darwin Kernel Version 24.6.0: Mon Jul 14 11:28:17 PDT 2025; root:xnu-11417.140.69~1/RELEASE_X86_64 x86_64
### OS release details
```
ProductName:		macOS
ProductVersion:		15.6
BuildVersion:		24G84
```

- **CPU:** Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz
- **RAM (bytes or human-readable):** 34359738368
### Disk usage for /
```
Filesystem        Size    Used   Avail Capacity iused ifree %iused  Mounted on
/dev/disk1s4s1   374Gi    10Gi    21Gi    34%    426k  216M    0%   /
```

- **Python version:** 3.10.11
- **conda path:** not found
- **mamba path:** not found
- **git version:** git version 2.50.1 (Apple Git-155)
- **nvidia-smi path:** not found
- **GPU model:** n/a (no nvidia-smi or no GPU)
- **GPU total memory (MB):** n/a
- **CUDA (nvcc) version:** n/a
- **JAX version:** not installed
- **JAX visible devices:** n/a (jax not installed or no devices)
- **Current git HEAD:** (not a git repo)
- **Git working tree dirty?:** n/a
- **Network: ColabDesign GitHub:** REACHABLE (HTTP/2 200)
- **Network: RSO notebook raw URL:** REACHABLE (HTTP/2 200)

## Execution Backend Decision

SELECTED: **B (Google Colab - network reachable, no local GPU)**

Rationale:
- No NVIDIA GPU or CUDA detected on this machine. RSO, ProteinMPNN inference, and AlphaFold validation require GPU compute.
- GitHub and raw notebook URLs are reachable, so Google Colab (or a local Linux box with NVIDIA) is appropriate.
- Local environment still supports configs, plotting, result collection, schema validation, and tests.

## Local capabilities

- Static code, configuration, and tests: OK
- Collect + visualize results CSV: OK
- RSO / AlphaFold / ProteinMPNN GPU computation: NOT RUNNABLE locally (use Colab)

