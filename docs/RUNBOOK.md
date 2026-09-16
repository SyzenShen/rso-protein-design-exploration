# RUNBOOK — Local / CPU-side Commands

This repository intentionally runs GPU compute (RSO / ProteinMPNN / AF2 validation) on **Google Colab** with a GPU runtime (see `docs/COLAB_RUNBOOK.md`). This RUNBOOK documents commands you run on your local machine *before* and *after* Colab.

Conventions:
- Command directory: repo root unless stated otherwise.
- `python3` is the interpreter in your PATH with the project `src/` importable (add via `PYTHONPATH=src` or install).

---

## 1. Clone and install local (non-GPU) deps

**What it does:** clones the repo, creates a virtualenv, and installs pinned CPU-side packages needed for configs, tests, plotting, and collection.

```bash
# Run in a working directory of your choice
git clone <YOUR_FORK_URL> rso-protein-design-exploration
cd rso-protein-design-exploration

python3 -m venv .venv
source .venv/bin/activate          # or activate.fish / .bat on Windows
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip install -e .         # optional: installs src/rso_exploration as package
```

- `requirements-lock.txt` pins CPU-side packages; GPU packages (jax cuda, torch cuda, colabdesign, AF2 params) are installed on Colab per notebook setup cells.
- On Apple Silicon Macs many of these packages install natively.

---

## 2. Environment audit

```bash
python3 scripts/check_environment.py --out docs/environment_audit.md
```

- **What it does:** detects OS/CPU/RAM/disk/Python/conda/git/NVIDIA/CUDA/JAX/git state and GitHub network reachability; writes the Markdown report; prints the backend decision on stdout (expected: `B (Google Colab - ...)` on this Mac).
- **Parameter:** `--out` (default `docs/environment_audit.md`) path to the report.
- **Directory:** repo root.
- **Expected output:** file present + exit 0.
- **Common errors:** `shasum` not on PATH (harmless, used for provenance only); network check failure (your firewall blocks GitHub; check proxy).

---

## 3. Validate configuration schemas and load all 4 configs

```bash
PYTHONPATH=src python3 -c "
from rso_exploration.config import load_config, validate_config
for n in ['smoke.yaml','reproduction_100aa.yaml','length_experiment_quick.yaml','length_experiment_full.yaml']:
    c = load_config(n); e = validate_config(c)
    print(n, '->', c.experiment_name, 'errors:', e)
"
```

- **What it does:** parses each YAML with `load_config()` and runs validation rules.
- **Directory:** repo root.
- **Expected:** all four configs print `errors: []`.
- **Common errors:** YAML indentation; if `stage1+stage2 != iterations` you accidentally changed iteration counts.

---

## 4. Scaffold dry-run run directories

```bash
PYTHONPATH=src python3 scripts/run_rso.py --config configs/reproduction_100aa.yaml --dry-run
PYTHONPATH=src python3 scripts/run_rso.py --config configs/length_experiment_quick.yaml --dry-run
```

- **What it does:** validates the config, creates empty `results/runs/<run_id>/` directories with pending `status.json` and `metadata.json` (provenance snapshot). NO GPU work happens.
- **Parameters:**
  - `--config <path>` — YAML config to use.
  - `--dry-run` — **required locally.** Refuses to run real compute without this flag.
- **Directory:** repo root.
- **Expected files created:** under `results/runs/`, one folder per (length × seed), each with `status.json` and `metadata.json`.
- **Common errors:** exit code 2 + message means config validation failed.

---

## 5. Run local test suite (non-GPU, no model params)

```bash
PYTHONPATH=src python3 -m pytest tests/ -v
```

- **What it does:** runs all pytest tests (config loading, paths, metrics schema, run directory layout, provenance round-trip, ranking logic).
- **Directory:** repo root.
- **Expected:** all tests pass (expect ~20 tests).
- **Common errors:** `ModuleNotFoundError: No module named 'rso_exploration'` — you forgot `PYTHONPATH=src` or didn't `pip install -e .`.

---

## 6. Help screens for script CLIs

```bash
for s in check_environment run_rso run_mpnn validate_designs run_experiment collect_metrics make_figures verify_outputs; do
  echo "=== $s ==="; PYTHONPATH=src python3 scripts/$s.py --help 2>&1 | head -20; echo
done
```

- **What it does:** sanity-checks each script imports cleanly and prints argparse help.
- **Expected:** every script exits 0 and shows a `--help` summary.

---

## 7. Run static import smoke test

```bash
PYTHONPATH=src python3 -c "
import rso_exploration
import rso_exploration.config, rso_exploration.metrics, rso_exploration.paths
import rso_exploration.provenance, rso_exploration.plotting
print('ok', rso_exploration.__version__)
"
```

- **Expected:** prints `ok 0.1.0`.

---

## 8. (Post-Colab) Download the zip from Colab → extract to repo root → collect metrics → verify → plot

```bash
# 8a. unzip your Colab archive into a temp dir, merge contents into results/
unzip -o ~/Downloads/rso_results_*.zip -d results/   # archive contains runs/ and metrics/ already

# 8b. (re-)collect unified metrics CSVs
PYTHONPATH=src python3 scripts/collect_metrics.py

# 8c. verify the output files schema
PYTHONPATH=src python3 scripts/verify_outputs.py --require-real-metrics

# 8d. generate real-data figures (only where numeric data exists)
PYTHONPATH=src python3 scripts/make_figures.py

# 8e. convenience one-liner (collect + plot + verify)
PYTHONPATH=src python3 scripts/run_experiment.py --collect-only --plot-only --verify
```

For each command in 8:
- **collect_metrics.py:** scans each `results/runs/<run_id>/` dir, reads per-run FASTA/CSV/metrics.json/PDB and appends to a single `results/metrics/all_candidates.csv` plus backbone summary CSV. Idempotent.
- **verify_outputs.py:** exits 0 only if every successful candidate has a predicted PDB with ATOM records + CSVs pass schema. Add `--require-real-metrics` to gate on ≥1 real successful candidate.
- **make_figures.py:** writes `figures/*.png` only if real numeric columns exist; never plots dummy/synthetic data.

---

## 9. git hygiene

```bash
git status
du -sh results/* figures/*
git check-ignore -v results/runs/*/params* 2>/dev/null || echo "nothing ignored"
```

Before committing:
- Confirm no large model weights, `.zip`, or `.env` accidentally got staged.
- Confirm `results/README.md` state matches reality — only commit CSV/PDB/PNG after `verify_outputs.py --require-real-metrics` passes.
