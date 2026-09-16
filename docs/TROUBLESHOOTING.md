# Troubleshooting

## Environment / setup

### `ModuleNotFoundError: No module named 'rso_exploration'`

Cause: `src/` is not on `PYTHONPATH` and the package isn't installed.

Fix (choose one):
```bash
# In repo root, prefix every python command:
PYTHONPATH=src python3 scripts/collect_metrics.py

# Or install the package in editable mode once:
python3 -m pip install -e .
```

### `ModuleNotFoundError: No module named 'jax'` locally

Expected. JAX with GPU support is installed inside the Colab notebook's setup cell. On macOS/CPU-only machines `jax` is intentionally not in `requirements-lock.txt` because the CPU build would be misleadingly slow and unusable for RSO. Local scripts gracefully skip JAX-dependent paths.

### `nvidia-smi not found` on Colab

You have a CPU runtime. Go to **Runtime → Change runtime type → Hardware accelerator → GPU**, reconnect, then rerun the GPU check cell.

### `JAX sees no GPU.` despite nvidia-smi existing

Either JAX was installed CPU-only, or CUDA is misconfigured. On Colab this almost always resolves by restarting the runtime and rerunning the setup cell (which reinstalls ColabDesign/JAX).

### `AttributeError: module 'jax.lib' has no attribute 'xla_bridge'` in `clear_mem()`

Observed on Colab runtimes preinstalled with **JAX 0.10/0.11** (Python 3.12). ColabDesign `main` (verified at commit `e31a56f`, 2026-09-16) still calls `jax.lib.xla_bridge.get_backend()` in `colabdesign/shared/utils.py`; that attribute was removed from `jax.lib` and the module now lives at `jax._src.xla_bridge` (which still exposes `get_backend()`). A repo-wide audit showed this is the **only** removed-API usage, so a two-line shim suffices — do NOT downgrade JAX:

```python
import jax
if not hasattr(jax.lib, "xla_bridge"):
    from jax._src import xla_bridge as _xla_bridge
    jax.lib.xla_bridge = _xla_bridge
print("xla_bridge backend:", jax.lib.xla_bridge.get_backend().platform)
```

The shim is included in the current `notebooks/01_rso_reproduction.ipynb` and `notebooks/02_length_experiment.ipynb` (imports cell, right after the `jax.devices()` print) and must execute before the first `clear_mem()` / `mk_afdesign_model()` call. Expected output: `xla_bridge backend: gpu`.

### `TypeError: clip() got an unexpected keyword argument 'a_max'` in the first RSO compile

JAX 0.11's `jax.numpy.clip(arr, /, min=None, max=None)` removed the NumPy-1-era `a_min`/`a_max` keyword names. ColabDesign's vendored AlphaFold still calls `jnp.clip(x, a_min=..., a_max=...)` at three sites (`colabdesign/af/alphafold/model/modules.py` relative-position encoding — hit on the monomer hallucination path — and two in `modules_multimer.py`). All three sites resolve `jnp.clip` via module globals at call time, so a wrapper installed on the `jax.numpy` module fixes them, including already-imported modules:

```python
import jax.numpy as jnp
if not getattr(jnp.clip, "_rso_compat", False):
    _orig = jnp.clip
    def _clip_compat(arr, a_min=None, a_max=None, min=None, max=None):
        if a_min is not None: min = a_min
        if a_max is not None: max = a_max
        return _orig(arr, min, max)
    _clip_compat._rso_compat = True
    jnp.clip = _clip_compat
print(jnp.clip(jnp.arange(5), a_max=2))   # [0 1 2 2 2]
```

The wrapper accepts old kwargs, new kwargs (`min`/`max`) and positional calls (verified by emulation against the JAX 0.11 signature). It is included in the same imports-cell shim block as the `xla_bridge` shim. A repo-wide scan of ColabDesign `main` (`e31a56f`) found no other NumPy-2/JAX-removed API usage beyond these and `xla_bridge`.

## Stage 1 RSO errors

### `AssertionError: ERROR: no model params defined` / `WARNING: 'model_*_ptm' not found`

AlphaFold parameters are missing. ColabDesign looks for `./params/params_model_*_ptm.npz` relative to the kernel CWD (`/content` on Colab). The current setup cell downloads them automatically; if you used an older notebook copy or skipped that block, run a cell with:

```python
import os
os.chdir("/content")
!mkdir -p params
!apt-get install -qq aria2
!aria2c -q -x 16 https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar
!tar -xf alphafold_params_2022-12-06.tar -C params
print(sorted(f for f in os.listdir("params") if f.endswith(".npz")))
```

Expect five `params_model_{1..5}_ptm.npz` files (plus monomer weights), then re-run the RSO cell — no runtime restart needed. The ~3.6 GB archive is gitignored; never commit it.

### CUDA OOM on length 150+

1. Reduce batch size where possible (RSO hallucination is single-chain; here OOM usually means the runtime has a small GPU).
2. Request a higher-memory GPU (A100 40 GB).
3. For length experiments exceeding available memory, drop the larger entries from `rso.lengths` and resume.

### Loss history has NaN

Rare. Restart with a different random seed. If systematic, try reducing the `rg_weight` slightly.

## Stage 2 ProteinMPNN errors

### `KeyError: 'seq'` in mpnn output

Almost always because `mpnn_model.prep_inputs` was called with a chain letter that doesn't exist in the backbone PDB. Our notebooks use `chain="A"` for the single-chain hallucination PDB. If you edit the pipeline for binders/heterodimers, pass `chain="A,B"` as needed.

## Stage 3 validation errors

### OOM during AF2 model_4_ptm for length 200/300

Solutions:
1. Reduce `num_recycles` from 3 to 1 (less accurate validation, cheaper).
2. Predict candidates one at a time and `clear_mem()` between them — notebooks already do this.
3. Use A100.
4. Explicitly `import gc; del af_val; gc.collect(); clear_mem()` after each prediction.

### Candidate failed with error: `prediction did not return positions array`

Rare transient. Usually rerunning that single candidate works; the notebook's try/except writes an `error` field in metrics.json so you can spot failed candidates without aborting the whole experiment.

## Post-run collection errors

### `collect_metrics.py` says 0 rows despite runs existing

Common cause: per-run FASTA files are missing. Investigate with:
```bash
ls results/runs/<run_id>/stage2_mpnn/
```
and confirm `*_candidates.fasta` exist. If Stage 2 did not complete on Colab, re-run only that stage by resuming the notebook.

### `verify_outputs.py` fails with "backbone PDB missing/empty ATOM records"

The Stage 1 output PDB is zero-byte or Stage 1 errored. Inspect the run's `status.json` and rerun that specific backbone task.

### Figures not generated after `make_figures.py`

Verify the candidates CSV actually has numeric values. Confirm you are using real Colab data (not a dry-run scaffold-only CSV). No figures are generated from NAs on purpose.

## Git / data hygiene

### `git status` shows huge files staged

Check with:
```bash
git status --short | head -30
du -sh *
```
Undo the add with `git reset HEAD <path>` then add the offending pattern to `.gitignore` if it should always be excluded.

### Colab zip download interrupted

The archive is usually a single zip. Re-download or copy the zip to Google Drive first and then download via Drive's web interface which is more reliable for large files.
