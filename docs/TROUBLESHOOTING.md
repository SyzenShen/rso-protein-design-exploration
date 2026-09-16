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

## Stage 1 RSO errors

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
