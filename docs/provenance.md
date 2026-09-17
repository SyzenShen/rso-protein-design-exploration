# Provenance

This file tracks software versions, notebook origins, and run metadata for every reproducible run. Per-run provenance is also written to each run directory as `metadata.json`.

## Official RSO notebook reference

- **ColabDesign repository URL:** https://github.com/sokrypton/ColabDesign
- **Official RSO notebook URL:** https://github.com/sokrypton/ColabDesign/blob/main/af/examples/RSO.ipynb
- **Official RSO notebook (direct raw):** https://raw.githubusercontent.com/sokrypton/ColabDesign/main/af/examples/RSO.ipynb
- **Download date (UTC) of the read-only reference copy stored in this repo:** `2026-09-16T09:19:01Z`
- **Reference notebook SHA-256 (of `notebooks/00_official_RSO_reference.ipynb`):**
  `4d2c9477e65127277bd1c56b90c7fb69fd2e7ed00435d95d16baad1ca0b24ffb`
- **Stored reference path:** `notebooks/00_official_RSO_reference.ipynb` (COPY — NOT MODIFIED)

Check the hash manually:
```bash
shasum -a 256 notebooks/00_official_RSO_reference.ipynb
```
It MUST match the value above. If it does not, the file was modified; restore from the raw URL and record a new download date + hash in that case.

## RSO paper sources

- Frank et al. 2024 *Science* paper: DOI [10.1126/science.adq1741](https://doi.org/10.1126/science.adq1741). Free full text at PMC [PMC11734486](https://pmc.ncbi.nlm.nih.gov/articles/PMC11734486/).
- Paper code snapshot (Zenodo): DOI [10.5281/zenodo.13309081](https://doi.org/10.5281/zenodo.13309081).
- Paper dataset (figshare): DOI [10.6084/m9.figshare.27009724](https://doi.org/10.6084/m9.figshare.27009724).
- Experimental crystal structure PDB: [8S89](https://www.rcsb.org/structure/8S89).

## Reproducibility records per run

Each run directory (`results/runs/<run_id>/metadata.json`) contains at minimum:

| key | meaning |
|---|---|
| `timestamp_utc` | ISO UTC timestamp when `collect_provenance()` was called. |
| `colabdesign_repo_url` | `https://github.com/sokrypton/ColabDesign` |
| `colabdesign_commit` | Git HEAD of the installed ColabDesign (detected at runtime or `null` if not in a git checkout). |
| `rso_notebook_url` | URL to the official RSO notebook used as API reference. |
| `rso_notebook_download_date` | Fixed to the value above — date the reference was last re-fetched. |
| `rso_notebook_sha256` | The SHA-256 of the reference notebook. |
| `python_version` | `platform.python_version()` at runtime. |
| `jax_version` | JAX `__version__` at runtime. |
| `jax_devices` | `[str(d) for d in jax.devices()]` — lists GPUs actually seen by JAX. |
| `cuda_version` | NVCC version if available. |
| `gpu_name` / `gpu_memory_mb` | NVIDIA GPU name and total memory in MiB. |
| `alphafold_params_source` | How AF2 params were obtained ("downloaded_during_setup" by default on Colab). |
| `mpnn_weights_type` | e.g. `"soluble"` for the reproduction. |
| `random_seed` | Integer seed used for the backbone. |
| `git_commit` | HEAD commit of THIS project repo at run time. |
| `git_dirty` | Whether the working tree had uncommitted changes at run time. |
| `platform_info` | OS/arch/machine/version. |
| `config_snapshot` | Full Experiment config dict for the run. |
| `notebook_modified_from_official` | `true` for our curated notebooks; `false` would mean the exact 00 reference was run directly. |

Additionally, `status.json` records:
- Overall run status (`pending`, `stage2_pending`, `stage3_pending`, `completed`, `failed`)
- Per-backbone status + per-stage runtime in seconds
- Per-candidate summaries (after Stage 3)

## Version pinning strategy

If a future ColabDesign `main` commit breaks our notebooks (API change):

1. **Record the exact error** in this file under "Version changes".
2. Re-check the current official notebook at the raw URL above — if it was updated, update the reference copy and the SHA-256 + download date entries above and in every script/notebook constant.
3. Compare against the Zenodo paper snapshot (`10.5281/zenodo.13309081`) to see what version was used during the experiments at publication time.
4. Pin a specific commit for environment.yml/pip install URLs **only after documenting why** in `README.md` (Installation section). Never silently change versions.
5. Run reproduction again with the pinned version; mark in provenance.

## Currently recorded version changes

### 2026-09-16 — JAX 0.11.1 removes `jax.lib.xla_bridge`; ColabDesign main not yet updated

- Environment: Google Colab T4 runtime, Python 3.13 (per traceback path `/usr/local/lib/python3.13/dist-packages/`; Colab's exact minor version may differ between sessions), preinstalled **JAX 0.11.1**.
- Error: `AttributeError: module 'jax.lib' has no attribute 'xla_bridge'` from `colabdesign/shared/utils.py::clear_mem()` (first GPU call).
- ColabDesign version audited: `main` HEAD `e31a56fe1d9b4de25c8697f3a28b75892941cc72` (cloned 2026-09-16). Full-repo grep confirms the single remaining removed-API usage is `jax.lib.xla_bridge.get_backend()` in that one function. In JAX 0.11.1 the module exists at `jax._src.xla_bridge` with `get_backend()` intact; `jax/lib/__init__.py` only re-exports `version_str`.
- Resolution chosen: in-notebook compatibility shim that aliases `jax.lib.xla_bridge = jax._src.xla_bridge` when missing, executed before any ColabDesign GPU call. Baked into `notebooks/01_rso_reproduction.ipynb` and `notebooks/02_length_experiment.ipynb` imports cells. Did **not** downgrade JAX (would require matching jaxlib/CUDA pinning on Colab). Details: `docs/TROUBLESHOOTING.md`.

### 2026-09-16 — JAX 0.11 removed `a_min`/`a_max` kwargs from `jnp.clip`; ColabDesign vendored AF not updated

- Error during the first RSO `design_logits` trace: `TypeError: clip() got an unexpected keyword argument 'a_max'`, originating from `colabdesign/af/alphafold/model/modules.py` relative-position encoding (`jnp.clip(offset + max_relative_feature, a_min=0, a_max=...)`).
- JAX 0.11.1 signature (verified from the wheel `jax/_src/numpy/lax_numpy.py`): `clip(arr, /, min=None, max=None)` — keyword rename from NumPy-1-era `a_min`/`a_max`.
- ColabDesign `main` (`e31a56f`) uses old kwargs at exactly 3 sites: 1 monomer (`modules.py:1452`, blocks our pipeline), 2 multimer (`modules_multimer.py:245,269`). All import via `import jax.numpy as jnp` and resolve `jnp.clip` as a module global at call time.
- Resolution: idempotent wrapper installed on the `jax.numpy` module in the imports-cell shim, translating `a_min/a_max` → `min/max` and accepting positional + new-kwarg call styles (all five call styles verified by emulation). No on-disk patching of ColabDesign, so `colabdesign_commit` provenance stays truthful; the shim is recorded here as an environment deviation. Repo-wide scan for other removed APIs (`np.float_`, `np.int_`, `traverse_util`, `DeviceArray`, `host_count`, etc.) was clean.

### 2026-09-16 — AlphaFold params + metric provenance aligned to official `designability_test()`

- AlphaFold parameters: official `alphafold_params_2022-12-06.tar` from `https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar`, downloaded with `aria2c -x 16` and extracted to `./params` (ColabDesign `data_dir="."`; Colab kernel CWD `/content`) — identical commands to the official RSO setup cell. The ~3.6 GB archive is gitignored, never committed.
- TM-score tooling: Zhang-group `TMscore` binary compiled from `https://zhanggroup.org/TM-score/TMscore.cpp` (`g++ -static -O3 -ffast-math -lm`), exactly as in the official setup cell.
- Validation metric source audit: the official `designability_test()` reports only `plddt, ptm, pae, rmsd, dgram_cce` from `af_model.aux["log"]` — no TM-score column. Our `rmsd_angstrom` therefore uses `log["rmsd"]` (fixbb aligned Cα RMSD, the paper metric) as primary, and adds a real TM-score via the Zhang binary. An earlier notebook draft computed both on *raw, unaligned* coordinates (a methodological bug: verified numerically, unaligned RMSD of a rotated+translated identical structure was 72.7 Å instead of 0). Fixed: Kabsch superposition (numerically verified RMSD=0/TM=1 on identity transform) and binary-first TM-score; each per-candidate JSON now records `rmsd_source` / `tm_source`.
- `mk_afdesign_model(protocol="fixbb", best_metric="rmsd", use_templates=False)` now matches the official helper exactly.

### 2026-09-16 — ColabDesign pLDDT is native 0–1; normalised to canonical 0–100

- Observed on the first real validation run: per-candidate `mean_plddt ≈ 0.92–0.94`. Source inspection of ColabDesign `main` (`colabdesign/af/loss.py::get_plddt`) shows it computes `softmax(predicted_lddt.logits)` averaged over bin centres spanning `[0, 1]` (`bin_width = 1/num_bins`), i.e. a 0–1 confidence fraction — unlike canonical AlphaFold/DeepMind outputs (0–100).
- Resolution: notebooks detect any mean ≤ 1 and multiply by 100 before writing `*_metrics.json`, recording the conversion in `plddt_scale`. `CandidateRecord` gained `rmsd_source`, `tm_source`, `plddt_scale` columns; `collect_metrics.py` defensively rescales unlabelled 0–1 legacy JSONs and tags them `auto_scaled_native_0_1_x100`, keeping the aggregate CSV on one scale. `ptm`/`tm_score` remain 0–1 (dimensionless); the Stage 1 console `plddt` loss component is a 0–1 internal loss value and is not a reported metric.

### 2026-09-16 — First real end-to-end run completed on Colab T4 (`repro_100aa_s42_20260916_140944`)

- Configuration: `configs/reproduction_100aa.yaml` (L = 100, seed 42, 100 steps = 90 gumbel + 10 save_best, 8 MPNN sequences T = 0.1, soluble weights, `rm_aa="C"` on both stages, validation single-sequence `model_4_ptm` / 3 recycles / 1 model).
- Runtime: Google Colab free-tier **Tesla T4**, Python 3.13.15, JAX 0.11.1, ColabDesign `e31a56fe1d9b4de25c8697f3a28b75892941cc72` (metadata key `colabdesign_commit`). Wall times from `status.json`: RSO 225.6 s, MPNN batch 12.0 s, validation 31.0 s (c0 incl. XLA compile) + 1.73/1.73/1.73/1.73/1.73/1.76/1.75 s (c1–c7).
- Outcome: overall `status: "completed"`; 8/8 candidates `success`; RMSD 0.509–0.872 Å (`rmsd_source: colabdesign_log`), TM 0.9534–0.9816 (`tm_source: tmscore_binary`), mean pLDDT 91.60–94.28 (`plddt_scale: colabdesign_native_0_1_x100`), pTM 0.793–0.843.
- Provenance gaps, disclosed: (1) `git_commit` in `metadata.json` is `null` because Colab's ColabDesign clone (and this project upload) lacked a `.git` directory — software identity is instead pinned by `colabdesign_commit` + JAX/Python versions above; (2) `peak_gpu_memory_mb` is NA (no pmon capture). Neither gap affects metric integrity.
- Result transfer: Colab export `rso_results_20260916T143204Z.zip` (32 entries, timestamped 2026-09-16T14:32:04Z) was downloaded to the repo root, unpacked into `results/runs/repro_100aa_s42_20260916_140944/`, then aggregated by `scripts/collect_metrics.py` and gated by `scripts/verify_outputs.py --require-real-metrics` → exit 0, 0 errors / 0 warnings. The zip itself is `*.zip`-gitignored and is not committed; its entire contents are committed under `results/runs/`.

### 2026-09-16 — Stage 1 loss-history CSV reconstructed from verbatim console capture

- The Stage 1 notebook cell's in-run loss dump silently wrote nothing because the installed ColabDesign version exposes the design log via `af_model.aux["log"]` rather than the `af_model.log` attribute the cell probed with `hasattr` (no error was raised).
- Resolution (no numeric invention): the 100 design-step rows were transcribed from the verbatim Colab console output captured during the run (fixed-width ColabDesign progress table, columns `step models recycles hard soft temp loss con plddt ptm rg`) into `stage1_rso/bb0_loss_history.csv`. The terminal row matches the run summary line (`final loss 0.87`), and all 100 steps are present (0–99). No other metric in the repository derives from console transcription.

### 2026-09-17 — Loss-history extraction code fixed to read `af_model.aux["log"]`

- Both Colab notebooks (`01_rso_reproduction.ipynb` cell 5, `02_length_experiment.ipynb` cell 5) now read the per-step design log from `af_model.aux["log"]` directly, with a fallback to `af_model.log` for older ColabDesign versions that expose the attribute. The old `hasattr(af_model, "log")` probe is removed. Future runs will produce `loss_history.csv` automatically; the existing committed `bb0_loss_history.csv` (reconstructed from console capture) is **not** altered or regenerated.
- This is a code-only fix; no GPU stages were re-run, and no existing PDBs, sequences, or validation metrics were changed.

### 2026-09-17 — `status.json` `final_loss` reconciled with loss-history CSV

- During the original run, `status.json` had `final_loss: null` at both the top level and `backbones.bb0` because the notebook's loss-derivation code (`float(np.mean(af_model.aux["losses"]["total"]))`) returned `None` due to a key-path mismatch in the installed ColabDesign version (the `aux["losses"]` dict was not populated the way the cell assumed at that point in the execution flow).
- The value `0.87` is unambiguously recoverable from the committed `stage1_rso/bb0_loss_history.csv` (step 99, `loss` column). Both `status.json` top-level `final_loss` and `backbones.bb0.final_loss` are now set to `0.87`, with a `final_loss_source` field of `"recovered_post_run_from_loss_history_csv"` and a `final_loss_note` documenting the recovery. The value was not fabricated or estimated; it is the exact terminal value from the per-step log, which itself matches the run's console summary line.
