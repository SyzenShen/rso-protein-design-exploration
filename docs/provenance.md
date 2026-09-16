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

(Empty as of initial project scaffolding. Populate after real runs.)
