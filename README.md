# rso-protein-design-exploration

## 2. Summary

This repository implements a computational de novo protein design pipeline based on Relaxed Sequence Optimization (RSO) as introduced by Frank et al. (2024). It integrates RSO backbone hallucination via ColabDesign, ProteinMPNN inverse folding for sequence generation, and AlphaFold-derived structure prediction for self-consistency validation. The project targets the unconditional 100-amino-acid generation benchmark from the original paper and extends it with a controlled length-scaling experiment. All experimental parameters are specified in versioned YAML configurations, outputs are structured with full provenance metadata, and results are validated by an automated verification script. This repository contains no wet-lab experiments and no fabricated computational results; all GPU-intensive stages are designed for execution on Google Colab with an NVIDIA runtime. Reproduction status: NOT YET EXECUTED.

## 3. Scientific Motivation

De novo protein design aims to generate amino-acid sequences and three-dimensional backbones that do not exist in nature yet fold into stable, functional structures. Traditional fixed-backbone design prescribes a geometry and searches for a compatible sequence; modern deep-learning methods such as hallucination and inverse folding blur this separation by jointly or iteratively optimizing sequence and backbone. Frank et al. (2024) proposed RSO as a practical two-step protocol: first, a structure-prediction network is relaxed into generating a physically plausible backbone with an implicit sequence representation; second, ProteinMPNN performs explicit inverse folding on that backbone to produce real amino-acid candidates. However, open questions remain about the robustness of this protocol with respect to target length, the interpretation of self-validation metrics, and the degree of confounding introduced when closely related neural-network families participate in both design and evaluation. This repository operationalises these questions as reproducible, configuration-driven experiments.

The following concepts are distinguished throughout this project:

- **Sequence**: the linear string of amino-acid residues (primary structure).
- **Backbone**: the 3D arrangement of N-C-alpha-C atoms, without side chains or an explicit sequence (tertiary-structure scaffold).
- **Structure prediction**: the forward problem of estimating a 3D conformation from a sequence (AlphaFold family, ESMFold).
- **Inverse folding**: the backward problem of assigning amino-acid probabilities to each position given a backbone (ProteinMPNN).
- **RSO**: Relaxed Sequence Optimization -- a gradient-based hallucination procedure in ColabDesign that optimises backbone geometry while treating sequence representation as a continuous latent variable, producing a designed backbone PDB.
- **ProteinMPNN**: a message-passing neural network that performs explicit inverse folding, outputting discrete candidate sequences and per-residue log-odds scores.
- **Validation**: the independent act of re-predicting the structure of a ProteinMPNN sequence from sequence alone and comparing it against the RSO-designed backbone.
- **Computational confidence**: model-internal scores such as pLDDT (per-residue local distance difference test) or pTM (predicted template modelling score) reported by the structure predictor for its own output. High computational confidence does **not** imply experimental success.
- **Experimental validation**: wet-lab confirmation of expression, solubility, secondary-structure content, thermal stability, and/or atomic-resolution structure determination (e.g., X-ray crystallography, cryo-EM). This project does **not** include any experimental validation.

## 4. What is Relaxed Sequence Optimization?

Relaxed Sequence Optimization (RSO) is a de novo backbone hallucination protocol implemented in the ColabDesign library on top of an AlphaFold-derived trunk. Unlike constrained fixed-backbone design, RSO starts from an unstructured polypeptide chain of user-specified length and iteratively refines both backbone geometry and a soft (continuous) amino-acid probability distribution. The optimisation is split into two stages. Stage 1 uses Gumbel-softmax sampling over the 20 amino-acid alphabet, optimising a composite loss that rewards radius of gyration consistency, helix propensity, geometric constraints, and internal model confidence (pLDDT and predicted aligned error). Stage 2 switches to a soft probability mode to refine the final backbone coordinates while reducing stochasticity. Because the sequence representation remains relaxed throughout optimisation, RSO does **not** output a discrete sequence. Instead, it emits a designed backbone PDB file which must subsequently be assigned explicit sequences by a separate inverse-folding step (here, ProteinMPNN). This decomposition separates the geometric creativity of the hallucination from the physico-chemical realism of the sequence assignment, at the cost of introducing a second model and a potential gap between what RSO intended and what a real amino-acid sequence can actually encode.

## 5. Pipeline Overview

The pipeline is organised into four sequential stages. Each stage writes its outputs under `results/runs/<run_id>/` into a dedicated subdirectory with its own status file, supporting checkpointed resumption.

1. **Stage 1 -- RSO backbone generation.** A random seed, target length, and loss weights are loaded from the YAML configuration. ColabDesign's `rf` or `af` backend (per configuration) runs RSO hallucination for the configured number of iterations (default 100, split as 90 gumbel + 10 soft). The final designed backbone is exported as `stage1_rso/<backbone_id>.pdb`, a loss-history CSV is saved, and run metadata (including per-step loss components, wall-clock time, and provenance) is serialised.

2. **Stage 2 -- ProteinMPNN sequence design.** For each RSO-designed backbone PDB, ProteinMPNN (ColabDesign integration, soluble weights by default, cysteine removed) generates `num_seqs` candidate sequences (default 8) at a fixed sampling temperature (default 0.1). Sequences, FASTA headers, ProteinMPNN scores, sampling temperature, and execution time are written to `stage2_mpnn/` as both `.fasta` and `.csv` files.

3. **Stage 3 -- Independent structure prediction and validation.** Each ProteinMPNN candidate sequence is fed back through a structure-prediction model (default: AlphaFold single-model `model_4_ptm` with 3 recycles) to obtain a predicted 3D structure and internal confidence metrics. The predicted structure is then structurally aligned to the RSO-designed backbone, and the backbone-only RMSD (in Angstrom) and TM-score are computed. Per-residue pLDDT values from the prediction are averaged to `mean_plddt`, and the predicted template-modelling score `ptm` is recorded if available. All outputs are written to `stage3_validation/`.

4. **Stage 4 -- Ranking, collection, and visualisation.** Per-backbone candidates are ranked independently by lowest RMSD, highest TM-score, and highest mean pLDDT; no arbitrary composite score is enforced by default. The `collect_metrics.py` script consolidates all candidates into `results/metrics/all_candidates.csv` (one row per candidate) and produces a per-backbone summary `backbone_summary.csv`. The `make_figures.py` script then reads the consolidated CSV and, if and only if real non-missing numeric values are present, writes publication-ready PNG figures to `figures/`.

A pipeline diagram is not included because no real runs have been produced yet; once the 100-aa reproduction completes, `03_results_analysis.ipynb` can render a schematic from the actual stage completion data.

## 6. Reproduction Target

The primary reproduction target is the **unconditional 100-amino-acid generation benchmark** of Frank et al. (2024, Fig. 2 and Extended Data), as parameterised in the official ColabDesign RSO notebook under its "manuscript" defaults. The configuration is fixed in `configs/reproduction_100aa.yaml` with: length = 100, copies = 1, RSO iterations = 100 (stage1 90 / stage2 10), protocol = hallucination, gumbel-then-soft mode, cysteine removed from both RSO and ProteinMPNN, ProteinMPNN soluble weights, temperature 0.1, 8 sequences per backbone, AlphaFold `model_4_ptm` validation with 3 recycles, and random seed 42.

**Reproduction status: NOT YET EXECUTED.** No GPU runs have been performed on this machine. The repository is scaffolded, configured, unit-tested on CPU-only code paths, and ready for execution on a Colab NVIDIA T4/A100 runtime per `docs/COLAB_RUNBOOK.md`. This section will be updated only after `scripts/verify_outputs.py --require-real-metrics` passes on merged-back Colab results.

## 7. Length Experiment

A controlled parameter sweep over protein length is specified in two complementary configurations. Both share the same loss weights, ProteinMPNN settings, and validation model as the reproduction benchmark so that length is the dominant variable.

- **Quick profile** (`configs/length_experiment_quick.yaml`): lengths = [100, 150, 200] amino acids, 1 independent backbone (random seed) per length, 8 ProteinMPNN sequences per backbone. Intended as a first real-data sanity check after the 100-aa reproduction succeeds. Expected total: 3 backbones x 8 candidates = 24 validation runs.
- **Full profile** (`configs/length_experiment_full.yaml`): lengths = [100, 150, 200, 300] amino acids, 3 independent random seeds (42, 123, 2024) per length, 8 ProteinMPNN sequences per backbone. Intended for generous GPU allocation. Expected total: 12 backbones x 8 candidates = 96 validation runs. Note: 300-aa runs risk out-of-memory on consumer-grade GPUs; the configuration is retained as a ceiling rather than a guaranteed-to-run workload.

**Status: Pending execution.** Neither quick nor full profiles have been executed. All length-dependent comparisons in this README are therefore framed as hypotheses, not observations.

## 8. Repository Structure

```
rso-protein-design-exploration/
├── README.md                              -- This file; 21-section project documentation
├── LICENSE                                -- MIT software license
├── CITATION.cff                           -- Citation metadata for Frank et al. & this repo
├── .gitignore                             -- Ignores weights, caches, __pycache__, Drive mounts
├── environment.yml                        -- Conda/mamba environment spec (CPU side)
├── requirements-lock.txt                  -- Pinned pip dependency versions (reproducible venv)
├── 任务.md                                -- Internal Chinese-language project brief / source of specs
├── configs/                               -- YAML experiment configurations
│   ├── smoke.yaml                         -- Minimal 50-aa / 5-step smoke test (not for science)
│   ├── reproduction_100aa.yaml            -- Frank et al. 100-aa unconditional benchmark target
│   ├── length_experiment_quick.yaml       -- Length sweep [100,150,200] x 1 backbone
│   └── length_experiment_full.yaml        -- Length sweep [100,150,200,300] x 3 backbones
├── notebooks/                             -- Intended location for .ipynb files; Colab runbooks
│                                          -- (GPU-stage notebooks are authored and run on Colab;
│                                          --  downloaded copies should be placed here after runs)
├── scripts/                               -- CLI entry points (CPU-side orchestration)
│   ├── check_environment.py               -- Print OS/Python/GPU/JAX status to stdout
│   ├── run_rso.py                         -- Scaffold / launch / dry-run RSO stage from YAML
│   ├── run_mpnn.py                        -- Scaffold / launch ProteinMPNN stage
│   ├── validate_designs.py                -- Scaffold / launch validation (AF prediction + RMSD/TM)
│   ├── run_experiment.py                  -- Convenience driver: --scaffold-only / --collect-only / --plot-only / --verify
│   ├── collect_metrics.py                 -- Aggregate per-run outputs into metrics CSVs
│   ├── make_figures.py                    -- Render figures from real metrics only (empty -> no-op)
│   └── verify_outputs.py                  -- End-to-end schema, file, provenance checks
├── src/rso_exploration/                   -- Reusable Python package (`PYTHONPATH=src`)
│   ├── __init__.py                        -- Package init, exposes __version__
│   ├── config.py                          -- YAML -> dataclass loader + semantic validator
│   ├── paths.py                           -- Project-rooted path constants + per-run layout builders
│   ├── metrics.py                         -- Candidate / backbone data classes, schema, ranking logic
│   ├── provenance.py                      -- SHA-256, git commit, GPU/JAX detection, metadata JSON
│   └── plotting.py                        -- Matplotlib/Seaborn figure generators (gated on real data)
├── tests/                                 -- CPU-only pytest suite
│   ├── test_config.py                     -- Config loading, validation, effective_lengths/seeds
│   ├── test_metrics.py                    -- CSV schema, ranking, backbone summary
│   ├── test_paths.py                      -- Path constants, run_dir layout helpers
│   └── test_result_schema.py              -- End-to-end CandidateRecord round-trip + verify_outputs
├── data/                                  -- Static inputs (kept small; external assets via .gitkeep)
│   ├── README.md                          -- Data policy and download instructions
│   └── external/.gitkeep                  -- Reserved for fetched AF params / MPNN weights
├── results/                               -- All run outputs (never committed before verification)
│   ├── README.md                          -- Results directory policy
│   ├── runs/.gitkeep                      -- Per-run subdirectories live here after execution
│   └── metrics/.gitkeep                   -- Consolidated all_candidates.csv + backbone_summary.csv
├── figures/                               -- PNG outputs (generated only from verified real metrics)
│   └── .gitkeep
└── docs/                                  -- Supplementary documentation
    ├── environment_audit.md               -- Snapshot of local OS/CPU/Python/GPU capabilities
    ├── COLAB_RUNBOOK.md                   -- Step-by-step Google Colab execution guide
    ├── RUNBOOK.md                         -- Local-only: scaffold, tests, collect, plot, verify
    ├── METRICS.md                         -- Column-by-column definition with units and ranges
    ├── METHOD.md                          -- Detailed method write-up aligned to pipeline stages
    ├── provenance.md                      -- Version/commit/SHA-256 ledger; DOIs
    ├── TROUBLESHOOTING.md                 -- Colab OOM, Drive sync, dependency workarounds
    └── scientific_notes.md                -- Rationale for design choices and open questions
```

## 9. Installation

### 9a. Local / CPU side (scaffolding, tests, collection, plotting, verification)

All local-side code runs on a standard CPU-only x86-64 machine. It does **not** require JAX with CUDA and it does **not** run RSO, ProteinMPNN, or AlphaFold inference locally.

1. Clone the repository and change into the project root:
   ```bash
   git clone <your-fork-or-this-repo> rso-protein-design-exploration
   cd rso-protein-design-exploration
   ```
2. Create a Python 3.10 virtual environment (3.10 is the minor version pinned in provenance capture):
   ```bash
   python3.10 -m venv .venv
   source .venv/bin/activate
   ```
3. Install pinned dependencies from the lock file:
   ```bash
   pip install --upgrade pip
   pip install -r requirements-lock.txt
   ```
4. Set `PYTHONPATH` so that the `src/` layout is importable. This is required for all scripts and tests; bake it into your shell session or use a `.env` file:
   ```bash
   export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
   ```
5. Verify the install:
   ```bash
   python scripts/check_environment.py
   python -m pytest tests/ -q
   python scripts/run_experiment.py --config configs/smoke.yaml --scaffold-only
   ```

### 9b. GPU side (actual RSO / MPNN / AF computation)

All model-weight downloads, JAX CUDA initialisation, RSO optimisation, ProteinMPNN sampling, and AlphaFold structure prediction are executed on **Google Colab with an NVIDIA runtime**, following the step-by-step instructions in `docs/COLAB_RUNBOOK.md`. A Colab notebook hosted in this repository (`notebooks/01_rso_reproduction.ipynb` once saved back from Colab) clones the repo, mounts Google Drive for persistent artefacts, runs the configured experiments, and packages outputs into a single `.zip` for download and merge-back into the local checkout. Do **not** attempt to run RSO on a local CPU-only machine -- ColabDesign is neither installed nor tested in that configuration via the local lock file, and memory/time requirements are prohibitive.

## 10. Running on Google Colab

Full annotated instructions live in `docs/COLAB_RUNBOOK.md`. A concise step summary:

1. **Open the notebook.** Navigate to the Colab-compatible `01_rso_reproduction.ipynb` (either hosted in this repo's `notebooks/` after your first upload, or derived from the official RSO notebook per the runbook). In Colab, use *File -> Upload notebook* or open the raw GitHub URL directly.
2. **Sign in** to a Google account with Drive access; a free-tier account is sufficient for the quick profile, though Colab Pro or Pro+ is recommended for the 300-aa full sweep.
3. **Select GPU runtime.** Under *Runtime -> Change runtime type*, pick *Hardware accelerator: GPU* and *Runtime shape: High-RAM* if available.
4. **Verify actual GPU allocation.** Run the environment check cell; it should print an NVIDIA T4, L4, or A100 with >12 GB VRAM. If a CPU or a P4 with <8 GB is assigned, terminate and reconnect until a suitable GPU is assigned; otherwise stop immediately -- the notebook itself aborts the main pipeline when no GPU is detected.
5. **Execute pipeline cells.** Run setup (clone repo, install ColabDesign pinned commit, fetch AF params), then the reproduction 100-aa experiment, then optionally the quick/full length sweep, then the results packaging cell. Download the produced `<timestamp>_rso_results.zip` file and, back in your local checkout, extract its contents directly over the repository root so that `results/runs/*`, `results/metrics/*`, and `figures/*` merge with the existing directory structure. Finally run `python scripts/verify_outputs.py --require-real-metrics` locally before updating the README results sections.

## 11. Running Locally

Only scaffolding, configuration dry-runs, static tests, post-hoc metrics collection, figure generation from previously-computed CSVs, and output verification are supported locally. None of the actual compute stages (RSO, ProteinMPNN, AlphaFold validation) execute on the local CPU; attempts to run them without a CUDA-enabled JAX installation will fail with informative environment errors.

The authoritative command reference for local operations is `docs/RUNBOOK.md`. The most common invocations, from the project root with `.venv` activated and `PYTHONPATH=src` exported, are:

```bash
# Environment and dependency check
python scripts/check_environment.py

# Unit tests (no GPU required)
python -m pytest tests/ -v

# Dry-run / scaffold a YAML config (creates run dirs, status files, config snapshots;
# does NOT launch RSO)
python scripts/run_experiment.py --config configs/reproduction_100aa.yaml --scaffold-only

# After merging Colab .zip back: collect per-run files into consolidated metrics CSVs
python scripts/run_experiment.py --collect-only

# Render figures from real consolidated metrics (empty CSV -> no-op, no fake plots)
python scripts/run_experiment.py --plot-only

# End-to-end output integrity check (add --require-real-metrics after Colab runs)
python scripts/run_experiment.py --verify
```

## 12. Configuration

All experimental parameters live in YAML files under `configs/`. The schema is typed via dataclasses in `src/rso_exploration/config.py` and validated on load; `validate_config()` raises human-readable semantic errors on stage-iteration mismatches, negative temperatures, etc.

| Configuration file | Purpose |
|---|---|
| `configs/smoke.yaml` | 50-amino-acid, 5-iteration smoke test with 2 ProteinMPNN sequences and 1 recycle. Exercises every code path and schema validator. **Not for scientific use; outputs must never be merged into results/metrics for analysis.** |
| `configs/reproduction_100aa.yaml` | Frank et al. 2024 100-aa unconditional reproduction target with manuscript defaults, seed 42, 8 ProteinMPNN sequences at T=0.1, AF model_4_ptm validation with 3 recycles. The primary scientific entry point. |
| `configs/length_experiment_quick.yaml` | Quick first-pass length sweep over [100, 150, 200] with 1 backbone per length. Run **after** reproduction_100aa has been verified. Shared loss/MPNN/validation settings with reproduction_100aa. |
| `configs/length_experiment_full.yaml` | Full length sweep over [100, 150, 200, 300] with 3 independent seeds (42, 123, 2024) per length. 300-aa is memory-intensive; if Colab OOMs, delete that length from the YAML list and re-run rather than silently capping. |

Common parameters across all four files:

| Parameter (YAML path) | Meaning | Typical value |
|---|---|---|
| `experiment_name` | Human-readable label used as a column in metrics CSVs | `reproduction_100aa` |
| `backend` | Where compute runs. Currently only `colab` is implemented. | `colab` |
| `seed` | Master random seed; per-backbone seeds are derived from this plus offsets. | `42` |
| `rso.length` / `rso.lengths` | Single target length (int) for reproduction, or a list of lengths for the sweep. | `100` or `[100, 150, 200]` |
| `rso.copies` | Symmetric copies within the asymmetric unit (hallucination; kept at 1 here). | `1` |
| `rso.iterations` | Total RSO gradient steps. | `100` |
| `rso.stage1_iterations` + `stage2_iterations` | Split between gumbel and soft mode; must sum to `iterations`. | `90` + `10` |
| `rso.protocol` | ColabDesign hallucination mode; only `hallucination` is used here. | `hallucination` |
| `rso.mode` | Ordered list of modes per stage. | `["gumbel", "soft"]` |
| `rso.rm_aa` | Comma-joined single-letter amino acids excluded from RSO design. | `"C"` (cysteine removed to avoid disulfide ambiguity) |
| `rso.loss.*_weight` | Float coefficients for each loss term (rg, helix, con, plddt, pae). | See `reproduction_100aa.yaml` for manuscript defaults |
| `mpnn.num_seqs` | How many ProteinMPNN candidate sequences to sample per backbone. | `8` |
| `mpnn.temperature` | Sampling temperature; 0.1 is near-greedy, higher is more diverse. | `0.1` |
| `mpnn.weights` | MPNN weight set; `soluble` (favouring surface-exposed polar residues) is the paper default. | `soluble` |
| `mpnn.rm_aa` | Residues masked from ProteinMPNN output alphabet. | `"C"` |
| `validation.model` | Validation backend identifier. | `alphafold_single` |
| `validation.alphafold_model` | Specific AF checkpoint name. | `model_4_ptm` |
| `validation.num_recycles` | Number of AF recycling iterations. | `3` |
| `output.base_dir` | Root for per-run subdirectories. | `results/runs` |
| `output.overwrite` / `output.resume` | Safety flags. `resume: true` + `overwrite: false` = checkpointed skip of completed tasks. | `resume: true`, `overwrite: false` |
| `experiment.backbones_per_length` | (Sweep configs only) independent backbones per length. | `1` (quick) or `3` (full) |

## 13. Output Files

All artefacts produced by pipeline execution live under `results/` (runs and consolidated metrics) and `figures/` (plots).

### Per-run layout: `results/runs/<run_id>/`

Each run ID is a deterministic string encoding experiment name, length, seed, and UTC timestamp, e.g. `reproduction_100aa_len100_seed42_20260916T093000Z`. The directory contains:

- `metadata.json` -- Complete provenance record: ColabDesign commit, RSO notebook SHA-256, Python/JAX/CUDA versions, GPU name and VRAM, AF params source, MPNN weight type, captured random seed, full YAML config snapshot, project git commit and dirty flag, platform triple, and a boolean indicating whether the notebook was modified from the official reference. Schema defined by `ProvenanceRecord` in `src/rso_exploration/provenance.py`.
- `status.json` -- Per-stage completion flag, last-updated timestamp, and per-candidate status rows. Read by resume logic to skip finished tasks after interruptions.
- `config_snapshot.yaml` -- A verbatim copy of the input YAML frozen at run start.
- `stage1_rso/`
  - `bb0.pdb` (and `bb1.pdb` ... for multi-backbone sweeps) -- Designed backbone PDBs output by RSO.
  - `bb0_loss_history.csv` -- Per-step total loss and individual loss components (rg, helix, con, plddt, pae).
- `stage2_mpnn/`
  - `bb0_candidates.fasta` -- Plain FASTA with `>` headers encoding `candidate_id`, `mpnn_score`, `temperature`.
  - `bb0_candidates.csv` -- Machine-readable table of candidate_id, sequence, mpnn_score, temperature, removed_aa, weights, seed, wall-clock seconds.
- `stage3_validation/`
  - `bb0_c00_predicted.pdb` (and `bb0_c01` ... `bb0_c07`) -- AlphaFold-predicted structure for each ProteinMPNN sequence candidate.
  - `bb0_c00_validation.json` (and siblings) -- Per-candidate dictionary with `rmsd_angstrom`, `tm_score`, `mean_plddt`, `ptm`, model name, recycles, wall-clock seconds, error message if any.

### Aggregated metrics: `results/metrics/`

- `all_candidates.csv` -- One row per ProteinMPNN sequence candidate across all merged runs. All 25 required columns from `REQUIRED_CANDIDATE_COLUMNS` in `src/rso_exploration/metrics.py` are guaranteed present; missing values are written as empty (NA), never as zero. Units are embedded in the column names.
- `backbone_summary.csv` -- One row per backbone. Aggregates `best_rmsd_angstrom`, `best_tm_score`, `best_mean_plddt`, `num_candidates`, `num_successful`, and `total_runtime_seconds`.

### Figures: `figures/`

PNG plots at 150 DPI, GitHub-friendly resolution. Each plot function in `src/rso_exploration/plotting.py` short-circuits and returns `None` (creating no file) when its required numeric column is entirely NA, so fake figures from empty runs cannot be produced. Expected figures after successful experiments: `rmsd_vs_length.png`, `tmscore_vs_length.png`, `plddt_vs_length.png`, `runtime_vs_length.png`, `optimization_loss_100aa.png`, `candidate_metrics_100aa.png`.

## 14. Metrics

A column-by-column reference with units, valid ranges, and computation methods is maintained in `docs/METRICS.md`. A brief summary of the core scientific metrics, all of which include their units in column names:

| Column | Description | Units / range |
|---|---|---|
| `rmsd_angstrom` | Backbone-atom (N, CA, C) root-mean-square deviation after global alignment of the AlphaFold-predicted structure to the RSO-designed backbone. Lower = better geometric self-consistency. | Angstrom; [0, +inf) |
| `tm_score` | Template Modelling score of the same superposition. Length-normalised, bounded [0, 1]; values > 0.5 are conventionally interpreted as a similar fold. | Dimensionless; [0, 1] |
| `mean_plddt` | Arithmetic mean of per-residue predicted local distance difference test scores reported internally by the AlphaFold validation model for its own prediction. Note this is a model-confidence score for the prediction, not a measure of agreement with the designed backbone. | Dimensionless; [0, 100] |
| `ptm` | Predicted TM-score reported internally by `model_4_ptm` for its own prediction. Present only when AF model supports it; NA otherwise. | Dimensionless; [0, 1] |
| `mpnn_score` | ProteinMPNN per-residue average negative log-likelihood (`-(S·log q).sum / L`, ColabDesign `mpnn.sample()["score"]`). **Lower = more probable / more favourable** under the MPNN model. | Nats (average cross-entropy); ≥ 0 |
| `rso_runtime_seconds`, `mpnn_runtime_seconds`, `validation_runtime_seconds`, `total_runtime_seconds` | Elapsed wall-clock time per stage. Sum of per-candidate times for total; recorded by per-stage wrappers on the Colab GPU instance. | Seconds |
| Provenance columns: `gpu_name`, `peak_gpu_memory_mb`, `colabdesign_commit`, `git_commit` | Captured by `collect_provenance()` at the start of each run. `peak_gpu_memory_mb` is NA if nvidia-smi pmon is unavailable in the runtime. | Mixed |

**Important caveats that apply to every metric in this project:** (a) high `mean_plddt` reflects AlphaFold's self-confidence for its prediction and does **not** equal experimental validation success -- a sequence can fold confidently to a structure entirely different from the intended backbone; (b) agreement between prediction and design (low RMSD, high TM-score) within a single AlphaFold-family pipeline does **not** mean the protein expresses, is soluble, or is thermodynamically stable in a test tube; (c) using an AlphaFold-derived model for both RSO hallucination and validation may introduce method bias, so the RMSD/TM reported here are best interpreted as self-consistency scores, not independent ground truth.

## 15. Results

**Pending execution.** No real computational runs have been performed on this machine because a CUDA-enabled NVIDIA GPU required by ColabDesign RSO and AlphaFold is not available locally. The pipeline scripts, configuration system, schema, metrics collectors, figure generators, and automated verification logic have all been implemented and unit-tested on CPU-only scaffolding and synthetic fixtures, but no RSO backbone, no ProteinMPNN sequence, no AlphaFold prediction, and no real RMSD/TM-score/pLDDT values exist in the repository.

Population of this Results section is strictly gated by the following checklist:

1. Colab GPU runtime confirmed; notebook executed end-to-end for `configs/reproduction_100aa.yaml`.
2. Result `.zip` downloaded from Colab and merged into the local `results/runs/`, `results/metrics/`, and `figures/` trees.
3. `python scripts/verify_outputs.py --require-real-metrics` exits with code 0. This verifies: required PDB files exist and contain ATOM records; FASTA counts match `mpnn.num_seqs`; predicted PDBs exist for every candidate; `all_candidates.csv` contains all required columns and has real numeric RMSD/TM/pLDDT values for at least one success-status row; `metadata.json` captures a non-null `colabdesign_commit`, `gpu_name`, and `config_snapshot`; and `status.json` stage flags match the actual files on disk.
4. Only after the 100-aa reproduction is verified above will the quick and full length experiments be eligible to populate length-dependent results.

No reproduction of Frank et al. (2024) is claimed, and no reproduction will be claimed, until this gating process completes cleanly on this repository's `main` branch.

## 16. Observations

**Pending execution.** This section is intentionally empty of empirical observations. No data exist yet; any stated patterns would be speculative. The following scientific hypotheses are pre-registered here to guide analysis after runs complete, and are explicitly labelled as hypotheses rather than observations:

- **H1 (length vs. quality):** Under a fixed 100-step RSO budget, RMSD to the designed backbone will increase monotonically with length and TM-score will decrease with length, because the conformational search space grows faster than the optimisation budget. Quick sweep hypothesis: RMSD at 200 aa >= RMSD at 150 aa >= RMSD at 100 aa.
- **H2 (confidence vs. agreement dissociation):** Some candidates with `mean_plddt` > 90 will nonetheless have RMSD > 4 A to the design target, because AlphaFold can be independently confident in an incorrect (off-backbone) fold that is internally self-consistent. Hypothesis: per-candidate `mean_plddt` and `rmsd_angstrom` will show only weak negative correlation (Pearson |r| < 0.4) across the full sweep.
- **H3 (validation model sensitivity):** If a second validation model (e.g., ESMFold) were added, per-candidate ranking by RMSD/TM-score would change non-trivially relative to the AlphaFold `model_4_ptm` ranking. This is not testable with the current single-validation-model configuration and is flagged as future work.
- **H4 (same-family bias):** Because both RSO (AF-derived ColabDesign trunk) and validation use AlphaFold-family models, the RMSD/TM distributions measured here are expected to be optimistic relative to what an orthogonal validation model or crystallographic ground truth would show. Operationalisation: compare this repo's TM-score distribution to the experimental 8S89 benchmark once enough data exists, and quantify the gap.
- **H5 (experimental feedback in ranking):** Candidate rankings that ignore pLDDT and rely purely on TM-score/RMSD will rank the experimentally best-characterised designs more accurately than rankings that use a naive composite. This cannot be addressed without wet-lab data and is flagged as a follow-up integration point.

All five hypotheses are revisable after real data is collected and are superseded by whatever the actual metrics show.

## 17. Limitations

The following caveats are non-negotiable and bind any future results of this project:

1. **No wet-lab validation whatsoever.** This project produces computational candidates and self-consistency metrics only. It does **not** include gene synthesis, expression trials, solubility assays, circular dichroism, thermal-shift assays, crystallography, NMR, or cryo-EM. A low RMSD or a high TM-score or a high pLDDT in this pipeline is not evidence that the corresponding polypeptide chain expresses, folds, or is stable in vitro or in vivo.
2. **High pLDDT does not equal experimental validation success.** pLDDT is an internal confidence score emitted by the AlphaFold structure-prediction model for its own predicted conformation. It has been calibrated against held-out CASP targets on naturally occurring sequences; its calibration on purely de novo hallucinated sequences from the same model family is unknown and should not be assumed to transfer.
3. **Same-family structure-prediction models in both design and validation introduce method bias.** RSO hallucination is implemented on an AlphaFold-derived ColabDesign trunk; the validation model is also an AlphaFold checkpoint (`model_4_ptm`). Two models from the same training distribution, architecture, and parameter lineage can be consistent with each other for reasons of shared inductive bias rather than physical reality. The RMSD and TM-score values reported here should therefore be interpreted as self-consistency or in-distribution agreement, not as independent estimates of design fidelity.
4. **Lengths are capped at 300 amino acids even in the full experiment.** Beyond 300 aa, ColabDesign RSO memory usage on a single GPU grows beyond what is routinely available on Colab free/Pro tiers. Protein design problems of biotechnological interest (e.g., multi-domain proteins, oligomeric assemblies, megadalton complexes) are outside this project's scope.
5. **No independent ESMFold validation has been implemented yet.** An orthogonal ESM-2-based structure predictor (ESMFold, Lin et al. 2023) would be needed to bound the same-family bias described in point 3. The validation pipeline currently supports only `alphafold_single`; ESMFold is retained as a planned extension. If/when added, the relevant Lin et al. bioRxiv/doi citation (currently reserved in References with a "not used here" parenthetical) will be promoted to an active citation.
6. **Sample size is small even when complete.** The full length experiment yields at most 12 backbones x 8 candidates = 96 rows in `all_candidates.csv`. No statistically strong claims about length effects across the whole proteome or across different fold classes will be defensible from this sample. All conclusions drawn after execution will be explicitly qualified with the effective n, and confidence intervals will be reported rather than point tests.

## 18. Questions Raised

The project is designed to operationalise the following five scientific research questions. Each question is stated below with a brief description of how this repository's experiments and metrics are intended to address it.

**Q1. How does candidate quality change with protein length under a fixed RSO budget?**
- Operationalised by the quick and full length sweeps (`length_experiment_quick.yaml`, `length_experiment_full.yaml`). For each backbone, `all_candidates.csv` captures `length` as a column alongside `rmsd_angstrom`, `tm_score`, `mean_plddt`, and `total_runtime_seconds`. Figure generators `plot_rmsd_vs_length`, `plot_tmscore_vs_length`, `plot_plddt_vs_length`, and `plot_runtime_vs_length` in `src/rso_exploration/plotting.py` display all individual candidate points alongside length-wise means and standard deviations; no averaging across candidates without showing the raw scatter. Rank-sum tests between length bins are computed in `03_results_analysis.ipynb` after verification passes.

**Q2. Why can a sequence have high prediction confidence but poor agreement with the designed backbone?**
- Operationalised by the joint distribution of `mean_plddt` (confidence internal to AF) and `rmsd_angstrom` / `tm_score` (agreement with RSO backbone). Per-candidate plots coloured by `mean_plddt` vs. RMSD, together with per-residue pLDDT traces on the aligned predicted vs. designed PDBs, localise the dissociation: local disorder in the designed backbone (high AF confidence in a different local structure) vs. global topology drift. The 8 candidates per backbone provide within-backbone controls for this effect.

**Q3. How sensitive are conclusions to the validation model?**
- The current repository implements only one validation model (`alphafold_single`, checkpoint `model_4_ptm`). Sensitivity analysis therefore requires adding a second validation backend (ESMFold is the planned addition, retained as a configurable option in `validation.model`). Once implemented, a per-candidate Bland-Altman plot of RMSD and a Spearman correlation of per-backbone rankings between validation models quantify sensitivity. Until then this question remains explicitly open, which is itself a result: no claimed finding should be interpreted as robust across validation models.

**Q4. To what extent can using related structure-prediction models for design and evaluation introduce bias?**
- Operationalised in two ways. (a) Methodologically: the provenance record in `metadata.json` always records the RSO ColabDesign commit AND the AF validation model name, so future analysts can compare runs that vary the validation side while keeping design fixed. (b) Empirically: when/if ESMFold validation (Q3) is added, the signed difference in TM-score between same-family (AlphaFold) and cross-family (ESMFold) validations on the same candidate set provides a direct estimate of same-family bias on this specific task. Frank et al.'s experimental benchmark PDB 8S89 is also cited as a future anchor point: if the pipeline's self-reported TM distribution for 100-aa designs is compared against the crystallographic TM distribution for experimentally validated 100-aa designs, the delta is an upper bound on combined bias.

**Q5. How might experimental feedback be incorporated into candidate ranking?**
- Operationalised by (i) deliberately **not** defining a single project-wide composite ranking score in the core pipeline -- the default output reports three fully independent rankings (by RMSD, by TM-score, by mean pLDDT) so that downstream experimental scorers can be trained without leakage, and (ii) including the schema columns `status` and `error_message` plus a reserved extension point in `CandidateRecord` that can accept experimental labels (expression, solubility, Tm, etc.) once produced. Once merged with a future experimental CSV the analysis notebook can fit a simple linear or ordinal model on top of the four existing per-candidate features and compare the resulting weighted ranking against the three naïve rankings.

## 19. Reproducibility and Provenance

This repository treats provenance as first-class data. A detailed ledger of version pins, DOIs, download timestamps, and SHA-256 hashes is maintained in `docs/provenance.md`; per-run provenance is captured into machine-readable JSON at execution time.

Every completed run directory under `results/runs/<run_id>/` contains two files before it is considered valid:

- **`metadata.json`** (written by `collect_provenance()` in `src/rso_exploration/provenance.py`). Fields include: ColabDesign repository URL and git commit SHA of the installed package; RSO notebook GitHub URL, download UTC timestamp, and pre-captured SHA-256 hash (`OFFICIAL_RSO_NOTEBOOK_SHA256` pinned in the same module); Python, JAX, and CUDA versions; JAX device list; GPU name and total VRAM in MB; source of AlphaFold parameters; ProteinMPNN weight type (e.g., `soluble`); the exact random seed passed to RSO; a snapshot of the entire YAML config; this repository's own git commit SHA and dirty-worktree flag; operating-system/machine triple; and a boolean flag recording whether the executed notebook was modified from the official upstream reference.
- **`status.json`** (updated by each stage driver). Contains per-stage success booleans, per-candidate status (`pending` / `success` / `failed` / `skipped`), stage wall-clock times, and error messages. The resume logic in the Colab notebook reads this file on restart and skips any stage/candidate already marked `success` with a file-existence check against the expected outputs, preventing silent duplication after connection drops.

Additional reproducibility controls: the official RSO notebook's SHA-256 is hard-coded into `provenance.py` and checked against the downloaded copy at the top of the Colab runbook; ColabDesign is pinned to a specific git commit in the Colab setup cell (recorded back to `metadata.json.colabdesign_commit`) rather than tracking `main`; every YAML config is snapshotted verbatim into the run directory independently of any future edits to `configs/`; random seeds are deterministic per backbone and captured both in the config snapshot and in the provenance record; and `scripts/verify_outputs.py` refuses to mark a run complete if any of the fields above are null, mismatched against the filesystem, or inconsistent with the consolidated metrics CSVs.

## 20. References

1. Frank, A. V. et al. De novo design of protein structures and sequences by relaxed sequence optimization. *Science* 385, 1070-1079 (2024). doi:[10.1126/science.adq1741](https://doi.org/10.1126/science.adq1741)
2. ColabDesign: *Open-source repository of colab notebooks for protein design and analysis.* Sokrypton et al. GitHub: [https://github.com/sokrypton/ColabDesign](https://github.com/sokrypton/ColabDesign)
3. Dauparas, J. et al. Robust deep learning-based protein sequence design using ProteinMPNN. *Science* 378, 49-56 (2022). doi:[10.1126/science.add2187](https://doi.org/10.1126/science.add2187)
4. Jumper, J. et al. Highly accurate protein structure prediction with AlphaFold. *Nature* 596, 583-589 (2021). doi:[10.1038/s41586-021-03819-2](https://doi.org/10.1038/s41586-021-03819-2). Cited here because both RSO (via the ColabDesign AF-derived trunk) and the validation stage use AlphaFold-family models.
5. ESMFold: Lin, Z. et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. *bioRxiv* (2023) / Nature 617, 485-491 (2023). Reserved for future use; **not used in this repository as currently configured.** An ESMFold validation backend will be added in a later release to enable cross-family model comparisons; if and when it is used in a run, this reference will be promoted and the provenance record will list ESMFold as the validation model.
6. Frank, A. V. et al. Supporting dataset for "De novo design of protein structures and sequences by relaxed sequence optimization." figshare (2024). doi:[10.6084/m9.figshare.27009724](https://doi.org/10.6084/m9.figshare.27009724)
7. Frank, A. V. et al. Code archive accompanying "De novo design of protein structures and sequences by relaxed sequence optimization." Zenodo (2024). doi:[10.5281/zenodo.13309081](https://doi.org/10.5281/zenodo.13309081)

## 21. License

MIT License. See the `LICENSE` file in the repository root for the full text. Third-party components (ColabDesign, ProteinMPNN, AlphaFold, ESMFold if later added) retain their own respective licenses and are not distributed with this repository; each is fetched from its canonical upstream location at Colab setup time, and applicable license terms are documented in `docs/provenance.md` alongside the captured commit SHAs.
