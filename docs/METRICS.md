# Metrics Specification

All quantities below are stored in `results/metrics/all_candidates.csv` (one row per MPNN-sequence candidate) and/or in per-candidate `*_metrics.json` files under each run's `stage3_validation/` directory.

## Units and column naming

| Column / key | Meaning | Unit / range | Notes |
|---|---|---|---|
| `rmsd_angstrom` | Cα RMSD after optimal superposition of prediction onto the designed backbone. Missing = NA. | Ångström (Å). ≥ 0. | Source priority recorded in the per-candidate JSON key `rmsd_source`: (1) `colabdesign_log` — `af_val.aux['log']['rmsd']`, the fixbb internally aligned Cα RMSD and the same quantity the official RSO notebook's `designability_test()` reports (primary, paper metric); (2) `tmscore_binary` — RMSD line from the Zhang `TMscore` executable; (3) `numpy_kabsch` — Cα RMSD after an in-notebook Kabsch superposition. |
| `tm_score` | TM-score of prediction vs. designed backbone, reference-length normalised. NA if missing. | [0, 1] (unitless). | Source priority recorded in JSON key `tm_source`: (1) `tmscore_binary` — native-normalised TM-score from the official Zhang `TMscore` executable (downloaded and compiled in the setup cell, same as the official RSO notebook); (2) `numpy_kabsch` — fallback `1/L·Σ 1/(1+(d/d0)²)` with `d0 = 1.24·(L−15)^(1/3) − 1.8` (floored at 0.5) computed after Kabsch alignment in the notebook. Both chains have equal length here, so normalisation choice cannot change the value. |
| `mean_plddt` | Arithmetic mean of AF2 per-residue pLDDT over the entire predicted chain. | [0, 100]. | Source: AF2 `plddt` b-factor-like array. Do not confuse pLDDT with experimental stability. **Scale note:** ColabDesign's `get_plddt()` (`colabdesign/af/loss.py`) averages confidence-bin centres in **[0, 1]**, not the canonical AlphaFold 0–100. The notebooks auto-detect and rescale (`mean_plddt × 100`); the applied conversion is recorded per candidate in `plddt_scale` (`colabdesign_native_0_1_x100` / `native_0_100`; `collect_metrics.py` labels unlabelled legacy 0–1 JSONs `auto_scaled_native_0_1_x100`). |
| `plddt_scale` | Provenance of the pLDDT scale conversion applied. | enum string or NA | See `mean_plddt`. |
| `rmsd_source`, `tm_source` | Which implementation produced the reported RMSD/TM value. | enum string | See the `rmsd_angstrom` / `tm_score` rows. |
| `ptm` | AF2 model_4_ptm predicted-TM output. Present only for `_ptm` models. | [0, 1]. | Source: `af_model.aux['log']['ptm']`. |
| `mpnn_score` | ProteinMPNN sample negative-log-probability score for the sequence (lower = more probable under MPNN). | Real number, unbounded. | Source: `mpnn_model.sample()["score"]`. |
| `mpnn_temperature` | Sampling temperature used in ProteinMPNN. | Float ≥ 0. | Set in config YAML. |
| `rso_iterations` | Number of design-logit iterations actually performed for this backbone. | Integer ≥ 1. | |
| `rso_final_loss` | Scalar total loss at the end of Stage 1 design. | Real number. | Source: final weighted-loss value from `stage1_rso/<bb>_loss_history.csv` (ColabDesign design log `loss` column); exact interpretation is model-internal so use for sanity only, not cross-paper comparison. |
| `rso_runtime_seconds` | Wall-clock seconds elapsed in the Stage 1 design call. | seconds. | Shared across all candidates of the backbone; read from `status.json` top-level `rso_runtime_seconds`. |
| `mpnn_runtime_seconds` | Wall-clock seconds elapsed in the Stage 2 MPNN sample call for the whole candidate batch. | seconds. | Shared across all candidates; read from `status.json -> stage2_mpnn.mpnn_runtime_seconds`. |
| `validation_runtime_seconds` | Wall-clock seconds elapsed in the Stage 3 AF2 prediction for the candidate. | seconds. | First candidate includes XLA compilation overhead. |
| `total_runtime_seconds` | Candidate-attributed wall time: `rso + mpnn + own_validation` (shared RSO/MPNN stages attributed to each row). | seconds. | Do NOT sum this column across candidates; the shared stages would be counted 8 times. |
| `gpu_name` | Human-readable GPU name from `nvidia-smi`. E.g. "NVIDIA A100-SXM4-40GB". | string. | |
| `peak_gpu_memory_mb` | Peak GPU memory consumption observed for the run. NA if instrumentation unavailable. | MiB. | Availability depends on environment; treat as best-effort. |
| `colabdesign_commit` | Git HEAD commit SHA of the installed ColabDesign repo at runtime, or NA. | 40-hex SHA or NA. | |
| `git_commit` | Git HEAD commit SHA of *this* project at runtime, or NA. | 40-hex SHA or NA. | |
| `status` | One of: `pending`, `success`, `failed`, `skipped`. | enum string. | `success` only when predicted PDB written AND metrics.json contains at least `rmsd_angstrom, tm_score, mean_plddt` without an `error` field. |
| `error_message` | First 500 chars of any exception raised while processing the candidate; NA otherwise. | string. | |

## Missing values

We use empty cells in CSV (read as `pd.NA`/`NaN`) for missing values. **Never** fill missing values with `0` in the stored CSV, because 0 is a meaningful value for e.g. RMSD 0 or pLDDT 0. Downstream code in `metrics.py` coerces columns via `pd.to_numeric(..., errors="coerce")` which treats non-numeric strings as NA.

## Per-backbone summary

`results/metrics/backbone_summary.csv` aggregates each `(experiment, run_id, length, seed, backbone_id)` group:

| Column | Meaning |
|---|---|
| `num_candidates` | Total rows (MPNN sequences) for the backbone. |
| `num_successful` | Rows with `status == 'success'`. |
| `best_rmsd_angstrom` | Minimum RMSD among successful candidates. |
| `best_tm_score` | Maximum TM-score among successful candidates. |
| `best_mean_plddt` | Maximum mean pLDDT among successful candidates. |
| `total_runtime_seconds` | True backbone wall time: shared `rso_runtime_seconds` + `mpnn_runtime_seconds` counted once each, plus the sum of per-candidate `validation_runtime_seconds`. (280.8 s for the 100-aa run: 225.6 + 12.0 + 43.2.) |

## Ranking semantics

We deliberately do not mix metrics into a single score without an explicit documented formula. The three independent rankings (`rank_by_rmsd`, `rank_by_tm`, `rank_by_plddt`) are computed by `rank_candidates()` and can be added to any downstream analysis. Any custom composite ranking must:
1. State the precise algebraic formula.
2. State the normalization method per component (e.g. min-max over the backbone's candidates, or z-score vs. a reference cohort).
3. Label clearly that it is a project custom ranking, not an official paper metric.
4. Keep alongside the raw per-metric rankings; never replace them.
