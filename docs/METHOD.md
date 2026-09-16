# Methods

## Pipeline overview

We reproduce the core in silico de novo protein design pipeline reported in Frank et al. 2024 *Science*:

1. **Stage 1 — RSO backbone generation.** Using ColabDesign's AlphaFold2-based hallucination model (`mk_afdesign_model(protocol="hallucination")`), gradient-descent optimize a relaxed (non-one-hot / PSSM-like) sequence representation so that AF2 predicts a compact, high-confidence, well-contacting fold. Losses:
   - Radius of gyration penalty (rg_weight): `0.1` for proteins ≤ 600 aa, `0.01` above 600 aa. Defined as `elu(rg - 2.38 * L^0.365)`.
   - Helix bias (helix_weight): `-0.2` to reduce excess helical content.
   - Intra-chain contacts (con_weight): `1.0`.
   - AF2 confidence losses (plddt_weight `0.5`, pae_weight `0.5`).
   - Sequence mode at restart: `["gumbel","soft"]`; `rm_aa="C"` (cysteines excluded throughout).
   - Schedule: `design_logits(90)` + `design_logits(10, save_best=True)` = 100 steps. 100 steps matches the paper.
2. **Stage 2 — ProteinMPNN sequence generation.** Discard the RSO relaxed sequence; feed only the converged backbone Cα trace to `mk_mpnn_model(weights="soluble")`. Sample 8 candidates with `temperature=0.1` and `rm_aa="C"`.
3. **Stage 3 — Independent structure-prediction validation.** Using `mk_afdesign_model(protocol="fixbb")` with `model_4_ptm` and `num_recycles=3`, predict each of the 8 MPNN sequences from scratch (no MSA, no template). Compute vs. Stage 1 backbone:
   - **Cα RMSD (Å)** after optimal superposition.
   - **TM-score** (length-normalized structural similarity 0..1).
   - **mean pLDDT** (AF2 local confidence 0..100).
   - **pTM** (AF2 predicted TM-score, from model_4_ptm auxiliary log).
4. **Stage 4 — Ranking.** Rank candidates per-backbone independently by lowest RMSD, highest TM-score, and highest mean pLDDT. No composite ranking is imposed by default; any composite must be documented explicitly.
5. **Length experiment.** Repeat the pipeline for lengths [100, 150, 200] (quick) or [100, 150, 200, 300] (full) × N independent seeds, and compare per-length candidate-quality distributions and runtime.

## Key settings confirmed from official RSO notebook (ColabDesign commit: current `main`; notebook SHA256 recorded in `docs/provenance.md`)

| Parameter | Value (Frank et al. reproduction) |
|---|---|
| RSO protocol | `hallucination` |
| mode on restart | `["gumbel","soft"]` |
| rm_aa (both RSO and MPNN) | `"C"` |
| iterations | 100 total (90 + 10 save_best) |
| copies | 1 (monomer) |
| rg weight | 0.1 for ≤600 aa; 0.01 otherwise |
| helix weight | -0.2 |
| con weight | 1.0 |
| plddt weight | 0.5 |
| pae weight | 0.5 |
| ProteinMPNN weights | `soluble` |
| MPNN num sequences | 8 |
| MPNN temperature | 0.1 |
| Validation model | AlphaFold `model_4_ptm` single-sequence |
| Validation num_recycles | 3 |
| seed (reproduction) | 42 |

## Validation model choice

The project defaults to **AlphaFold2 single-sequence model_4_ptm** for validation (the path used inside `designability_test()` in the official RSO notebook). For ESMFold-based designability screening the official notebook *also* provides a separate cell (CELL 11). Users may enable ESMFold by editing validation configs, but the README must then accurately describe which validation model produced each metric. We do not mix validation models in the same results CSV without tagging.

## Length-dependent heuristics

- rg weight: `0.1` for length ≤ 600, `0.01` for length > 600. Paper uses `0.001` at 1000 aa; the length_experiment_full profile caps at 300 aa by default so `0.1` applies.
- Note on 300 aa: AF2 single-sequence memory use is higher; use A100 40 GB or reduce `copies` / `num_recycles` if OOM occurs.

## Provenance and reproducibility

Every run directory contains `metadata.json` (software versions, GPU info, seeds, config snapshot, git commit) and `status.json` (stage states, stage runtimes, per-candidate status). See `docs/provenance.md`.

## Important limitations

- No wet-lab validation is performed. Computational confidence ≠ experimental success.
- Because AF2 is used both in the design loss and (same-family) validation, method-related bias may inflate self-consistency scores.
- All conclusions are provisional pending real executions that pass `scripts/verify_outputs.py`.
