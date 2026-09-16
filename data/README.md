# Data Directory

This directory holds small reference data files used by the pipeline.
Only small PDBs, FASTA, CSV, YAML files and provenance logs are tracked
in git (see top-level `.gitignore`).

## Subdirectories

### `external/`
Downloaded reference PDB structures used as fixed backbones for design.

**Reference structures to obtain:**

| PDB ID | Source | Purpose |
|--------|--------|---------|
| 8S89   | [RCSB Protein Data Bank](https://www.rcsb.org/structure/8S89) | Example fixed backbone from Frank et al. 2024 |

**How to download 8S89:**
- Via browser: <https://www.rcsb.org/structure/8S89> → Download Files → PDB Format
- Via CLI (curl / wget):
  ```
  curl -o external/8S89.pdb "https://files.rcsb.org/download/8S89.pdb"
  ```

## Important Notes

- **NO model parameters, AlphaFold 2 weights, or large checkpoint files are committed to this repository.** Those are downloaded at runtime on the compute backend (Colab / GPU node) and are excluded via `.gitignore`.
- Keep individual data files small (<~10 MB each) if committing to git.
