#!/usr/bin/env python3
"""LOCAL placeholder for Stage 3 validation (AlphaFold/ColabDesign or ESMFold).

Again: actual structure prediction on GPU only (Colab). This script:
  - Prints validation model expected by each config
  - Writes a JSON schema template that GPU notebooks should produce per candidate
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from rso_exploration.config import load_config  # noqa: E402


METRICS_JSON_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Per-candidate validation metrics",
    "type": "object",
    "required": ["rmsd_angstrom", "tm_score", "mean_plddt"],
    "properties": {
        "rmsd_angstrom": {"type": "number", "minimum": 0},
        "tm_score": {"type": "number", "minimum": 0, "maximum": 1},
        "mean_plddt": {"type": "number", "minimum": 0, "maximum": 100},
        "ptm": {"type": "number", "minimum": 0, "maximum": 1},
        "mpnn_runtime_seconds": {"type": "number", "minimum": 0},
        "validation_runtime_seconds": {"type": "number", "minimum": 0},
        "peak_gpu_memory_mb": {"type": "number", "minimum": 0},
        "error": {"type": "string"},
    },
    "additionalProperties": True,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/reproduction_100aa.yaml")
    ap.add_argument("--write-schema-to", default="")
    args = ap.parse_args()
    try:
        cfg = load_config(args.config)
        print(f"Config: {cfg.experiment_name}")
        print(f"Validation model: {cfg.validation.model}")
        print(f"  AlphaFold model: {cfg.validation.alphafold_model}")
        print(f"  num_recycles: {cfg.validation.num_recycles}")
        print(f"  use_initial_guess: {cfg.validation.use_initial_guess}")
        print(f"  use_aa_initialization: {cfg.validation.use_aa_initialization}")
    except Exception as e:
        print(f"[warn] config load failed: {e}", file=sys.stderr)
    if args.write_schema_to:
        p = Path(args.write_schema_to)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(METRICS_JSON_SCHEMA, indent=2) + "\n")
        print(f"Wrote schema -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
