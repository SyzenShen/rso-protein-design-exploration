#!/usr/bin/env python3
"""LOCAL DRY-RUN ONLY wrapper for RSO stage.

Real execution requires GPU + ColabDesign install and lives in notebooks (or
Google Colab). This script:

  - Loads a YAML config.
  - Validates it.
  - Creates empty run directories + status files in 'pending' state.
  - Prints a report and optionally writes a shell snippet showing how to
    invoke the real ColabDesign design loop (for user copy-paste on GPU).

Usage:
  python scripts/run_rso.py --config configs/reproduction_100aa.yaml --dry-run
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from rso_exploration.config import load_config, validate_config  # noqa: E402
from rso_exploration.paths import ensure_run_dir, RUNS_DIR, metadata_path, status_path  # noqa: E402
from rso_exploration.provenance import collect_provenance, save_provenance  # noqa: E402


def make_run_id(cfg, length: int, seed: int) -> str:
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    h = hashlib.sha1(f"{cfg.experiment_name}|{length}|{seed}".encode()).hexdigest()[:6]
    return f"{cfg.experiment_name}_L{length}_s{seed}_{stamp}_{h}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config path")
    ap.add_argument("--dry-run", action="store_true",
                    help="Required: only dry-run scaffolding is supported locally.")
    args = ap.parse_args()

    if not args.dry_run:
        print("ERROR: Local real RSO execution is disabled by default.", file=sys.stderr)
        print("This project expects GPU execution via notebooks/colab.", file=sys.stderr)
        print("Pass --dry-run for scaffolding only.", file=sys.stderr)
        return 2

    cfg = load_config(args.config)
    errs = validate_config(cfg)
    if errs:
        for e in errs:
            print(f"CONFIG ERROR: {e}", file=sys.stderr)
        return 2

    run_ids: list[str] = []
    for length in cfg.effective_lengths():
        for seed in cfg.effective_seeds():
            rid = make_run_id(cfg, length, seed)
            rd = ensure_run_dir(rid)
            prov = collect_provenance(seed=seed, config_snapshot=cfg.to_dict(),
                                      mpnn_weights=cfg.mpnn.weights)
            save_provenance(prov, metadata_path(rd))
            status_path(rd).write_text(json.dumps({
                "status": "pending",
                "stage": "rso_not_started",
                "config": {"length": length, "seed": seed,
                           "experiment_name": cfg.experiment_name},
                "backbones": {},
            }, indent=2))
            run_ids.append(rid)
            print(f"[dry-run] scaffolded {rd}")
    print()
    print(f"Total runs scaffolded: {len(run_ids)}")
    print("Next: upload project + configs to Google Colab, attach GPU runtime,")
    print("and run notebooks/01_rso_reproduction.ipynb (or 02_length_experiment.ipynb).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
