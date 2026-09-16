#!/usr/bin/env python3
"""LOCAL DRY-RUN placeholder for ProteinMPNN stage.

Real ProteinMPNN runs happen on GPU via Colab notebook cells. Running here
without GPU/params would produce FAKE outputs, which is forbidden; therefore
this script only validates inputs and prints usage help.

Usage:
  python scripts/run_mpnn.py --help
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="DRY-RUN placeholder. Real ProteinMPNN runs only on GPU (Colab notebook).")
    ap.add_argument("--config", default="")
    ap.add_argument("--run-dir", default="")
    args = ap.parse_args()
    print("NOTE: run_mpnn.py does NOT locally execute ProteinMPNN.")
    print("ProteinMPNN inference requires ColabDesign and GPU; run the appropriate")
    print("cell block inside notebooks/01_rso_reproduction.ipynb (Colab GPU runtime).")
    print("This wrapper only exists as a documented CLI entry point for pipeline tooling.")
    print(f"  --config: {args.config or '(unset)'}")
    print(f"  --run-dir: {args.run_dir or '(unset)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
