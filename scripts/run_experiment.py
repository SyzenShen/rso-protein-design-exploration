#!/usr/bin/env python3
"""Convenience driver: load a YAML experiment, scaffold run dirs, collect, plot.

Usage:
  python scripts/run_experiment.py --config configs/reproduction_100aa.yaml --scaffold-only
  python scripts/run_experiment.py --collect-only
  python scripts/run_experiment.py --plot-only
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def _run(script: str, extra_args: list[str]) -> int:
    cmd = [sys.executable, str(SCRIPTS_DIR / script)] + extra_args
    print("+", " ".join(cmd))
    return subprocess.call(cmd)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/reproduction_100aa.yaml")
    ap.add_argument("--scaffold-only", action="store_true")
    ap.add_argument("--collect-only", action="store_true")
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    if sum([args.scaffold_only, args.collect_only, args.plot_only, args.verify]) == 0:
        ap.print_help()
        return 2

    rc = 0
    if args.scaffold_only:
        rc |= _run("run_rso.py", ["--config", args.config, "--dry-run"])
    if args.collect_only:
        rc |= _run("collect_metrics.py", [])
    if args.plot_only:
        rc |= _run("make_figures.py", [])
    if args.verify:
        rc |= _run("verify_outputs.py", [])
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
