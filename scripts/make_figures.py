#!/usr/bin/env python3
"""Generate figures from results/metrics/all_candidates.csv.

Usage:
  python scripts/make_figures.py [--candidates-csv results/metrics/all_candidates.csv]
                                 [--figures-dir figures]

If no real numeric data exists, figures are NOT generated and the script
exits with a clear message. Never produces fake plots.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from rso_exploration.paths import METRICS_DIR, FIGURES_DIR, RUNS_DIR, loss_history_path  # noqa: E402
from rso_exploration.metrics import load_all_candidates  # noqa: E402
from rso_exploration.plotting import generate_all_figures, plot_loss_history  # noqa: E402


def _collect_loss_histories(runs_dir: Path) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    if not runs_dir.is_dir():
        return out
    for rd in sorted([p for p in runs_dir.iterdir() if p.is_dir()]):
        s1 = rd / "stage1_rso"
        for csv_path in sorted(s1.glob("*_loss_history.csv")):
            key = f"{rd.name}__{csv_path.stem.replace('_loss_history','')}"
            try:
                df = pd.read_csv(csv_path)
                if "step" in df.columns:
                    df = df.set_index("step")
                out[key] = df
            except Exception:
                pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates-csv", default=str(METRICS_DIR / "all_candidates.csv"))
    ap.add_argument("--figures-dir", default=str(FIGURES_DIR))
    args = ap.parse_args()

    df = load_all_candidates(Path(args.candidates_csv))
    if df.empty:
        print("[info] No candidate rows found. No figures generated.", file=sys.stderr)
        return 0
    numeric = {c: pd.to_numeric(df[c], errors="coerce").notna().sum()
               for c in ["rmsd_angstrom", "tm_score", "mean_plddt", "total_runtime_seconds"]}
    print(f"[info] Candidates loaded: {len(df)}")
    print(f"[info] Numeric value counts: {numeric}")

    losses = _collect_loss_histories(RUNS_DIR)
    print(f"[info] Loss histories loaded: {len(losses)}")
    generated = generate_all_figures(df, loss_history_dfs=losses or None)
    if not generated:
        print("[info] No real numeric data. No figures generated (no synthetic plots).")
    else:
        for p in generated:
            print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
