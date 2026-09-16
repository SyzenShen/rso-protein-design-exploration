#!/usr/bin/env python3
"""Verify an experiment: files present, schema valid, PDBs non-empty, metrics complete.

Usage:
  python scripts/verify_outputs.py [--runs-dir results/runs]
                                   [--candidates-csv results/metrics/all_candidates.csv]
                                   [--summary-csv results/metrics/backbone_summary.csv]
                                   [--require-real-metrics / --allow-empty]

Exit code 0 if checks pass; 1 otherwise.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from rso_exploration.metrics import (  # noqa: E402
    load_all_candidates, validate_candidates_schema,
    REQUIRED_CANDIDATE_COLUMNS, STATUS_SUCCESS,
)
from rso_exploration.paths import (  # noqa: E402
    RUNS_DIR, METRICS_DIR, metadata_path, status_path,
    backbone_pdb_path, candidate_fasta_path, predicted_pdb_path,
)


def _has_atom_records(pdb: Path) -> bool:
    if not pdb.is_file():
        return False
    txt = pdb.read_text(errors="ignore")
    return any(line.startswith("ATOM  ") or line.startswith("HETATM") for line in txt.splitlines())


def _fasta_sequences_ok(fasta: Path, expected_min_len: int = 20) -> tuple[bool, int]:
    if not fasta.is_file():
        return False, 0
    count = 0
    min_len = 10**9
    cur: list[str] = []
    for line in fasta.read_text().splitlines():
        if line.startswith(">"):
            if cur:
                s = "".join(cur)
                count += 1
                min_len = min(min_len, len(s))
                cur = []
        else:
            cur.append(line.strip())
    if cur:
        s = "".join(cur)
        count += 1
        min_len = min(min_len, len(s))
    if count == 0:
        return False, 0
    return (count > 0 and min_len >= expected_min_len), count


def check_run(rd: Path, errors: list[str], warnings: list[str]) -> None:
    meta = json.loads(metadata_path(rd).read_text()) if metadata_path(rd).exists() else None
    st = json.loads(status_path(rd).read_text()) if status_path(rd).exists() else None
    if meta is None:
        errors.append(f"{rd.name}: missing metadata.json")
    if st is None:
        warnings.append(f"{rd.name}: missing status.json")

    s2 = rd / "stage2_mpnn"
    s3 = rd / "stage3_validation"
    for fasta in sorted(s2.glob("*_candidates.fasta")):
        bb = fasta.stem.replace("_candidates", "")
        bp = backbone_pdb_path(rd, bb)
        if not _has_atom_records(bp):
            errors.append(f"{rd.name}/{bb}: backbone PDB missing/empty ATOM records")
        ok, n_seq = _fasta_sequences_ok(fasta)
        if not ok:
            errors.append(f"{rd.name}/{bb}: FASTA has no sequences or too short")
        # check prediction files
        for i in range(n_seq):
            cid = f"c{i}"
            pred = predicted_pdb_path(rd, bb, cid)
            m_file = s3 / f"{bb}_{cid}_metrics.json"
            if not pred.exists():
                warnings.append(f"{rd.name}/{bb}/{cid}: predicted PDB not yet written")
            elif not _has_atom_records(pred):
                errors.append(f"{rd.name}/{bb}/{cid}: predicted PDB has no ATOM records")
            if not m_file.exists():
                warnings.append(f"{rd.name}/{bb}/{cid}: metrics JSON missing")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(RUNS_DIR))
    ap.add_argument("--candidates-csv", default=str(METRICS_DIR / "all_candidates.csv"))
    ap.add_argument("--summary-csv", default=str(METRICS_DIR / "backbone_summary.csv"))
    ap.add_argument("--require-real-metrics", action="store_true",
                    help="Fail if no successful candidate metrics exist")
    ap.add_argument("--allow-empty", action="store_true",
                    help="Allow empty CSVs (pre-execution state)")
    args = ap.parse_args()

    errors: list[str] = []
    warnings: list[str] = []

    runs_dir = Path(args.runs_dir)
    if runs_dir.is_dir():
        for rd in sorted([p for p in runs_dir.iterdir() if p.is_dir()]):
            check_run(rd, errors, warnings)

    c_csv = Path(args.candidates_csv)
    s_csv = Path(args.summary_csv)
    if not c_csv.exists():
        (errors if args.require_real_metrics else warnings).append(
            f"candidates CSV missing: {c_csv}")
    else:
        df = load_all_candidates(c_csv)
        for e in validate_candidates_schema(df):
            errors.append(f"schema: {e}")
        if args.require_real_metrics and df.empty:
            errors.append("--require-real-metrics set but candidates CSV is empty")
        if not df.empty:
            success = df[df["status"] == STATUS_SUCCESS]
            if args.require_real_metrics and success.empty:
                errors.append("--require-real-metrics set but 0 successful candidates")

    if not s_csv.exists() and args.require_real_metrics:
        errors.append(f"summary CSV missing: {s_csv}")

    print(f"Errors: {len(errors)}")
    for e in errors:
        print(f"  ERROR: {e}")
    print(f"Warnings: {len(warnings)}")
    for w in warnings:
        print(f"  WARN:  {w}")
    if errors:
        print("VERIFICATION: FAILED")
        return 1
    print("VERIFICATION: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
