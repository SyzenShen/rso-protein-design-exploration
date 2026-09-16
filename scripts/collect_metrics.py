#!/usr/bin/env python3
"""Collect per-run metadata and status files into results/metrics/*.csv.

Usage:
  python scripts/collect_metrics.py [--runs-dir results/runs]
                                    [--out-csv results/metrics/all_candidates.csv]
                                    [--summary-csv results/metrics/backbone_summary.csv]

For each completed run directory, parses candidate JSON/CSVs and writes the
unified candidates CSV + backbone summary CSV. Does NOT re-run any models.
Safe to re-run (idempotent).
"""
from __future__ import annotations
import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Optional
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from rso_exploration.metrics import (  # noqa: E402
    CandidateRecord, REQUIRED_CANDIDATE_COLUMNS, STATUS_SUCCESS, STATUS_FAILED, STATUS_PENDING,
    save_all_candidates, save_backbone_summary, build_backbone_summary,
)
from rso_exploration.paths import RUNS_DIR, METRICS_DIR, metadata_path, status_path


def _parse_fasta(path: Path) -> list[tuple[str, str]]:
    seqs: list[tuple[str, str]] = []
    current_name: Optional[str] = None
    buf: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if current_name is not None:
                seqs.append((current_name, "".join(buf)))
            current_name = line[1:].split()[0]
            buf = []
        else:
            buf.append(line.strip())
    if current_name is not None:
        seqs.append((current_name, "".join(buf)))
    return seqs


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def scan_run(rd: Path) -> list[CandidateRecord]:
    meta = _read_json(metadata_path(rd))
    stats = _read_json(status_path(rd))
    records: list[CandidateRecord] = []
    stage2 = rd / "stage2_mpnn"
    stage3 = rd / "stage3_validation"
    cfg = meta.get("config_snapshot") or {}
    mpnn_cfg = cfg.get("mpnn") or {}
    rso_cfg = cfg.get("rso") or {}
    val_cfg = cfg.get("validation") or {}

    fasta_files = sorted(stage2.glob("*_candidates.fasta"))
    csv_files = sorted(stage2.glob("*_candidates.csv"))
    mpnn_scores: dict[tuple[str, str], float] = {}
    for cpath in csv_files:
        try:
            cdf = pd.read_csv(cpath)
            bb = cpath.stem.replace("_candidates", "")
            if "score" in cdf.columns and "seq" in cdf.columns and "candidate_id" in cdf.columns:
                for _, row in cdf.iterrows():
                    mpnn_scores[(bb, str(row["candidate_id"]))] = float(row["score"])
        except Exception:
            pass

    for fpath in fasta_files:
        backbone_id = fpath.stem.replace("_candidates", "")
        bb_status = (stats.get("backbones") or {}).get(backbone_id, {})
        rso_runtime = bb_status.get("rso_runtime_seconds") or meta.get("rso_runtime_seconds")
        final_loss = bb_status.get("final_loss") or meta.get("rso_final_loss")
        for cand_id, seq in _parse_fasta(fpath):
            pred_pdb = stage3 / f"{backbone_id}_{cand_id}_predicted.pdb"
            metrics_json = stage3 / f"{backbone_id}_{cand_id}_metrics.json"
            m = _read_json(metrics_json) if metrics_json.exists() else {}
            err = None
            st = STATUS_PENDING
            if pred_pdb.exists():
                st = STATUS_SUCCESS
            if m.get("error"):
                st = STATUS_FAILED
                err = str(m.get("error"))[:500]
            # pLDDT: canonical scale is 0-100. ColabDesign's get_plddt() is native 0-1;
            # notebooks record the conversion in "plddt_scale". Coerce unlabelled legacy
            # 0-1 values defensively so the CSV stays on a single scale.
            _plddt = float(m["mean_plddt"]) if m.get("mean_plddt") is not None else None
            _plddt_scale = m.get("plddt_scale")
            if _plddt is not None and _plddt_scale is None and 0.0 <= _plddt <= 1.0:
                _plddt *= 100.0
                _plddt_scale = "auto_scaled_native_0_1_x100"
            r = CandidateRecord(
                experiment_name=str(meta.get("experiment_name") or rd.name),
                run_id=rd.name,
                timestamp=str(meta.get("timestamp_utc") or meta.get("created_at") or ""),
                length=int(rso_cfg.get("length") or m.get("length") or len(seq) or 0),
                copies=int(cfg.get("copies") or rso_cfg.get("copies") or 1),
                seed=int(meta.get("random_seed") or cfg.get("seed") or 0),
                backbone_id=backbone_id,
                candidate_id=cand_id,
                sequence=seq,
                rso_iterations=int(rso_cfg.get("iterations") or 0) or None,
                rso_final_loss=float(final_loss) if final_loss is not None else None,
                mpnn_score=mpnn_scores.get((backbone_id, cand_id)),
                mpnn_temperature=float(mpnn_cfg.get("temperature") or 0.0) or None,
                validation_model=str(val_cfg.get("model") or meta.get("validation_model") or ""),
                rmsd_angstrom=float(m["rmsd_angstrom"]) if m.get("rmsd_angstrom") is not None else None,
                rmsd_source=m.get("rmsd_source"),
                tm_score=float(m["tm_score"]) if m.get("tm_score") is not None else None,
                tm_source=m.get("tm_source"),
                mean_plddt=_plddt,
                plddt_scale=_plddt_scale,
                ptm=float(m["ptm"]) if m.get("ptm") is not None else None,
                rso_runtime_seconds=float(rso_runtime) if rso_runtime is not None else None,
                mpnn_runtime_seconds=float(m.get("mpnn_runtime_seconds") or bb_status.get("mpnn_runtime_seconds") or 0) or None,
                validation_runtime_seconds=float(m.get("validation_runtime_seconds")) if m.get("validation_runtime_seconds") is not None else None,
                total_runtime_seconds=None,
                gpu_name=meta.get("gpu_name"),
                peak_gpu_memory_mb=meta.get("gpu_memory_mb") or m.get("peak_gpu_memory_mb"),
                colabdesign_commit=meta.get("colabdesign_commit"),
                git_commit=meta.get("git_commit"),
                status=st,
                error_message=err,
            )
            records.append(r)
    return records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(RUNS_DIR))
    ap.add_argument("--out-csv", default=str(METRICS_DIR / "all_candidates.csv"))
    ap.add_argument("--summary-csv", default=str(METRICS_DIR / "backbone_summary.csv"))
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    all_records: list[CandidateRecord] = []
    if runs_dir.is_dir():
        for rd in sorted([p for p in runs_dir.iterdir() if p.is_dir()]):
            try:
                all_records.extend(scan_run(rd))
            except Exception as e:
                print(f"[warn] skip {rd.name}: {e}", file=sys.stderr)
    print(f"Scanned runs: {len([p for p in runs_dir.iterdir() if p.is_dir()]) if runs_dir.is_dir() else 0}")
    print(f"Total candidate rows collected: {len(all_records)}")

    if all_records:
        df = pd.DataFrame([r.to_row() for r in all_records])
    else:
        df = pd.DataFrame(columns=REQUIRED_CANDIDATE_COLUMNS)
    save_all_candidates(df, Path(args.out_csv))
    summ = build_backbone_summary(df)
    save_backbone_summary(summ, Path(args.summary_csv))
    print(f"Wrote {args.out_csv} ({len(df)} rows)")
    print(f"Wrote {args.summary_csv} ({len(summ)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
