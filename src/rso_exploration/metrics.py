from __future__ import annotations
import csv
import json
import math
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Optional
import pandas as pd
from .paths import ALL_CANDIDATES_CSV, BACKBONE_SUMMARY_CSV

REQUIRED_CANDIDATE_COLUMNS = [
    "experiment_name", "run_id", "timestamp", "length", "copies", "seed",
    "backbone_id", "candidate_id", "sequence", "sequence_length",
    "rso_iterations", "rso_final_loss",
    "mpnn_score", "mpnn_temperature",
    "validation_model",
    "rmsd_angstrom", "tm_score", "mean_plddt", "ptm",
    "rso_runtime_seconds", "mpnn_runtime_seconds",
    "validation_runtime_seconds", "total_runtime_seconds",
    "gpu_name", "peak_gpu_memory_mb",
    "colabdesign_commit", "git_commit",
    "status", "error_message",
]

STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"
STATUS_PENDING = "pending"
STATUS_SKIPPED = "skipped"


@dataclass
class CandidateRecord:
    experiment_name: str
    run_id: str
    timestamp: str
    length: int
    copies: int
    seed: int
    backbone_id: str
    candidate_id: str
    sequence: str
    sequence_length: Optional[int] = None
    rso_iterations: Optional[int] = None
    rso_final_loss: Optional[float] = None
    mpnn_score: Optional[float] = None
    mpnn_temperature: Optional[float] = None
    validation_model: Optional[str] = None
    rmsd_angstrom: Optional[float] = None
    rmsd_source: Optional[str] = None
    tm_score: Optional[float] = None
    tm_source: Optional[str] = None
    mean_plddt: Optional[float] = None
    plddt_scale: Optional[str] = None
    ptm: Optional[float] = None
    rso_runtime_seconds: Optional[float] = None
    mpnn_runtime_seconds: Optional[float] = None
    validation_runtime_seconds: Optional[float] = None
    total_runtime_seconds: Optional[float] = None
    gpu_name: Optional[str] = None
    peak_gpu_memory_mb: Optional[float] = None
    colabdesign_commit: Optional[str] = None
    git_commit: Optional[str] = None
    status: str = STATUS_PENDING
    error_message: Optional[str] = None

    def __post_init__(self) -> None:
        if self.sequence_length is None and self.sequence:
            self.sequence_length = len(self.sequence)

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


def load_all_candidates(path: Optional[Path] = None) -> pd.DataFrame:
    p = path or ALL_CANDIDATES_CSV
    if not p.exists():
        return pd.DataFrame(columns=REQUIRED_CANDIDATE_COLUMNS)
    return pd.read_csv(p, dtype=str).fillna(pd.NA)


def save_all_candidates(df: pd.DataFrame, path: Optional[Path] = None) -> None:
    p = path or ALL_CANDIDATES_CSV
    p.parent.mkdir(parents=True, exist_ok=True)
    for col in REQUIRED_CANDIDATE_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df[REQUIRED_CANDIDATE_COLUMNS].to_csv(p, index=False)


def append_candidates(records: list[CandidateRecord], path: Optional[Path] = None) -> pd.DataFrame:
    df = load_all_candidates(path)
    new_rows = pd.DataFrame([r.to_row() for r in records])
    combined = pd.concat([df, new_rows], ignore_index=True)
    save_all_candidates(combined, path)
    return combined


def build_backbone_summary(candidates_df: pd.DataFrame) -> pd.DataFrame:
    if candidates_df.empty:
        return pd.DataFrame(columns=[
            "experiment_name", "run_id", "length", "seed", "backbone_id",
            "num_candidates", "num_successful",
            "best_rmsd_angstrom", "best_tm_score", "best_mean_plddt",
            "total_runtime_seconds",
        ])
    numeric_cols = ["rmsd_angstrom", "tm_score", "mean_plddt",
                    "rso_runtime_seconds", "mpnn_runtime_seconds",
                    "validation_runtime_seconds", "total_runtime_seconds"]
    for c in numeric_cols:
        candidates_df[c] = pd.to_numeric(candidates_df[c], errors="coerce")
    group_keys = ["experiment_name", "run_id", "length", "seed", "backbone_id"]
    rows = []
    for keys, g in candidates_df.groupby(group_keys, dropna=False):
        success = g[g["status"] == STATUS_SUCCESS]
        total = g["total_runtime_seconds"].sum()
        row = {
            "experiment_name": keys[0], "run_id": keys[1],
            "length": keys[2], "seed": keys[3], "backbone_id": keys[4],
            "num_candidates": len(g),
            "num_successful": len(success),
            "best_rmsd_angstrom": success["rmsd_angstrom"].min() if not success.empty else None,
            "best_tm_score": success["tm_score"].max() if not success.empty else None,
            "best_mean_plddt": success["mean_plddt"].max() if not success.empty else None,
            "total_runtime_seconds": total if not pd.isna(total) else None,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def save_backbone_summary(df: pd.DataFrame, path: Optional[Path] = None) -> None:
    p = path or BACKBONE_SUMMARY_CSV
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)


def rank_candidates(candidates_df: pd.DataFrame, backbone_id: Optional[str] = None) -> pd.DataFrame:
    df = candidates_df.copy()
    if backbone_id is not None:
        df = df[df["backbone_id"] == backbone_id]
    if df.empty:
        return df
    df["rmsd_angstrom"] = pd.to_numeric(df["rmsd_angstrom"], errors="coerce")
    df["tm_score"] = pd.to_numeric(df["tm_score"], errors="coerce")
    df["mean_plddt"] = pd.to_numeric(df["mean_plddt"], errors="coerce")
    df["rank_by_rmsd"] = df["rmsd_angstrom"].rank(method="min", ascending=True, na_option="bottom")
    df["rank_by_tm"] = df["tm_score"].rank(method="min", ascending=False, na_option="bottom")
    df["rank_by_plddt"] = df["mean_plddt"].rank(method="min", ascending=False, na_option="bottom")
    return df


def validate_candidates_schema(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    missing = [c for c in REQUIRED_CANDIDATE_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")
    if not df.empty and "sequence" in df.columns and "sequence_length" in df.columns:
        seq_lens = df["sequence"].astype(str).str.len().astype("Int64")
        declared_lens = pd.to_numeric(df["sequence_length"], errors="coerce").astype("Int64")
        neq = seq_lens != declared_lens
        na_mask = seq_lens.isna() | declared_lens.isna()
        bad_len = df[neq & ~na_mask]
        if not bad_len.empty:
            errors.append(f"{len(bad_len)} rows have sequence_length mismatch")
    return errors
