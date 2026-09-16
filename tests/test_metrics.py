import pandas as pd
import pytest
from rso_exploration.metrics import (
    CandidateRecord, REQUIRED_CANDIDATE_COLUMNS,
    load_all_candidates, save_all_candidates, append_candidates,
    build_backbone_summary, rank_candidates, validate_candidates_schema,
    STATUS_SUCCESS, STATUS_FAILED, STATUS_PENDING,
)
from rso_exploration.paths import METRICS_DIR


def _make_records():
    return [
        CandidateRecord(
            experiment_name="exp1", run_id="r1", timestamp="2026-01-01T00:00:00Z",
            length=100, copies=1, seed=42, backbone_id="bb0", candidate_id="c0",
            sequence="ACDEFGHIKLMNPQRSTVW"*5,
            rso_iterations=100, rso_final_loss=0.5,
            mpnn_score=-100.0, mpnn_temperature=0.1,
            validation_model="alphafold_single",
            rmsd_angstrom=1.5, tm_score=0.9, mean_plddt=85.0, ptm=0.8,
            rso_runtime_seconds=60.0, mpnn_runtime_seconds=10.0,
            validation_runtime_seconds=120.0, total_runtime_seconds=190.0,
            gpu_name="Tesla T4", peak_gpu_memory_mb=8000.0,
            colabdesign_commit="abc123", git_commit="def456",
            status=STATUS_SUCCESS, error_message=None,
        ),
        CandidateRecord(
            experiment_name="exp1", run_id="r1", timestamp="2026-01-01T00:00:00Z",
            length=100, copies=1, seed=42, backbone_id="bb0", candidate_id="c1",
            sequence="ACDEFGHIKLMNPQRSTVW"*5,
            rso_iterations=100, rso_final_loss=0.5,
            mpnn_score=-95.0, mpnn_temperature=0.1,
            validation_model="alphafold_single",
            rmsd_angstrom=3.0, tm_score=0.7, mean_plddt=70.0, ptm=0.6,
            status=STATUS_SUCCESS,
        ),
        CandidateRecord(
            experiment_name="exp1", run_id="r2", timestamp="2026-01-01T00:00:05Z",
            length=150, copies=1, seed=123, backbone_id="bb0", candidate_id="c0",
            sequence="A"*150,
            status=STATUS_FAILED, error_message="CUDA OOM",
        ),
    ]


def test_candidate_record_sets_sequence_length():
    r = CandidateRecord(
        experiment_name="e", run_id="r", timestamp="t",
        length=10, copies=1, seed=0, backbone_id="b", candidate_id="c",
        sequence="AAAAA",
    )
    assert r.sequence_length == 5


def test_append_and_roundtrip(tmp_path, monkeypatch):
    csv_path = tmp_path / "all_candidates.csv"
    monkeypatch.setattr("rso_exploration.paths.ALL_CANDIDATES_CSV", csv_path)
    recs = _make_records()
    append_candidates(recs, path=csv_path)
    df = load_all_candidates(csv_path)
    assert len(df) == 3
    errs = validate_candidates_schema(df)
    assert len(errs) == 0, f"Schema errors: {errs}"


def test_schema_has_all_required_columns(tmp_path):
    csv_path = tmp_path / "x.csv"
    save_all_candidates(pd.DataFrame(), path=csv_path)
    df = load_all_candidates(csv_path)
    for c in REQUIRED_CANDIDATE_COLUMNS:
        assert c in df.columns


def test_build_backbone_summary_ranks_and_counts():
    recs = _make_records()
    df = pd.DataFrame([r.to_row() for r in recs])
    summ = build_backbone_summary(df)
    # Two groups: (exp1, r1, 100, 42, bb0) and (exp1, r2, 150, 123, bb0)
    assert len(summ) == 2
    row_success = summ[(summ["run_id"] == "r1")].iloc[0]
    assert row_success["num_successful"] == 2
    assert row_success["best_rmsd_angstrom"] == 1.5
    assert row_success["best_tm_score"] == 0.9
    assert row_success["best_mean_plddt"] == 85.0
    row_fail = summ[(summ["run_id"] == "r2")].iloc[0]
    assert row_fail["num_successful"] == 0
    assert pd.isna(row_fail["best_rmsd_angstrom"]) or row_fail["best_rmsd_angstrom"] is None


def test_rank_candidates_adds_rank_columns():
    recs = _make_records()
    df = pd.DataFrame([r.to_row() for r in recs])
    ranked = rank_candidates(df, backbone_id="bb0")
    assert "rank_by_rmsd" in ranked.columns
    assert "rank_by_tm" in ranked.columns
    assert "rank_by_plddt" in ranked.columns


def test_validate_schema_catches_missing_columns():
    df = pd.DataFrame({"experiment_name": ["e1"]})
    errs = validate_candidates_schema(df)
    assert any("Missing required columns" in e for e in errs)


def test_validate_schema_catches_length_mismatch():
    recs = _make_records()
    df = pd.DataFrame([r.to_row() for r in recs])
    df.loc[0, "sequence_length"] = 99999
    errs = validate_candidates_schema(df)
    assert any("sequence_length mismatch" in e for e in errs)
