import json
from pathlib import Path
import pandas as pd
import pytest
from rso_exploration.paths import (
    run_dir, ensure_run_dir, backbone_pdb_path,
    candidate_fasta_path, candidate_csv_path, predicted_pdb_path,
    metadata_path, status_path, loss_history_path,
)
from rso_exploration.provenance import ProvenanceRecord, collect_provenance, save_provenance, load_provenance


def _write_dummy_pdb(path: Path, length: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for i in range(1, length + 1):
        lines.append(f"ATOM  {i:5d}  CA  ALA A{i:4d}    8.000   8.000   8.000  1.00 20.00           C")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


def test_run_complete_layout(tmp_path):
    base = tmp_path / "runs"
    rd = ensure_run_dir("run_abc", base_dir=base)
    bb = "bb0"
    _write_dummy_pdb(backbone_pdb_path(rd, bb), 100)
    # FASTA
    fasta = candidate_fasta_path(rd, bb)
    fasta.write_text(">c0\nAAAAA\n>c1\nTTTTT\n")
    # CSV
    pd.DataFrame([
        {"candidate_id": "c0", "score": -1.0, "seq": "AAAAA"},
        {"candidate_id": "c1", "score": -0.8, "seq": "TTTTT"},
    ]).to_csv(candidate_csv_path(rd, bb), index=False)
    # Predicted PDBs
    for cid in ["c0", "c1"]:
        _write_dummy_pdb(predicted_pdb_path(rd, bb, cid), 100)
    # status and metadata
    status_path(rd).write_text(json.dumps({"status": "success", "backbones": {"bb0": {"status": "success"}}}))
    # loss history minimal
    loss_history_path(rd, bb).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"step": [0], "total": [1.0]}).to_csv(loss_history_path(rd, bb), index=False)
    # metadata + provenance
    prov = collect_provenance(seed=42)
    save_provenance(prov, metadata_path(rd))

    assert backbone_pdb_path(rd, bb).is_file()
    assert candidate_fasta_path(rd, bb).is_file()
    assert candidate_csv_path(rd, bb).is_file()
    assert predicted_pdb_path(rd, bb, "c0").is_file()
    assert metadata_path(rd).is_file()
    assert status_path(rd).is_file()
    assert loss_history_path(rd, bb).is_file()


def test_provenance_record_serialises(tmp_path):
    prov = ProvenanceRecord(
        timestamp_utc="2026-01-01",
        colabdesign_commit="abcd",
        python_version="3.10.11",
        random_seed=42,
        mpnn_weights_type="soluble",
    )
    p = tmp_path / "prov.json"
    from rso_exploration.provenance import save_provenance, load_provenance
    save_provenance(prov, p)
    loaded = load_provenance(p)
    assert loaded.colabdesign_commit == "abcd"
    assert loaded.random_seed == 42


def test_detect_provenance_runs_safely():
    prov = collect_provenance(seed=0)
    assert prov.timestamp_utc
    assert isinstance(prov.jax_devices, list)
