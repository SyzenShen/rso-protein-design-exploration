from pathlib import Path
from rso_exploration.paths import (
    PROJECT_ROOT, CONFIGS_DIR, RESULTS_DIR, RUNS_DIR, METRICS_DIR,
    FIGURES_DIR, DOCS_DIR, DATA_DIR,
    run_dir, ensure_run_dir, backbone_pdb_path,
    candidate_fasta_path, predicted_pdb_path,
    metadata_path, status_path,
)


def test_project_paths_exist(tmp_path, monkeypatch):
    for p in [PROJECT_ROOT, CONFIGS_DIR, RESULTS_DIR, RUNS_DIR, METRICS_DIR,
              FIGURES_DIR, DOCS_DIR, DATA_DIR]:
        assert isinstance(p, Path)


def test_run_dir_convention():
    d = run_dir("run_test_001")
    assert d.name == "run_test_001"
    assert "runs" in str(d)


def test_ensure_run_dir_creates_subdirs(tmp_path, monkeypatch):
    base = tmp_path / "runs"
    d = ensure_run_dir("my_run", base_dir=base)
    assert (d / "stage1_rso").is_dir()
    assert (d / "stage2_mpnn").is_dir()
    assert (d / "stage3_validation").is_dir()


def test_file_paths_live_under_run_dir(tmp_path):
    base = tmp_path / "runs"
    rd = ensure_run_dir("r1", base_dir=base)
    assert backbone_pdb_path(rd, "bb0").parent.name == "stage1_rso"
    assert candidate_fasta_path(rd, "bb0").parent.name == "stage2_mpnn"
    pp = predicted_pdb_path(rd, "bb0", "c3")
    assert pp.parent.name == "stage3_validation"
    assert "c3" in pp.name
    assert metadata_path(rd).name == "metadata.json"
    assert status_path(rd).name == "status.json"
