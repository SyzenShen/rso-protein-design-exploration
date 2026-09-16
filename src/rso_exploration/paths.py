from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
RUNS_DIR = RESULTS_DIR / "runs"
FIGURES_DIR = PROJECT_ROOT / "figures"
DOCS_DIR = PROJECT_ROOT / "docs"

ALL_CANDIDATES_CSV = METRICS_DIR / "all_candidates.csv"
BACKBONE_SUMMARY_CSV = METRICS_DIR / "backbone_summary.csv"


def run_dir(run_id: str, base_dir: Optional[Path] = None) -> Path:
    p = Path(base_dir) if base_dir else RUNS_DIR
    d = p / run_id
    return d


def ensure_run_dir(run_id: str, base_dir: Optional[Path] = None) -> Path:
    d = run_dir(run_id, base_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "stage1_rso").mkdir(exist_ok=True)
    (d / "stage2_mpnn").mkdir(exist_ok=True)
    (d / "stage3_validation").mkdir(exist_ok=True)
    return d


def backbone_pdb_path(run_dir_path: Path, backbone_id: str = "bb0") -> Path:
    return run_dir_path / "stage1_rso" / f"{backbone_id}.pdb"


def candidate_fasta_path(run_dir_path: Path, backbone_id: str = "bb0") -> Path:
    return run_dir_path / "stage2_mpnn" / f"{backbone_id}_candidates.fasta"


def candidate_csv_path(run_dir_path: Path, backbone_id: str = "bb0") -> Path:
    return run_dir_path / "stage2_mpnn" / f"{backbone_id}_candidates.csv"


def predicted_pdb_path(run_dir_path: Path, backbone_id: str, candidate_id: str) -> Path:
    return run_dir_path / "stage3_validation" / f"{backbone_id}_{candidate_id}_predicted.pdb"


def metadata_path(run_dir_path: Path) -> Path:
    return run_dir_path / "metadata.json"


def status_path(run_dir_path: Path) -> Path:
    return run_dir_path / "status.json"


def loss_history_path(run_dir_path: Path, backbone_id: str = "bb0") -> Path:
    return run_dir_path / "stage1_rso" / f"{backbone_id}_loss_history.csv"
