from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
import seaborn as sns
from .paths import FIGURES_DIR

sns.set_theme(style="whitegrid", context="paper")


def read_ca_xyz(pdb_path: Path) -> np.ndarray:
    """Extract CA coordinates (Nx3, residue order) from a PDB file."""
    xyz = []
    for line in Path(pdb_path).read_text(errors="ignore").splitlines():
        if line.startswith("ATOM  ") and line[12:16].strip() == "CA":
            xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.asarray(xyz, dtype=float)


def kabsch_align(mod_ca: np.ndarray, ref_ca: np.ndarray) -> np.ndarray:
    """Rigid-body superposition of mod onto ref (equal-length Nx3 CA coords)."""
    n = min(len(mod_ca), len(ref_ca))
    P = mod_ca[:n] - mod_ca[:n].mean(axis=0)
    q_mean = ref_ca[:n].mean(axis=0)
    Q = ref_ca[:n] - q_mean
    H = P.T @ Q
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return P @ R.T + q_mean


def plot_structure_overlay(ref_pdb: Path, pred_pdb: Path,
                           out_path: Optional[Path] = None,
                           title: Optional[str] = None,
                           rmsd: Optional[float] = None,
                           tm: Optional[float] = None) -> Optional[Path]:
    """Overlay the designed backbone CA trace and the best AF2 prediction (Kabsch-aligned)."""
    ref_ca, pred_ca = read_ca_xyz(ref_pdb), read_ca_xyz(pred_pdb)
    n = min(len(ref_ca), len(pred_ca))
    if n < 10:
        return None
    aligned = kabsch_align(pred_ca, ref_ca)
    if rmsd is None:
        rmsd = float(np.sqrt(np.mean(np.sum((ref_ca[:n] - aligned[:n]) ** 2, axis=-1))))

    fig = plt.figure(figsize=(7.2, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(ref_ca[:n, 0], ref_ca[:n, 1], ref_ca[:n, 2],
            color="#1f4e79", lw=2.4, label="RSO-designed backbone")
    ax.plot(aligned[:, 0], aligned[:, 1], aligned[:, 2],
            color="#d62728", lw=1.4, alpha=0.85, label="AF2 prediction (MPNN seq.)")
    # equal aspect
    pts = np.vstack([ref_ca[:n], aligned])
    mins, maxs = pts.min(axis=0), pts.max(axis=0)
    ctr = (mins + maxs) / 2
    half = (maxs - mins).max() / 2 * 1.05
    ax.set_xlim(ctr[0]-half, ctr[0]+half)
    ax.set_ylim(ctr[1]-half, ctr[1]+half)
    ax.set_zlim(ctr[2]-half, ctr[2]+half)
    ax.set_xlabel("x (Å)"); ax.set_ylabel("y (Å)"); ax.set_zlabel("z (Å)")
    ax.set_title(title or f"Structure overlay (CA, {n} residues)")
    sub = f"CA RMSD {rmsd:.2f} Å" + (f", TM-score {tm:.3f}" if tm is not None else "")
    ax.text2D(0.5, 0.01, sub, transform=ax.transAxes, ha="center", va="bottom", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    p = Path(out_path) if out_path else FIGURES_DIR / "structure_overlay.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p



def _require_real_data(df: pd.DataFrame, col: str) -> Optional[pd.Series]:
    if df.empty:
        return None
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    return s if len(s) > 0 else None


def plot_rmsd_vs_length(df: pd.DataFrame, out_path: Optional[Path] = None) -> Optional[Path]:
    s = _require_real_data(df, "rmsd_angstrom")
    if s is None:
        return None
    fig, ax = plt.subplots(figsize=(6, 4.5))
    work = df.copy()
    work["rmsd_angstrom"] = pd.to_numeric(work["rmsd_angstrom"], errors="coerce")
    work["length"] = pd.to_numeric(work["length"], errors="coerce")
    work = work.dropna(subset=["rmsd_angstrom", "length"])
    if work.empty:
        plt.close(fig)
        return None
    sns.stripplot(data=work, x="length", y="rmsd_angstrom", ax=ax,
                  size=4, alpha=0.7, color="steelblue", zorder=1)
    sns.pointplot(data=work, x="length", y="rmsd_angstrom", ax=ax,
                  color="crimson", join=False, ci="sd", markers="D", scale=0.7)
    ax.set_xlabel("Protein length (amino acids)")
    ax.set_ylabel("RMSD (Angstrom) vs. designed backbone")
    ax.set_title(f"Candidate RMSD vs. length (n={len(work)})")
    fig.tight_layout()
    p = Path(out_path) if out_path else FIGURES_DIR / "rmsd_vs_length.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def plot_tmscore_vs_length(df: pd.DataFrame, out_path: Optional[Path] = None) -> Optional[Path]:
    work = df.copy()
    work["tm_score"] = pd.to_numeric(work["tm_score"], errors="coerce")
    work["length"] = pd.to_numeric(work["length"], errors="coerce")
    work = work.dropna(subset=["tm_score", "length"])
    if work.empty:
        return None
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.stripplot(data=work, x="length", y="tm_score", ax=ax,
                  size=4, alpha=0.7, color="steelblue", zorder=1)
    sns.pointplot(data=work, x="length", y="tm_score", ax=ax,
                  color="forestgreen", join=False, ci="sd", markers="D", scale=0.7)
    ax.set_xlabel("Protein length (amino acids)")
    ax.set_ylabel("TM-score vs. designed backbone")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Candidate TM-score vs. length (n={len(work)})")
    fig.tight_layout()
    p = Path(out_path) if out_path else FIGURES_DIR / "tmscore_vs_length.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def plot_plddt_vs_length(df: pd.DataFrame, out_path: Optional[Path] = None) -> Optional[Path]:
    work = df.copy()
    work["mean_plddt"] = pd.to_numeric(work["mean_plddt"], errors="coerce")
    work["length"] = pd.to_numeric(work["length"], errors="coerce")
    work = work.dropna(subset=["mean_plddt", "length"])
    if work.empty:
        return None
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.stripplot(data=work, x="length", y="mean_plddt", ax=ax,
                  size=4, alpha=0.7, color="steelblue", zorder=1)
    sns.pointplot(data=work, x="length", y="mean_plddt", ax=ax,
                  color="darkorange", join=False, ci="sd", markers="D", scale=0.7)
    ax.set_xlabel("Protein length (amino acids)")
    ax.set_ylabel("Mean pLDDT")
    ax.set_ylim(0, 105)
    ax.set_title(f"Mean pLDDT vs. length (n={len(work)})")
    fig.tight_layout()
    p = Path(out_path) if out_path else FIGURES_DIR / "plddt_vs_length.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def plot_runtime_vs_length(df: pd.DataFrame, out_path: Optional[Path] = None) -> Optional[Path]:
    work = df.copy()
    work["total_runtime_seconds"] = pd.to_numeric(work["total_runtime_seconds"], errors="coerce")
    work["length"] = pd.to_numeric(work["length"], errors="coerce")
    work = work.dropna(subset=["total_runtime_seconds", "length"])
    if work.empty:
        return None
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.stripplot(data=work, x="length", y="total_runtime_seconds", ax=ax,
                  size=4, alpha=0.7, color="steelblue", zorder=1)
    sns.pointplot(data=work, x="length", y="total_runtime_seconds", ax=ax,
                  color="purple", join=False, ci="sd", markers="D", scale=0.7)
    ax.set_xlabel("Protein length (amino acids)")
    ax.set_ylabel("Total runtime (seconds / backbone)")
    ax.set_title(f"Runtime vs. length (n={len(work)})")
    fig.tight_layout()
    p = Path(out_path) if out_path else FIGURES_DIR / "runtime_vs_length.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def plot_loss_history(loss_df: pd.DataFrame, out_path: Optional[Path] = None, title: str = "RSO optimization loss") -> Optional[Path]:
    if loss_df is None or loss_df.empty:
        return None
    # ColabDesign af_model.log uses "loss" for the weighted total (older drafts
    # assumed "total"); accept both, preferring total if present.
    total_col = "total" if "total" in loss_df.columns else ("loss" if "loss" in loss_df.columns else None)
    value_cols = ([total_col] if total_col else []) + \
                 [c for c in ["rg", "helix", "con", "plddt", "pae"] if c in loss_df.columns]
    if not value_cols:
        return None
    label_map = {"loss": "total loss", "total": "total loss"}
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for c in value_cols:
        ax.plot(loss_df.index, pd.to_numeric(loss_df[c], errors="coerce"),
                label=label_map.get(c, c), linewidth=1.4)
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Loss value")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    p = Path(out_path) if out_path else FIGURES_DIR / "optimization_loss_100aa.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def plot_candidate_metrics(df: pd.DataFrame, backbone_id: Optional[str] = None,
                           out_path: Optional[Path] = None) -> Optional[Path]:
    work = df.copy()
    if backbone_id is not None:
        work = work[work["backbone_id"] == backbone_id]
    for c in ["rmsd_angstrom", "tm_score", "mean_plddt"]:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna(subset=["rmsd_angstrom", "tm_score", "mean_plddt"])
    if work.empty:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, col, title, color in zip(
        axes,
        ["rmsd_angstrom", "tm_score", "mean_plddt"],
        ["RMSD distribution", "TM-score distribution", "Mean pLDDT distribution"],
        ["steelblue", "forestgreen", "darkorange"],
    ):
        ax.hist(work[col].values, bins=min(12, max(3, len(work)//2)),
                color=color, edgecolor="white")
        ax.set_xlabel(col)
        ax.set_ylabel("count")
        ax.set_title(title)
    b_id = f" backbone {backbone_id}" if backbone_id else ""
    fig.suptitle(f"Candidate metrics{b_id} (n={len(work)})", y=1.02)
    fig.tight_layout()
    p = Path(out_path) if out_path else FIGURES_DIR / "candidate_metrics_100aa.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def generate_all_figures(candidates_df: pd.DataFrame,
                         loss_history_dfs: Optional[dict[str, pd.DataFrame]] = None) -> list[Path]:
    generated: list[Path] = []
    for fn in [plot_rmsd_vs_length, plot_tmscore_vs_length, plot_plddt_vs_length, plot_runtime_vs_length]:
        r = fn(candidates_df)
        if r is not None:
            generated.append(r)
    if loss_history_dfs:
        for bb_id, ldf in loss_history_dfs.items():
            suffix = f"_{bb_id}" if len(loss_history_dfs) > 1 else "_100aa"
            out = FIGURES_DIR / f"optimization_loss{suffix}.png"
            bb_short = bb_id.split("__")[-1]
            r = plot_loss_history(ldf, out_path=out,
                                  title=f"RSO loss history — backbone {bb_short}")
            if r is not None:
                generated.append(r)
    if len(candidates_df):
        r = plot_candidate_metrics(candidates_df)
        if r is not None:
            generated.append(r)
    return generated
