from __future__ import annotations
from pathlib import Path
from typing import Any, Optional
import yaml
from dataclasses import dataclass, field, asdict
from .paths import CONFIGS_DIR


@dataclass
class RSOLossConfig:
    rg_weight: float = 0.1
    rg_weight_large_over_600: Optional[float] = None
    helix_weight: float = -0.2
    con_weight: float = 1.0
    plddt_weight: float = 0.5
    pae_weight: float = 0.5


@dataclass
class RSOConfig:
    length: Optional[int] = None
    lengths: Optional[list[int]] = None
    copies: int = 1
    iterations: int = 100
    stage1_iterations: int = 90
    stage2_iterations: int = 10
    protocol: str = "hallucination"
    mode: list[str] = field(default_factory=lambda: ["gumbel", "soft"])
    rm_aa: str = "C"
    loss: RSOLossConfig = field(default_factory=RSOLossConfig)


@dataclass
class MPNNConfig:
    num_seqs: int = 8
    temperature: float = 0.1
    weights: str = "soluble"
    rm_aa: str = "C"


@dataclass
class ValidationConfig:
    model: str = "alphafold_single"
    alphafold_model: str = "model_4_ptm"
    num_recycles: int = 3
    use_initial_guess: bool = False
    use_aa_initialization: bool = False


@dataclass
class OutputConfig:
    base_dir: str = "results/runs"
    overwrite: bool = False
    resume: bool = True


@dataclass
class ExperimentConfig:
    backbones_per_length: int = 1
    seeds: Optional[list[int]] = None
    seeds_override: Optional[list[int]] = None


@dataclass
class Experiment:
    experiment_name: str
    backend: str
    seed: int
    notes: str = ""
    rso: RSOConfig = field(default_factory=RSOConfig)
    mpnn: MPNNConfig = field(default_factory=MPNNConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    smoke_test: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def effective_seeds(self) -> list[int]:
        if self.experiment.seeds_override:
            return list(self.experiment.seeds_override)
        if self.experiment.seeds:
            return list(self.experiment.seeds)
        return [self.seed + i for i in range(max(1, self.experiment.backbones_per_length))]

    def effective_lengths(self) -> list[int]:
        if self.rso.lengths:
            return list(self.rso.lengths)
        if self.rso.length:
            return [self.rso.length]
        raise ValueError("Neither rso.length nor rso.lengths is set")


def load_config(path: str | Path) -> Experiment:
    p = Path(path)
    # Resolve: use the path as given if it exists; otherwise, if it is a bare
    # filename (e.g. "smoke.yaml"), look it up under CONFIGS_DIR. This accepts
    # both "configs/smoke.yaml" (preferred from project root) and "smoke.yaml".
    if p.exists():
        resolved = p
    elif not p.is_absolute() and (CONFIGS_DIR / p.name).exists():
        resolved = CONFIGS_DIR / p.name
    elif not p.is_absolute():
        resolved = CONFIGS_DIR / p
    else:
        resolved = p
    with open(resolved) as f:
        raw = yaml.safe_load(f)
    rso_raw = {k: v for k, v in raw.get("rso", {}).items() if k != "loss"}
    loss_raw = raw.get("rso", {}).get("loss", {})
    return Experiment(
        experiment_name=raw.get("experiment_name", "unnamed"),
        backend=raw.get("backend", "colab"),
        seed=raw.get("seed", 0),
        notes=raw.get("notes", ""),
        rso=RSOConfig(**rso_raw, loss=RSOLossConfig(**loss_raw)),
        mpnn=MPNNConfig(**raw.get("mpnn", {})),
        validation=ValidationConfig(**raw.get("validation", {})),
        output=OutputConfig(**raw.get("output", {})),
        experiment=ExperimentConfig(**raw.get("experiment", {})),
        smoke_test=bool(raw.get("smoke_test", False)),
    )


def validate_config(cfg: Experiment) -> list[str]:
    errors: list[str] = []
    if cfg.rso.length is None and cfg.rso.lengths is None:
        errors.append("rso.length or rso.lengths must be set")
    if cfg.mpnn.num_seqs < 1:
        errors.append("mpnn.num_seqs must be >= 1")
    if cfg.mpnn.temperature < 0:
        errors.append("mpnn.temperature must be >= 0")
    if cfg.validation.num_recycles < 0:
        errors.append("validation.num_recycles must be >= 0")
    if cfg.rso.iterations < 1:
        errors.append("rso.iterations must be >= 1")
    s1 = cfg.rso.stage1_iterations
    s2 = cfg.rso.stage2_iterations
    if s1 + s2 != cfg.rso.iterations:
        errors.append(f"stage1+stage2 ({s1}+{s2}) != iterations ({cfg.rso.iterations})")
    return errors
