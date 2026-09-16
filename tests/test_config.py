from pathlib import Path
import pytest
from rso_exploration.config import (
    load_config, validate_config, Experiment,
    RSOConfig, MPNNConfig, ValidationConfig, RSOLossConfig,
)
from rso_exploration.paths import CONFIGS_DIR

CONFIG_NAMES = ["smoke.yaml", "reproduction_100aa.yaml",
                "length_experiment_quick.yaml", "length_experiment_full.yaml"]


@pytest.mark.parametrize("name", CONFIG_NAMES)
def test_load_each_config(name):
    p = CONFIGS_DIR / name
    assert p.exists(), f"Missing config {name}"
    cfg = load_config(p)
    assert isinstance(cfg, Experiment)
    assert cfg.experiment_name
    assert cfg.backend in {"colab", "local"}
    assert cfg.rso.iterations > 0
    assert cfg.mpnn.num_seqs > 0


def test_smoke_config_flags_smoke():
    cfg = load_config("smoke.yaml")
    assert cfg.smoke_test is True


def test_reproduction_100aa_defaults():
    cfg = load_config("reproduction_100aa.yaml")
    assert cfg.rso.length == 100
    assert cfg.rso.copies == 1
    assert cfg.rso.iterations == 100
    assert cfg.rso.stage1_iterations + cfg.rso.stage2_iterations == 100
    assert cfg.mpnn.num_seqs == 8
    assert cfg.mpnn.temperature == 0.1
    assert cfg.mpnn.weights == "soluble"
    assert "C" in cfg.mpnn.rm_aa


def test_length_experiment_quick_has_lengths():
    cfg = load_config("length_experiment_quick.yaml")
    lengths = cfg.effective_lengths()
    assert 100 in lengths
    assert len(lengths) >= 3


def test_validate_config_catches_stage_mismatch():
    cfg = Experiment(experiment_name="x", backend="colab", seed=0)
    cfg.rso.length = 50
    cfg.rso.iterations = 100
    cfg.rso.stage1_iterations = 10
    cfg.rso.stage2_iterations = 10
    errors = validate_config(cfg)
    assert any("stage1+stage2" in e for e in errors)


def test_validate_config_catches_no_length():
    cfg = Experiment(experiment_name="x", backend="colab", seed=0)
    errors = validate_config(cfg)
    assert any("length" in e.lower() for e in errors)


def test_effective_seeds_from_experiment_seeds():
    cfg = Experiment(experiment_name="x", backend="colab", seed=0)
    cfg.experiment.seeds = [1, 2, 3]
    assert cfg.effective_seeds() == [1, 2, 3]


def test_effective_lengths_prefers_lengths_over_length():
    cfg = Experiment(experiment_name="x", backend="colab", seed=0)
    cfg.rso.length = 100
    cfg.rso.lengths = [100, 200]
    assert cfg.effective_lengths() == [100, 200]
