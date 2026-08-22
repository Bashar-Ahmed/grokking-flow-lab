from pathlib import Path

from grokking_lab.config import load_config

ROOT = Path(__file__).parents[1]


def test_smoke_matrix_and_checkpoints() -> None:
    config = load_config(ROOT / "configs/smoke.toml")
    assert config.num_runs == 3
    assert config.checkpoint_epochs == (0, 1, 2, 5, 10, 15, 20)
    assert config.plan()["flow_graphs_total"] == 3 * 7 * 4 * 4


def test_scale_template_has_more_seeds_and_resolution() -> None:
    config = load_config(ROOT / "configs/scale_template.toml")
    assert config.num_runs == 27
    assert len(config.checkpoint_epochs) == 507
    assert config.plan()["optimizer_steps"] == 1_350_000


def test_anchor_template_is_fifteen_runs() -> None:
    config = load_config(ROOT / "configs/anchor_template.toml")
    assert config.num_runs == 15
