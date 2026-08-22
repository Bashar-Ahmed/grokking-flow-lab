from __future__ import annotations

import json

import torch

from grokking_lab.config import LabConfig
from grokking_lab.train import train_sweep


def tiny_config() -> LabConfig:
    return LabConfig(
        operations=("add",),
        seeds=(0,),
        train_fractions=(0.3,),
        weight_decays=(1.0,),
        cells=(),
        p=5,
        d_model=8,
        num_heads=2,
        d_mlp=16,
        epochs=1,
        learning_rate=1e-3,
        warmup_steps=0,
        evaluation_batch_size=64,
        checkpoint_every=1,
        checkpoint_include=(0, 1),
        examples_per_split=1,
        flow_kinds=("target_support",),
    )


def test_training_writes_reloadable_checkpoints(tmp_path) -> None:
    [run] = train_sweep(tiny_config(), tmp_path, "cpu")
    manifest = json.loads((run / "checkpoint_manifest.json").read_text())
    assert [row["epoch"] for row in manifest] == [0, 1]
    assert all(len(row["sha256"]) == 64 for row in manifest)
    final = torch.load(run / manifest[-1]["path"], map_location="cpu", weights_only=True)
    assert final["epoch"] == 1
    assert final["model_state"]
    assert (run / "REPORT.md").exists()
    assert (tmp_path / "behavior_summary.csv").exists()
    assert "Behavior-only" in (tmp_path / "REPORT.md").read_text()
