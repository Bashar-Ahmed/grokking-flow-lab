from __future__ import annotations

import json

from grokking_lab.config import LabConfig
from grokking_lab.io import sha256
from grokking_lab.posthoc import extend_run
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


def test_posthoc_extension_is_isolated_and_labeled(tmp_path) -> None:
    [source] = train_sweep(tiny_config(), tmp_path / "main", "cpu")
    source_files = ("protocol.json", "metrics.csv", "checkpoint_manifest.json", "latest.pt")
    before = {name: sha256(source / name) for name in source_files}

    output = extend_run(source, tmp_path / "posthoc", 2, checkpoint_every=1, requested_device="cpu")

    assert {name: sha256(source / name) for name in source_files} == before
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["evidential_status"] == "POST_HOC_DO_NOT_POOL_WITH_MAIN_STUDY"
    assert protocol["flow_plan"]["status"] == "not_run"
    manifest = json.loads((output / "checkpoint_manifest.json").read_text())
    assert [row["epoch"] for row in manifest] == [1, 2]
    assert "purely post-hoc" in (output / "REPORT.md").read_text()
