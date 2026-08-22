from __future__ import annotations

import gzip
import json

import pytest
import torch

from grokking_lab.config import LabConfig
from grokking_lab.flow import (
    FLOW_KINDS,
    DegenerateFlowError,
    _build_raw_flow_cached,
    build_raw_flow,
    extract_run,
    node_catalog,
)
from grokking_lab.model import (
    ModelConfig,
    Transformer,
    attention_input_parts,
    make_dataset,
    seed_everything,
)
from grokking_lab.train import train_sweep


@pytest.mark.parametrize("kind", FLOW_KINDS)
def test_raw_flow_is_unit_normalized_and_conserved(kind: str) -> None:
    model_config = ModelConfig("add", 7, 0.3, 8, 2, 16, 2)
    seed_everything(model_config.seed)
    model = Transformer(model_config)
    data = make_dataset(model_config)
    try:
        flow = build_raw_flow(model, data.tokens[0], int(data.labels[0]), kind)
    except DegenerateFlowError:
        pytest.skip("this random initialization has no positive mass for the requested polarity")
    assert sum(weight for _, weight in flow.paths) == pytest.approx(1.0, abs=1e-7)
    assert flow.conservation_error() < 1e-7
    assert all(value >= 0 for _, _, value in flow.edges)


def test_shared_forward_cache_is_exactly_equivalent() -> None:
    model_config = ModelConfig("add", 7, 0.3, 8, 2, 16, 2)
    seed_everything(model_config.seed)
    model = Transformer(model_config)
    data = make_dataset(model_config)
    tokens = data.tokens[0]
    target = int(data.labels[0])
    logits, cache = model(tokens[None], return_cache=True)
    candidates = logits[0, -1, : model_config.p].clone()
    candidates[target] = -torch.inf
    competitor = int(candidates.argmax())
    shared_parts = attention_input_parts(model, cache)[0]
    catalog = node_catalog(model)

    for kind in FLOW_KINDS:
        try:
            reference = build_raw_flow(model, tokens, target, kind, competitor)
        except DegenerateFlowError:
            with pytest.raises(DegenerateFlowError):
                _build_raw_flow_cached(
                    model,
                    cache,
                    target,
                    kind,
                    competitor,
                    attention_parts=shared_parts,
                    catalog=catalog,
                )
        else:
            optimized = _build_raw_flow_cached(
                model,
                cache,
                target,
                kind,
                competitor,
                attention_parts=shared_parts,
                catalog=catalog,
            )
            assert optimized == reference


def test_parallel_extraction_matches_serial_and_labels_split(tmp_path) -> None:
    config = LabConfig(
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
        flow_kinds=FLOW_KINDS,
    )
    [source] = train_sweep(config, tmp_path / "source", "cpu")
    serial = extract_run(source, tmp_path / "serial", workers=1)
    parallel = extract_run(source, tmp_path / "parallel", workers=2)

    assert (serial / "manifest.json").read_bytes() == (parallel / "manifest.json").read_bytes()
    assert (serial / "files.json").read_bytes() == (parallel / "files.json").read_bytes()
    for serial_path in sorted(serial.glob("epoch_*.jsonl.gz")):
        parallel_path = parallel / serial_path.name
        assert serial_path.read_bytes() == parallel_path.read_bytes()
        with gzip.open(serial_path, "rt", encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle]
        assert {record["split"] for record in records} == {"train", "test"}

    before = (serial / "files.json").read_bytes()
    extract_run(source, tmp_path / "serial", workers=2, resume=True)
    assert (serial / "files.json").read_bytes() == before
