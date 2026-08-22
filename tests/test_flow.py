from __future__ import annotations

import gzip
import json

import pytest
import torch

from grokking_lab.config import LabConfig
from grokking_lab.flow import (
    FLOW_KINDS,
    FLOW_SCALE,
    FLOW_STORAGE_FACTOR,
    DegenerateFlowError,
    NeuronDecomposition,
    RawFlow,
    _build_raw_flow_cached,
    _prepare_neuron_decomposition,
    _record,
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


def test_scaled_record_is_normalized_and_conserved() -> None:
    flow = RawFlow(
        node_labels=("source", "left", "right", "sink"),
        edges=((0, 1, 1 / 3), (0, 2, 2 / 3), (1, 3, 1 / 3), (2, 3, 2 / 3)),
        paths=(((0, 1, 3), 1 / 3), ((0, 2, 3), 2 / 3)),
        kind="target_support",
        target=1,
        competitor=2,
    )
    record = _record(flow, "run", 10, 4, "train", [1, 2])

    assert record["flow_scale"] == FLOW_SCALE
    assert record["storage_subunits_per_unit"] == FLOW_STORAGE_FACTOR
    assert sum(row[1] for row in record["canonical_paths"]) == pytest.approx(FLOW_SCALE)
    assert sum(row[2] for row in record["edges"] if row[0] == 0) == pytest.approx(FLOW_SCALE)
    decoded_paths = [row[1] / FLOW_SCALE for row in record["canonical_paths"]]
    assert decoded_paths == pytest.approx(
        (1 / 3, 2 / 3), abs=1 / (FLOW_SCALE * FLOW_STORAGE_FACTOR)
    )
    incoming = [0.0] * len(flow.node_labels)
    outgoing = [0.0] * len(flow.node_labels)
    for tail, head, units in record["edges"]:
        outgoing[tail] += units
        incoming[head] += units
    assert incoming[1:-1] == pytest.approx(outgoing[1:-1], abs=1e-6)
    assert record["conservation_error_units"] / FLOW_SCALE <= 1e-7
    assert record["split"] == "train"


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
    decomposition = _prepare_neuron_decomposition(model, cache, shared_parts)

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
                    neuron_decomposition=decomposition,
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
                neuron_decomposition=decomposition,
            )
            assert optimized == reference


def test_vectorized_neuron_decomposition_matches_scalar_reference() -> None:
    model_config = ModelConfig("add", 7, 0.3, 8, 2, 16, 2)
    seed_everything(model_config.seed)
    model = Transformer(model_config)
    data = make_dataset(model_config)
    _, cache = model(data.tokens[0][None], return_cache=True)
    attention_parts = attention_input_parts(model, cache)[0]
    vectorized = _prepare_neuron_decomposition(model, cache, attention_parts)

    def probabilities(values: torch.Tensor) -> tuple[list[float], float]:
        positive = values.detach().clamp_min(0)
        total = float(positive.sum())
        if total == 0:
            return [0.0] * len(values), total
        return (positive / total).tolist(), total

    component_probabilities = []
    component_totals = []
    residual_probabilities = []
    residual_totals = []
    head_probabilities = []
    head_totals = []
    for neuron in range(model_config.d_mlp):
        direction = model.W_in[neuron]
        component_values = torch.cat(
            (
                torch.dot(cache.residual_pre[0, -1], direction)[None],
                torch.einsum("hd,d->h", cache.head_output[0, :, -1], direction),
                model.b_in[neuron][None],
            )
        )
        expected, total = probabilities(component_values)
        component_probabilities.append(expected)
        component_totals.append(total)
        assert vectorized.component_probabilities[neuron] == pytest.approx(expected, abs=5e-7)
        assert vectorized.component_totals[neuron] == pytest.approx(total, abs=5e-7)

        residual_values = torch.stack(
            (
                torch.dot(cache.token_embedding[0, -1], direction),
                torch.dot(cache.position_embedding[0, -1], direction),
            )
        )
        expected, total = probabilities(residual_values)
        residual_probabilities.append(expected)
        residual_totals.append(total)
        assert vectorized.residual_input_probabilities[neuron] == pytest.approx(expected, abs=5e-7)
        assert vectorized.residual_input_totals[neuron] == pytest.approx(total, abs=5e-7)

        neuron_head_probabilities = []
        neuron_head_totals = []
        for head in range(model_config.num_heads):
            head_values = torch.einsum("pkd,d->pk", attention_parts[head], direction)
            expected, total = probabilities(head_values.flatten())
            neuron_head_probabilities.append([expected[:2], expected[2:4], expected[4:]])
            neuron_head_totals.append(total)
            actual = [
                value
                for position in vectorized.head_input_probabilities[neuron][head]
                for value in position
            ]
            # Batched and scalar float32 reductions may differ by a few ULPs here;
            # end-to-end output equivalence is gated separately at 1e-7.
            assert actual == pytest.approx(expected, abs=5e-7)
            assert vectorized.head_input_totals[neuron][head] == pytest.approx(total, abs=5e-7)
        head_probabilities.append(neuron_head_probabilities)
        head_totals.append(neuron_head_totals)

    scalar = NeuronDecomposition(
        component_probabilities=component_probabilities,
        component_totals=component_totals,
        residual_input_probabilities=residual_probabilities,
        residual_input_totals=residual_totals,
        head_input_probabilities=head_probabilities,
        head_input_totals=head_totals,
    )
    target = int(data.labels[0])
    logits = model(data.tokens[0][None])
    candidates = logits[0, -1, : model_config.p].clone()
    candidates[target] = -torch.inf
    competitor = int(candidates.argmax())
    catalog = node_catalog(model)
    for kind in FLOW_KINDS:
        try:
            reference = _build_raw_flow_cached(
                model,
                cache,
                target,
                kind,
                competitor,
                attention_parts=attention_parts,
                catalog=catalog,
                neuron_decomposition=scalar,
            )
        except DegenerateFlowError:
            with pytest.raises(DegenerateFlowError):
                _build_raw_flow_cached(
                    model,
                    cache,
                    target,
                    kind,
                    competitor,
                    attention_parts=attention_parts,
                    catalog=catalog,
                    neuron_decomposition=vectorized,
                )
            continue
        optimized = _build_raw_flow_cached(
            model,
            cache,
            target,
            kind,
            competitor,
            attention_parts=attention_parts,
            catalog=catalog,
            neuron_decomposition=vectorized,
        )
        assert [path for path, _ in optimized.paths] == [path for path, _ in reference.paths]
        assert [weight for _, weight in optimized.paths] == pytest.approx(
            [weight for _, weight in reference.paths], abs=1e-7
        )
        assert [(tail, head) for tail, head, _ in optimized.edges] == [
            (tail, head) for tail, head, _ in reference.edges
        ]
        assert [value for _, _, value in optimized.edges] == pytest.approx(
            [value for _, _, value in reference.edges], abs=1e-7
        )


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
    extraction_manifest = json.loads((serial / "manifest.json").read_text())
    assert extraction_manifest["flow_encoding"]["scale"] == FLOW_SCALE
    assert extraction_manifest["flow_encoding"]["decoded_resolution"] <= 1e-7
    assert (serial / "files.json").read_bytes() == (parallel / "files.json").read_bytes()
    for serial_path in sorted(serial.glob("epoch_*.jsonl.gz")):
        parallel_path = parallel / serial_path.name
        assert serial_path.read_bytes() == parallel_path.read_bytes()
        with gzip.open(serial_path, "rt", encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle]
        assert {record["split"] for record in records} == {"train", "test"}
        assert all(record["flow_scale"] == FLOW_SCALE for record in records)
        for record in records:
            if record["status"] == "ok":
                assert sum(row[1] for row in record["canonical_paths"]) == pytest.approx(FLOW_SCALE)

    before = (serial / "files.json").read_bytes()
    extract_run(source, tmp_path / "serial", workers=2, resume=True)
    assert (serial / "files.json").read_bytes() == before
