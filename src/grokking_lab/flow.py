"""Raw positive relevance-flow extraction, deliberately separate from training."""

from __future__ import annotations

import gzip
import json
import math
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from grokking_lab.io import sha256, write_json
from grokking_lab.model import ModelConfig, Transformer, attention_input_parts, make_dataset

FLOW_KINDS = (
    "target_support",
    "target_oppose",
    "competitor_support",
    "competitor_oppose",
)


class DegenerateFlowError(ValueError):
    """A requested polarity has no positive local decomposition."""


@dataclass(frozen=True)
class RawFlow:
    node_labels: tuple[str, ...]
    edges: tuple[tuple[int, int, float], ...]
    paths: tuple[tuple[tuple[int, ...], float], ...]
    kind: str
    target: int
    competitor: int

    def conservation_error(self) -> float:
        incoming = [0.0] * len(self.node_labels)
        outgoing = [0.0] * len(self.node_labels)
        for tail, head, value in self.edges:
            outgoing[tail] += value
            incoming[head] += value
        return max(
            (abs(incoming[node] - outgoing[node]) for node in range(1, len(incoming) - 1)),
            default=0.0,
        )


def node_catalog(model: Transformer) -> tuple[tuple[str, ...], dict[str, int]]:
    labels = ["source", "root:direct_residual"]
    labels.extend(f"root:mlp_neuron:{index}" for index in range(model.config.d_mlp))
    labels.append("root:mlp_output_bias")
    labels.extend(("component:residual_pre", "component:mlp_input_bias"))
    labels.extend(f"component:attention_head:{index}" for index in range(model.config.num_heads))
    labels.extend(f"input_position:{index}" for index in range(model.sequence_length))
    for position in range(model.sequence_length):
        labels.extend((f"input:token_embedding:{position}", f"input:position_embedding:{position}"))
    labels.extend(("input:bias", "sink"))
    frozen = tuple(labels)
    return frozen, {label: index for index, label in enumerate(frozen)}


def _distribution(values: Tensor, tolerance: float) -> Tensor:
    positive = values.clamp_min(0)
    total = positive.sum()
    if float(total) <= tolerance:
        raise DegenerateFlowError("positive local decomposition is empty")
    return positive / total


@torch.inference_mode()
def build_raw_flow(
    model: Transformer,
    tokens: Tensor,
    target: int,
    kind: str,
    competitor: int | None = None,
    tolerance: float = 1e-12,
) -> RawFlow:
    """Construct a unit flow while holding attention probabilities fixed."""

    if kind not in FLOW_KINDS:
        raise ValueError(f"unknown flow kind: {kind}")
    if tokens.shape != (model.sequence_length,):
        raise ValueError("tokens must describe exactly one three-token example")

    logits, cache = model(tokens[None], return_cache=True)
    assert isinstance(logits, Tensor)
    numeric_logits = logits[0, -1, : model.config.p]
    if competitor is None:
        candidates = numeric_logits.clone()
        candidates[target] = -torch.inf
        competitor = int(candidates.argmax())
    if competitor == target:
        raise ValueError("competitor must differ from target")

    objective = target if kind.startswith("target") else competitor
    polarity = 1 if kind.endswith("support") else -1
    direction_out = model.W_U[:, objective]
    residual_mid = cache.residual_mid[0, -1]
    mlp_post = cache.mlp_post[0, -1]
    neuron_coefficients = torch.einsum("dm,d->m", model.W_out, direction_out)
    root_contributions = torch.cat(
        (
            torch.dot(residual_mid, direction_out)[None],
            mlp_post * neuron_coefficients,
            torch.dot(model.b_out, direction_out)[None],
        )
    )
    try:
        root_distribution = _distribution(polarity * root_contributions, tolerance)
    except DegenerateFlowError as error:
        raise DegenerateFlowError(f"{kind} has no positive root relevance") from error

    labels, index_of = node_catalog(model)
    semantic_paths: list[tuple[tuple[str, ...], float]] = []
    source, sink = "source", "sink"
    token_parts = cache.token_embedding[0]
    position_parts = cache.position_embedding[0]
    attention_parts = attention_input_parts(model, cache)[0]
    head_outputs = cache.head_output[0, :, -1]

    def add_input(
        prefix: tuple[str, ...], mass: float, contributions: Tensor, position: int
    ) -> None:
        probabilities = _distribution(contributions, tolerance)
        for input_kind, probability in enumerate(probabilities):
            weight = mass * float(probability)
            if weight <= tolerance:
                continue
            name = "token_embedding" if input_kind == 0 else "position_embedding"
            semantic_paths.append(
                ((*prefix, f"input_position:{position}", f"input:{name}:{position}", sink), weight)
            )

    def add_head(prefix: tuple[str, ...], mass: float, head: int, direction: Tensor) -> None:
        contributions = torch.einsum("pkd,d->pk", attention_parts[head], direction)
        probabilities = _distribution(contributions.flatten(), tolerance).reshape(3, 2)
        for position in range(3):
            for input_kind in range(2):
                weight = mass * float(probabilities[position, input_kind])
                if weight <= tolerance:
                    continue
                name = "token_embedding" if input_kind == 0 else "position_embedding"
                semantic_paths.append(
                    (
                        (
                            *prefix,
                            f"component:attention_head:{head}",
                            f"input_position:{position}",
                            f"input:{name}:{position}",
                            sink,
                        ),
                        weight,
                    )
                )

    direct_mass = float(root_distribution[0])
    if direct_mass > tolerance:
        direction = polarity * direction_out
        components = torch.cat((cache.residual_pre[0, -1][None], head_outputs))
        probabilities = _distribution(torch.einsum("cd,d->c", components, direction), tolerance)
        residual_mass = direct_mass * float(probabilities[0])
        if residual_mass > tolerance:
            add_input(
                (source, "root:direct_residual", "component:residual_pre"),
                residual_mass,
                torch.stack(
                    (
                        torch.dot(token_parts[-1], direction),
                        torch.dot(position_parts[-1], direction),
                    )
                ),
                2,
            )
        for head in range(model.config.num_heads):
            mass = direct_mass * float(probabilities[head + 1])
            if mass > tolerance:
                add_head((source, "root:direct_residual"), mass, head, direction)

    for neuron in range(model.config.d_mlp):
        root_mass = float(root_distribution[neuron + 1])
        if root_mass <= tolerance:
            continue
        root = f"root:mlp_neuron:{neuron}"
        direction = model.W_in[neuron]
        contributions = torch.cat(
            (
                torch.dot(cache.residual_pre[0, -1], direction)[None],
                torch.einsum("hd,d->h", head_outputs, direction),
                model.b_in[neuron][None],
            )
        )
        probabilities = _distribution(contributions, tolerance)
        residual_mass = root_mass * float(probabilities[0])
        if residual_mass > tolerance:
            add_input(
                (source, root, "component:residual_pre"),
                residual_mass,
                torch.stack(
                    (
                        torch.dot(token_parts[-1], direction),
                        torch.dot(position_parts[-1], direction),
                    )
                ),
                2,
            )
        for head in range(model.config.num_heads):
            mass = root_mass * float(probabilities[head + 1])
            if mass > tolerance:
                add_head((source, root), mass, head, direction)
        bias_mass = root_mass * float(probabilities[-1])
        if bias_mass > tolerance:
            semantic_paths.append(
                ((source, root, "component:mlp_input_bias", "input:bias", sink), bias_mass)
            )

    output_bias_mass = float(root_distribution[-1])
    if output_bias_mass > tolerance:
        semantic_paths.append(
            ((source, "root:mlp_output_bias", "input:bias", sink), output_bias_mass)
        )

    total = sum(weight for _, weight in semantic_paths)
    if not math.isclose(total, 1.0, rel_tol=1e-5, abs_tol=1e-8):
        raise AssertionError(f"canonical paths sum to {total}, expected one")
    indexed_paths = tuple(
        (tuple(index_of[label] for label in path), weight / total)
        for path, weight in semantic_paths
    )
    edge_values: dict[tuple[int, int], float] = {}
    for path, weight in indexed_paths:
        for edge in pairwise(path):
            edge_values[edge] = edge_values.get(edge, 0.0) + weight
    edges = tuple((tail, head, value) for (tail, head), value in sorted(edge_values.items()))
    result = RawFlow(labels, edges, indexed_paths, kind, target, competitor)
    if result.conservation_error() > 1e-7:
        raise AssertionError(f"flow is not conserved: {result.conservation_error()}")
    return result


def _record(
    flow: RawFlow,
    run_id: str,
    epoch: int,
    example_index: int,
    split: str,
    operands: list[int],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "epoch": epoch,
        "example_index": example_index,
        "split": split,
        "operands": operands,
        "target": flow.target,
        "competitor": flow.competitor,
        "flow_kind": flow.kind,
        "status": "ok",
        "conservation_error": flow.conservation_error(),
        "edges": [[tail, head, value] for tail, head, value in flow.edges],
        "canonical_paths": [[list(path), weight] for path, weight in flow.paths],
    }


def extract_run(run_dir: Path, output_root: Path, device: str = "cpu") -> Path:
    """Write one compressed raw record per example, checkpoint, and flow kind."""

    protocol = json.loads((run_dir / "protocol.json").read_text())
    model_config = ModelConfig(**protocol["model"])
    plan = protocol["flow_plan"]
    kinds = tuple(plan["kinds"])
    if any(kind not in FLOW_KINDS for kind in kinds):
        raise ValueError("protocol contains an unknown flow kind")
    target_dir = output_root / run_dir.name
    if target_dir.exists() and any(target_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty {target_dir}")
    target_dir.mkdir(parents=True)

    torch_device = torch.device(device)
    dataset = make_dataset(model_config, torch_device)
    count = int(plan["examples_per_split"])
    selected = [(int(index), "train") for index in dataset.train_indices[:count].cpu()]
    selected += [(int(index), "test") for index in dataset.test_indices[:count].cpu()]
    checkpoint_manifest = json.loads((run_dir / "checkpoint_manifest.json").read_text())
    empty_model = Transformer(model_config)
    labels, _ = node_catalog(empty_model)
    write_json(
        target_dir / "manifest.json",
        {
            "schema_version": 1,
            "run_id": run_dir.name,
            "source_protocol_sha256": sha256(run_dir / "protocol.json"),
            "node_labels": labels,
            "selected_examples": [{"index": index, "split": split} for index, split in selected],
            "flow_kinds": kinds,
            "storage": "one gzip JSONL file per checkpoint; raw edge marginals and canonical paths",
        },
    )

    files = []
    for checkpoint_row in checkpoint_manifest:
        epoch = int(checkpoint_row["epoch"])
        checkpoint = torch.load(
            run_dir / checkpoint_row["path"], map_location=torch_device, weights_only=True
        )
        model = Transformer(model_config).to(torch_device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        output_path = target_dir / f"epoch_{epoch:06d}.jsonl.gz"
        with gzip.open(output_path, "wt", encoding="utf-8") as handle:
            for example_index, split in selected:
                tokens = dataset.tokens[example_index]
                target = int(dataset.labels[example_index])
                logits = model(tokens[None])
                assert isinstance(logits, Tensor)
                candidates = logits[0, -1, : model_config.p].clone()
                candidates[target] = -torch.inf
                competitor = int(candidates.argmax())
                operands = [int(value) for value in tokens[:2].cpu()]
                for kind in kinds:
                    try:
                        flow = build_raw_flow(model, tokens, target, kind, competitor)
                        row = _record(flow, run_dir.name, epoch, example_index, split, operands)
                    except DegenerateFlowError as error:
                        row = {
                            "schema_version": 1,
                            "run_id": run_dir.name,
                            "epoch": epoch,
                            "example_index": example_index,
                            "split": split,
                            "operands": operands,
                            "target": target,
                            "competitor": competitor,
                            "flow_kind": kind,
                            "status": "degenerate",
                            "error": str(error),
                        }
                    handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        files.append(
            {
                "epoch": epoch,
                "path": output_path.name,
                "bytes": output_path.stat().st_size,
                "sha256": sha256(output_path),
            }
        )
        print(f"extracted {run_dir.name} epoch={epoch}")
    write_json(target_dir / "files.json", files)
    return target_dir
