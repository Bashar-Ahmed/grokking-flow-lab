"""Raw positive relevance-flow extraction, deliberately separate from training."""

from __future__ import annotations

import gzip
import io
import json
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from grokking_lab.io import sha256, write_json
from grokking_lab.model import (
    ForwardCache,
    ModelConfig,
    Transformer,
    attention_input_parts,
    make_dataset,
)

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

    return _build_raw_flow_cached(model, cache, target, kind, competitor, tolerance)


@torch.inference_mode()
def _build_raw_flow_cached(
    model: Transformer,
    cache: ForwardCache,
    target: int,
    kind: str,
    competitor: int,
    tolerance: float = 1e-12,
    attention_parts: Tensor | None = None,
    catalog: tuple[tuple[str, ...], dict[str, int]] | None = None,
) -> RawFlow:
    """Build a flow from a shared forward cache without changing the decomposition."""

    if kind not in FLOW_KINDS:
        raise ValueError(f"unknown flow kind: {kind}")
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

    labels, index_of = catalog if catalog is not None else node_catalog(model)
    semantic_paths: list[tuple[tuple[str, ...], float]] = []
    source, sink = "source", "sink"
    token_parts = cache.token_embedding[0]
    position_parts = cache.position_embedding[0]
    if attention_parts is None:
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


_WORKER_MODEL: Transformer | None = None
_WORKER_DATASET: Any = None
_WORKER_SELECTED: tuple[tuple[int, str], ...] = ()
_WORKER_KINDS: tuple[str, ...] = ()
_WORKER_CATALOG: tuple[tuple[str, ...], dict[str, int]] | None = None
_WORKER_RUN_DIR: Path | None = None
_WORKER_TARGET_DIR: Path | None = None
_WORKER_RUN_ID = ""
_WORKER_DEVICE = torch.device("cpu")
_WORKER_COMPRESSION_LEVEL = 6


def _initialize_worker(
    model_config: ModelConfig,
    selected: tuple[tuple[int, str], ...],
    kinds: tuple[str, ...],
    run_dir: str,
    target_dir: str,
    device: str,
    compression_level: int,
) -> None:
    """Create immutable per-process context and prevent CPU thread oversubscription."""

    global _WORKER_CATALOG, _WORKER_COMPRESSION_LEVEL, _WORKER_DATASET
    global _WORKER_DEVICE, _WORKER_KINDS, _WORKER_MODEL, _WORKER_RUN_DIR
    global _WORKER_RUN_ID, _WORKER_SELECTED, _WORKER_TARGET_DIR

    torch.set_num_threads(1)
    _WORKER_DEVICE = torch.device(device)
    _WORKER_DATASET = make_dataset(model_config, _WORKER_DEVICE)
    _WORKER_MODEL = Transformer(model_config).to(_WORKER_DEVICE)
    _WORKER_MODEL.eval()
    _WORKER_CATALOG = node_catalog(_WORKER_MODEL)
    _WORKER_SELECTED = selected
    _WORKER_KINDS = kinds
    _WORKER_RUN_DIR = Path(run_dir)
    _WORKER_TARGET_DIR = Path(target_dir)
    _WORKER_RUN_ID = _WORKER_RUN_DIR.name
    _WORKER_COMPRESSION_LEVEL = compression_level


def _write_gzip_records(path: Path, rows: list[dict[str, Any]], compression_level: int) -> None:
    """Atomically write deterministic gzip JSONL."""

    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as raw_handle:
        compressed = gzip.GzipFile(
            filename="",
            mode="wb",
            compresslevel=compression_level,
            fileobj=raw_handle,
            mtime=0,
        )
        with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as text_handle:
            for row in rows:
                text_handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    temporary.replace(path)


def _extract_checkpoint(checkpoint_row: dict[str, Any]) -> dict[str, Any]:
    """Worker entry point: extract one checkpoint into one independent artifact."""

    if (
        _WORKER_MODEL is None
        or _WORKER_CATALOG is None
        or _WORKER_RUN_DIR is None
        or _WORKER_TARGET_DIR is None
    ):
        raise RuntimeError("flow worker was not initialized")
    checkpoint_path = _WORKER_RUN_DIR / checkpoint_row["path"]
    expected_digest = checkpoint_row.get("sha256")
    if expected_digest and sha256(checkpoint_path) != expected_digest:
        raise ValueError(f"checkpoint digest mismatch: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=_WORKER_DEVICE, weights_only=True)
    epoch = int(checkpoint_row["epoch"])
    if int(checkpoint["epoch"]) != epoch:
        raise ValueError(f"checkpoint epoch mismatch: {checkpoint_path}")
    _WORKER_MODEL.load_state_dict(checkpoint["model_state"])

    rows: list[dict[str, Any]] = []
    for example_index, split in _WORKER_SELECTED:
        tokens = _WORKER_DATASET.tokens[example_index]
        target = int(_WORKER_DATASET.labels[example_index])
        logits, cache = _WORKER_MODEL(tokens[None], return_cache=True)
        numeric_logits = logits[0, -1, : _WORKER_MODEL.config.p]
        candidates = numeric_logits.clone()
        candidates[target] = -torch.inf
        competitor = int(candidates.argmax())
        operands = [int(value) for value in tokens[:2].cpu()]
        shared_attention_parts = attention_input_parts(_WORKER_MODEL, cache)[0]
        for kind in _WORKER_KINDS:
            try:
                flow = _build_raw_flow_cached(
                    _WORKER_MODEL,
                    cache,
                    target,
                    kind,
                    competitor,
                    attention_parts=shared_attention_parts,
                    catalog=_WORKER_CATALOG,
                )
                row = _record(flow, _WORKER_RUN_ID, epoch, example_index, split, operands)
            except DegenerateFlowError as error:
                row = {
                    "schema_version": 1,
                    "run_id": _WORKER_RUN_ID,
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
            rows.append(row)

    output_path = _WORKER_TARGET_DIR / f"epoch_{epoch:06d}.jsonl.gz"
    _write_gzip_records(output_path, rows, _WORKER_COMPRESSION_LEVEL)
    return {
        "epoch": epoch,
        "path": output_path.name,
        "records": len(rows),
        "bytes": output_path.stat().st_size,
        "sha256": sha256(output_path),
    }


def extract_run(
    run_dir: Path,
    output_root: Path,
    device: str = "cpu",
    workers: int = 1,
    resume: bool = False,
    compression_level: int = 6,
) -> Path:
    """Write raw records with shared forward caches and checkpoint parallelism."""

    if workers < 1:
        raise ValueError("workers must be positive")
    if not 0 <= compression_level <= 9:
        raise ValueError("compression_level must be between 0 and 9")
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and workers != 1:
        raise ValueError("CUDA extraction supports one worker; use CPU for parallel extraction")

    run_dir = run_dir.resolve()
    target_dir = (output_root / run_dir.name).resolve()
    protocol = json.loads((run_dir / "protocol.json").read_text())
    model_config = ModelConfig(**protocol["model"])
    plan = protocol["flow_plan"]
    kinds = tuple(plan["kinds"])
    if any(kind not in FLOW_KINDS for kind in kinds):
        raise ValueError("protocol contains an unknown flow kind")
    dataset = make_dataset(model_config)
    count = int(plan["examples_per_split"])
    selected = tuple(
        [(int(index), "train") for index in dataset.train_indices[:count]]
        + [(int(index), "test") for index in dataset.test_indices[:count]]
    )
    checkpoint_manifest_path = run_dir / "checkpoint_manifest.json"
    checkpoint_manifest = json.loads(checkpoint_manifest_path.read_text())
    epochs = [int(row["epoch"]) for row in checkpoint_manifest]
    if len(epochs) != len(set(epochs)):
        raise ValueError("checkpoint manifest contains duplicate epochs")
    labels, _ = node_catalog(Transformer(model_config))
    extraction_manifest = {
        "schema_version": 2,
        "raw_record_schema_version": 1,
        "run_id": run_dir.name,
        "source_protocol_sha256": sha256(run_dir / "protocol.json"),
        "source_checkpoint_manifest_sha256": sha256(checkpoint_manifest_path),
        "node_labels": list(labels),
        "selected_examples": [{"index": index, "split": split} for index, split in selected],
        "flow_kinds": list(kinds),
        "records_per_checkpoint": len(selected) * len(kinds),
        "compression": {"format": "gzip", "level": compression_level, "mtime": 0},
        "storage": "one gzip JSONL file per checkpoint; raw edge marginals and canonical paths",
    }

    target_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = target_dir / "manifest.json"
    if any(target_dir.iterdir()):
        if not resume:
            raise FileExistsError(f"refusing to overwrite nonempty {target_dir}")
        existing_manifest = (
            json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        )
        if existing_manifest != extraction_manifest:
            raise ValueError("existing raw-flow manifest differs from the requested extraction")
    else:
        write_json(manifest_path, extraction_manifest)

    expected_names = {epoch: f"epoch_{epoch:06d}.jsonl.gz" for epoch in epochs}
    completed: dict[int, dict[str, Any]] = {}
    files_path = target_dir / "files.json"
    if resume and files_path.exists():
        for row in json.loads(files_path.read_text()):
            epoch = int(row["epoch"])
            path = target_dir / row["path"]
            if (
                expected_names.get(epoch) == row["path"]
                and path.is_file()
                and sha256(path) == row["sha256"]
            ):
                completed[epoch] = row

    pending = [row for row in checkpoint_manifest if int(row["epoch"]) not in completed]
    initargs = (
        model_config,
        selected,
        kinds,
        str(run_dir),
        str(target_dir),
        str(torch_device),
        compression_level,
    )

    def record_completion(row: dict[str, Any]) -> None:
        completed[int(row["epoch"])] = row
        ordered = [completed[epoch] for epoch in sorted(completed)]
        write_json(files_path, ordered)
        print(f"extracted {run_dir.name} epoch={row['epoch']}")

    if workers == 1:
        _initialize_worker(*initargs)
        for checkpoint_row in pending:
            record_completion(_extract_checkpoint(checkpoint_row))
    elif pending:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(workers, len(pending)),
            mp_context=context,
            initializer=_initialize_worker,
            initargs=initargs,
        ) as executor:
            futures = [executor.submit(_extract_checkpoint, row) for row in pending]
            for future in as_completed(futures):
                record_completion(future.result())

    if len(completed) != len(checkpoint_manifest):
        raise AssertionError("raw-flow extraction ended with missing checkpoints")
    return target_dir
