"""TOML configuration and experiment-matrix expansion."""

from __future__ import annotations

import itertools
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from grokking_lab.model import OPERATIONS, ModelConfig, parameter_count


@dataclass(frozen=True)
class LabConfig:
    operations: tuple[str, ...]
    seeds: tuple[int, ...]
    train_fractions: tuple[float, ...]
    weight_decays: tuple[float, ...]
    cells: tuple[tuple[str, float, float], ...]
    p: int
    d_model: int
    num_heads: int
    d_mlp: int
    epochs: int
    learning_rate: float
    warmup_steps: int
    evaluation_batch_size: int
    checkpoint_every: int
    checkpoint_include: tuple[int, ...]
    examples_per_split: int
    flow_kinds: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("seeds cannot be empty")
        if self.cells:
            if any(operation not in OPERATIONS for operation, _, _ in self.cells):
                raise ValueError(f"cell operations must be in {OPERATIONS}")
        elif (
            not self.operations
            or not self.train_fractions
            or not self.weight_decays
            or any(operation not in OPERATIONS for operation in self.operations)
        ):
            raise ValueError("provide explicit cells or complete nonempty matrix axes")
        if self.epochs < 1 or self.learning_rate <= 0 or self.checkpoint_every < 1:
            raise ValueError("epochs, learning_rate, and checkpoint_every must be positive")
        if self.warmup_steps < 0 or self.examples_per_split < 1:
            raise ValueError("warmup_steps must be nonnegative and examples_per_split positive")
        if any(epoch < 0 or epoch > self.epochs for epoch in self.checkpoint_include):
            raise ValueError("explicit checkpoint epoch is outside [0, epochs]")

    @property
    def checkpoint_epochs(self) -> tuple[int, ...]:
        regular = range(0, self.epochs + 1, self.checkpoint_every)
        return tuple(sorted({0, self.epochs, *regular, *self.checkpoint_include}))

    @property
    def num_runs(self) -> int:
        if self.cells:
            return len(self.cells) * len(self.seeds)
        return (
            len(self.operations)
            * len(self.seeds)
            * len(self.train_fractions)
            * len(self.weight_decays)
        )

    def runs(self) -> list[tuple[str, ModelConfig, float]]:
        rows = []
        if self.cells:
            axes = (
                (operation, fraction, decay, seed)
                for operation, fraction, decay in self.cells
                for seed in self.seeds
            )
        else:
            axes = itertools.product(
                self.operations, self.train_fractions, self.weight_decays, self.seeds
            )
        for operation, fraction, weight_decay, seed in axes:
            model = ModelConfig(
                operation=operation,
                p=self.p,
                train_fraction=fraction,
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_mlp=self.d_mlp,
                seed=seed,
            )
            fraction_text = f"{fraction:.3f}".rstrip("0").rstrip(".").replace(".", "p")
            decay_text = f"{weight_decay:g}".replace(".", "p")
            run_id = f"{operation}_frac{fraction_text}_wd{decay_text}_seed{seed}"
            rows.append((run_id, model, weight_decay))
        return rows

    def plan(self) -> dict[str, Any]:
        example = self.runs()[0][1]
        parameters = parameter_count(example)
        checkpoints = len(self.checkpoint_epochs)
        weight_bytes = parameters * 4
        return {
            "runs": self.num_runs,
            "epochs_per_run": self.epochs,
            "optimizer_steps": self.num_runs * self.epochs,
            "checkpoints_per_run": checkpoints,
            "checkpoints_total": self.num_runs * checkpoints,
            "parameters_per_model": parameters,
            "estimated_weight_checkpoints_gib": self.num_runs * checkpoints * weight_bytes / 2**30,
            "estimated_single_resume_checkpoint_mib": parameters * 16 / 2**20,
            "flow_graphs_total": self.num_runs
            * checkpoints
            * self.examples_per_split
            * 2
            * len(self.flow_kinds),
            "note": "Sizes exclude serialization overhead; raw-flow size depends on sparsity.",
        }


def load_config(path: Path) -> LabConfig:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    experiment = raw["experiment"]
    model = raw["model"]
    training = raw["training"]
    checkpointing = raw["checkpointing"]
    flow = raw["flow"]
    cells = tuple(
        (
            str(cell["operation"]),
            float(cell["train_fraction"]),
            float(cell["weight_decay"]),
        )
        for cell in experiment.get("cells", [])
    )
    return LabConfig(
        operations=tuple(experiment.get("operations", [])),
        seeds=tuple(int(value) for value in experiment["seeds"]),
        train_fractions=tuple(float(value) for value in experiment.get("train_fractions", [])),
        weight_decays=tuple(float(value) for value in experiment.get("weight_decays", [])),
        cells=cells,
        p=int(model["p"]),
        d_model=int(model["d_model"]),
        num_heads=int(model["num_heads"]),
        d_mlp=int(model["d_mlp"]),
        epochs=int(training["epochs"]),
        learning_rate=float(training["learning_rate"]),
        warmup_steps=int(training["warmup_steps"]),
        evaluation_batch_size=int(training["evaluation_batch_size"]),
        checkpoint_every=int(checkpointing["every"]),
        checkpoint_include=tuple(int(value) for value in checkpointing.get("include", [])),
        examples_per_split=int(flow["examples_per_split"]),
        flow_kinds=tuple(flow["kinds"]),
    )
