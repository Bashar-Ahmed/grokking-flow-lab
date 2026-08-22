"""Behavior-only training with dense, resumable checkpointing."""

from __future__ import annotations

import csv
import json
import platform
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from grokking_lab.config import LabConfig
from grokking_lab.io import sha256, write_csv, write_json
from grokking_lab.model import (
    ModelConfig,
    Transformer,
    all_logits,
    choose_device,
    fourier_fraction,
    make_dataset,
    metrics,
    seed_everything,
)


def _cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _existing_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    integer_fields = {"epoch"}
    return [
        {key: int(value) if key in integer_fields else float(value) for key, value in row.items()}
        for row in rows
    ]


def _evaluate(
    model: Transformer,
    dataset: Any,
    batch_size: int,
    operation: str,
    p: int,
) -> dict[str, float]:
    model.eval()
    train_tokens, train_labels = dataset.split("train")
    test_tokens, test_labels = dataset.split("test")
    train_values = metrics(model, train_tokens, train_labels, batch_size)
    test_values = metrics(model, test_tokens, test_labels, batch_size)
    logits = all_logits(model, dataset.tokens, batch_size)
    return {
        "train_loss": train_values["loss"],
        "train_accuracy": train_values["accuracy"],
        "test_loss": test_values["loss"],
        "test_accuracy": test_values["accuracy"],
        "fourier_fraction": fourier_fraction(logits, p, operation),
    }


def _run_report(run_id: str, protocol: dict[str, Any], history: list[dict[str, Any]]) -> str:
    final = history[-1]
    grokking = next(
        (
            int(row["epoch"])
            for row in history
            if row["train_accuracy"] >= 0.99 and row["test_accuracy"] >= 0.90
        ),
        None,
    )
    return f"""# Run report: `{run_id}`

## Setup

- Operation: `{protocol["model"]["operation"]}` modulo {protocol["model"]["p"]}
- Seed: {protocol["model"]["seed"]}
- Train fraction: {protocol["model"]["train_fraction"]}
- Weight decay: {protocol["training"]["weight_decay"]}
- Planned epochs: {protocol["training"]["epochs"]}
- Saved behavior checkpoints: {len(history)}

## Behavior

- Last recorded epoch: {int(final["epoch"])}
- Final train accuracy: {final["train_accuracy"]:.6f}
- Final test accuracy: {final["test_accuracy"]:.6f}
- First recorded train>=0.99/test>=0.90 checkpoint: {grokking}

No flow extraction or numbered definition was computed during training.
"""


def train_one(
    run_id: str,
    model_config: ModelConfig,
    weight_decay: float,
    lab_config: LabConfig,
    output_root: Path,
    requested_device: str,
) -> Path:
    run_dir = output_root / run_id
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = run_dir / "protocol.json"
    protocol = {
        "schema_version": 1,
        "run_id": run_id,
        "model": asdict(model_config),
        "training": {
            "epochs": lab_config.epochs,
            "learning_rate": lab_config.learning_rate,
            "weight_decay": weight_decay,
            "betas": [0.9, 0.98],
            "warmup_steps": lab_config.warmup_steps,
            "evaluation_batch_size": lab_config.evaluation_batch_size,
        },
        "checkpoint_epochs": lab_config.checkpoint_epochs,
        "flow_plan": {
            "examples_per_split": lab_config.examples_per_split,
            "kinds": lab_config.flow_kinds,
            "status": "not_run",
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
    }
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text())
        if existing != protocol:
            raise ValueError(f"existing protocol differs for {run_dir}")
    else:
        write_json(protocol_path, protocol)

    device = choose_device(requested_device)
    dataset = make_dataset(model_config, device)
    seed_everything(model_config.seed)
    model = Transformer(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lab_config.learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.98),
    )
    warmup = lab_config.warmup_steps
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(step / warmup, 1.0) if warmup else 1.0
    )

    latest_path = run_dir / "latest.pt"
    start_epoch = 0
    history = _existing_history(run_dir / "metrics.csv")
    if latest_path.exists():
        resume = torch.load(latest_path, map_location=device, weights_only=True)
        model.load_state_dict(resume["model_state"])
        optimizer.load_state_dict(resume["optimizer_state"])
        scheduler.load_state_dict(resume["scheduler_state"])
        start_epoch = int(resume["epoch"])
        if start_epoch >= lab_config.epochs:
            return run_dir

    checkpoint_set = set(lab_config.checkpoint_epochs)

    def evaluate_and_checkpoint(epoch: int, step_loss: float | None) -> None:
        values = _evaluate(
            model,
            dataset,
            lab_config.evaluation_batch_size,
            model_config.operation,
            model_config.p,
        )
        row = {
            "epoch": epoch,
            "step_loss": float("nan") if step_loss is None else step_loss,
            **values,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        checkpoint_path = checkpoint_dir / f"epoch_{epoch:06d}.pt"
        _atomic_torch_save(
            {"epoch": epoch, "model_state": _cpu_state(model), "metrics": row},
            checkpoint_path,
        )
        _atomic_torch_save(
            {
                "epoch": epoch,
                "model_state": _cpu_state(model),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
            },
            latest_path,
        )
        write_csv(run_dir / "metrics.csv", history)
        print(
            f"{run_id} epoch={epoch} train={values['train_accuracy']:.3f} "
            f"test={values['test_accuracy']:.3f}"
        )

    if start_epoch == 0 and not history:
        evaluate_and_checkpoint(0, None)

    last_loss = None
    train_tokens, train_labels = dataset.split("train")
    for epoch in range(start_epoch + 1, lab_config.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_tokens)
        assert isinstance(logits, torch.Tensor)
        loss = F.cross_entropy(logits[:, -1], train_labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        last_loss = float(loss.detach())
        if epoch in checkpoint_set:
            evaluate_and_checkpoint(epoch, last_loss)

    manifest = []
    for checkpoint in sorted(checkpoint_dir.glob("epoch_*.pt")):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        manifest.append(
            {
                "epoch": int(payload["epoch"]),
                "path": str(checkpoint.relative_to(run_dir)),
                "bytes": checkpoint.stat().st_size,
                "sha256": sha256(checkpoint),
            }
        )
    write_json(run_dir / "checkpoint_manifest.json", manifest)
    (run_dir / "REPORT.md").write_text(_run_report(run_id, protocol, history))
    return run_dir


def train_sweep(config: LabConfig, output: Path, device: str) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "plan.json", config.plan())
    runs = [
        train_one(run_id, model, decay, config, output, device)
        for run_id, model, decay in config.runs()
    ]
    rows = []
    for run in runs:
        history = _existing_history(run / "metrics.csv")
        final = history[-1]
        rows.append(
            {
                "run_id": run.name,
                "final_epoch": int(final["epoch"]),
                "train_accuracy": final["train_accuracy"],
                "test_accuracy": final["test_accuracy"],
            }
        )
    write_csv(output / "runs.csv", rows)
    write_behavior_report(output)
    return runs


def write_behavior_report(output: Path) -> Path:
    """Aggregate completed training metrics without touching flow artifacts."""

    rows: list[dict[str, Any]] = []
    for run in sorted(path for path in output.iterdir() if path.is_dir()):
        protocol_path = run / "protocol.json"
        metrics_path = run / "metrics.csv"
        manifest_path = run / "checkpoint_manifest.json"
        if not (protocol_path.exists() and metrics_path.exists() and manifest_path.exists()):
            continue
        protocol = json.loads(protocol_path.read_text())
        history = _existing_history(metrics_path)
        manifest = json.loads(manifest_path.read_text())
        grokking_epoch = next(
            (
                int(row["epoch"])
                for row in history
                if row["train_accuracy"] >= 0.99 and row["test_accuracy"] >= 0.90
            ),
            None,
        )
        first_test_10 = next(
            (int(row["epoch"]) for row in history if row["test_accuracy"] > 0.10),
            None,
        )
        final = history[-1]
        model = protocol["model"]
        rows.append(
            {
                "run_id": run.name,
                "operation": model["operation"],
                "train_fraction": model["train_fraction"],
                "weight_decay": protocol["training"]["weight_decay"],
                "seed": model["seed"],
                "grokking_epoch": grokking_epoch,
                "first_test_gt_10_epoch": first_test_10,
                "final_train_accuracy": final["train_accuracy"],
                "final_test_accuracy": final["test_accuracy"],
                "checkpoints": len(manifest),
            }
        )
    write_csv(output / "behavior_summary.csv", rows)

    grokked = [row for row in rows if row["grokking_epoch"] is not None]
    late = [row for row in grokked if int(row["grokking_epoch"]) > 30_000]
    by_operation = {
        operation: (
            sum(row["grokking_epoch"] is not None for row in rows if row["operation"] == operation),
            sum(row["operation"] == operation for row in rows),
        )
        for operation in ("add", "sub", "mul")
    }
    table = "\n".join(
        "| {run_id} | {grok} | {final:.3f} |".format(
            run_id=row["run_id"],
            grok=row["grokking_epoch"] if row["grokking_epoch"] is not None else "not reached",
            final=row["final_test_accuracy"],
        )
        for row in rows
    )
    operation_text = ", ".join(
        f"{operation} {passed}/{total}" for operation, (passed, total) in by_operation.items()
    )
    late_text = ", ".join(f"`{row['run_id']}` at {row['grokking_epoch']}" for row in late) or "none"
    not_grokked = [row["run_id"] for row in rows if row["grokking_epoch"] is None]
    not_grokked_text = ", ".join(f"`{name}`" for name in not_grokked) or "none"
    report = f"""# Behavior-only training report

## Scope

- Nine task-specific cells: three each for addition, subtraction, and multiplication.
- Seeds: 0, 1, and 2.
- Training horizon: 50,000 full-batch AdamW epochs per run.
- Checkpoint interval: every 100 epochs plus six early checkpoints.
- Completed runs: {len(rows)}/27.
- Saved checkpoints: {sum(int(row["checkpoints"]) for row in rows)}.

No flow extraction or numbered definition was computed for this report.

## Behavioral result

Grokking is the first saved checkpoint with training accuracy at least 0.99 and test
accuracy at least 0.90. Overall, {len(grokked)}/{len(rows)} runs reached this threshold.
By operation: {operation_text}.

Runs crossing the threshold only after the source study's 30,000-epoch budget:
{late_text}.

Runs not reaching the threshold by epoch 50,000: {not_grokked_text}.

The extended horizon therefore distinguishes late grokking from genuine censoring at
50,000 epochs. These are behavioral observations only; they do not establish any flow
or mechanistic conclusion.

## Per-run results

| Run | First grokking checkpoint | Final test accuracy |
|---|---:|---:|
{table}

Machine-readable values are in `behavior_summary.csv`. Per-run metrics, protocols,
checkpoint manifests, and Markdown reports remain in each run directory.
"""
    report_path = output / "REPORT.md"
    report_path.write_text(report)
    return report_path
