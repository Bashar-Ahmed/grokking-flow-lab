"""Isolated post-hoc continuation of one completed behavior-only run."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from grokking_lab.io import sha256, write_csv, write_json
from grokking_lab.model import ModelConfig, Transformer, choose_device, make_dataset
from grokking_lab.train import (
    _atomic_torch_save,
    _cpu_state,
    _evaluate,
    _existing_history,
)


def _source_fingerprints(source_run: Path) -> dict[str, str]:
    names = ("protocol.json", "metrics.csv", "checkpoint_manifest.json", "latest.pt")
    return {name: sha256(source_run / name) for name in names}


def _report(
    run_name: str,
    source_epoch: int,
    target_epoch: int,
    history: list[dict[str, Any]],
    manifest_count: int,
) -> str:
    grokking = next(
        (
            int(row["epoch"])
            for row in history
            if row["train_accuracy"] >= 0.99 and row["test_accuracy"] >= 0.90
        ),
        None,
    )
    initial = history[0]
    final = history[-1]
    best = max(history, key=lambda row: row["test_accuracy"])
    result = (
        f"The run first crossed the behavioral grokking threshold at epoch {grokking}."
        if grokking is not None
        else f"The run did not cross the behavioral grokking threshold by epoch {target_epoch}."
    )
    return f"""# POST-HOC continuation report: `{run_name}`

## Evidential status

This is a purely post-hoc extension requested after the main 27-run behavior matrix was
complete and inspected. It must not alter, replace, or be pooled into the main study's
registered outcomes. The source run remains unchanged and is referenced by SHA-256 in
`protocol.json`.

## Scope

- Source checkpoint: epoch {source_epoch}.
- Continuation target: epoch {target_epoch}.
- Saved checkpoints in this isolated artifact: {manifest_count}.
- Flow extraction: not run.

## Result

{result}

- Final training accuracy: {final["train_accuracy"]:.6f}.
- Final test accuracy: {final["test_accuracy"]:.6f}.
- Best test accuracy: {best["test_accuracy"]:.6f} at epoch {int(best["epoch"])}.
- Test accuracy at the source checkpoint: {initial["test_accuracy"]:.6f}.

The threshold is the first saved checkpoint with training accuracy at least 0.99 and
test accuracy at least 0.90. This result is descriptive and post-hoc.
"""


def extend_run(
    source_run: Path,
    output: Path,
    target_epoch: int,
    checkpoint_every: int = 100,
    requested_device: str = "auto",
) -> Path:
    """Resume one completed run into a separate, explicitly post-hoc directory."""

    source_protocol = json.loads((source_run / "protocol.json").read_text())
    source_latest = torch.load(source_run / "latest.pt", map_location="cpu", weights_only=True)
    source_epoch = int(source_latest["epoch"])
    if target_epoch <= source_epoch:
        raise ValueError("target_epoch must exceed the source epoch")
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be positive")
    if source_protocol["flow_plan"]["status"] != "not_run":
        raise ValueError("source run unexpectedly contains flow status")

    fingerprints = _source_fingerprints(source_run)
    model_config = ModelConfig(**source_protocol["model"])
    training = source_protocol["training"]
    protocol = {
        "schema_version": 1,
        "evidential_status": "POST_HOC_DO_NOT_POOL_WITH_MAIN_STUDY",
        "source_run": str(source_run.resolve()),
        "source_epoch": source_epoch,
        "source_fingerprints": fingerprints,
        "model": asdict(model_config),
        "training": {
            **training,
            "source_planned_epochs": training["epochs"],
            "target_epoch": target_epoch,
            "checkpoint_every": checkpoint_every,
        },
        "flow_plan": {"status": "not_run"},
    }
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("existing post-hoc protocol differs from the requested continuation")
    elif any(output.iterdir()):
        raise FileExistsError(f"refusing to mix artifacts in nonempty {output}")
    else:
        write_json(protocol_path, protocol)

    device = choose_device(requested_device)
    dataset = make_dataset(model_config, device)
    model = Transformer(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        betas=tuple(training["betas"]),
    )
    warmup = int(training["warmup_steps"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(step / warmup, 1.0) if warmup else 1.0
    )

    output_latest = output / "latest.pt"
    history_path = output / "metrics.csv"
    if output_latest.exists():
        resume = torch.load(output_latest, map_location=device, weights_only=True)
        history = _existing_history(history_path)
    else:
        resume = torch.load(source_run / "latest.pt", map_location=device, weights_only=True)
        source_history = _existing_history(source_run / "metrics.csv")
        history = [source_history[-1]]
        write_csv(history_path, history)
    model.load_state_dict(resume["model_state"])
    optimizer.load_state_dict(resume["optimizer_state"])
    scheduler.load_state_dict(resume["scheduler_state"])
    start_epoch = int(resume["epoch"])

    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    source_manifest = json.loads((source_run / "checkpoint_manifest.json").read_text())
    source_checkpoint_row = next(
        row for row in source_manifest if int(row["epoch"]) == source_epoch
    )
    initial_checkpoint = checkpoint_dir / f"epoch_{source_epoch:06d}.pt"
    if not initial_checkpoint.exists():
        shutil.copy2(source_run / source_checkpoint_row["path"], initial_checkpoint)

    checkpoint_epochs = set(
        range(source_epoch + checkpoint_every, target_epoch + 1, checkpoint_every)
    )
    checkpoint_epochs.add(target_epoch)
    train_tokens, train_labels = dataset.split("train")

    for epoch in range(start_epoch + 1, target_epoch + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_tokens)
        assert isinstance(logits, torch.Tensor)
        loss = F.cross_entropy(logits[:, -1], train_labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if epoch not in checkpoint_epochs:
            continue
        values = _evaluate(
            model,
            dataset,
            int(training["evaluation_batch_size"]),
            model_config.operation,
            model_config.p,
        )
        row = {
            "epoch": epoch,
            "step_loss": float(loss.detach()),
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
            output_latest,
        )
        write_csv(history_path, history)
        print(
            f"POST_HOC {output.name} epoch={epoch} train={values['train_accuracy']:.3f} "
            f"test={values['test_accuracy']:.3f}"
        )

    manifest = []
    for checkpoint in sorted(checkpoint_dir.glob("epoch_*.pt")):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        manifest.append(
            {
                "epoch": int(payload["epoch"]),
                "path": str(checkpoint.relative_to(output)),
                "bytes": checkpoint.stat().st_size,
                "sha256": sha256(checkpoint),
            }
        )
    write_json(output / "checkpoint_manifest.json", manifest)
    (output / "REPORT.md").write_text(
        _report(output.name, source_epoch, target_epoch, history, len(manifest))
    )
    if _source_fingerprints(source_run) != fingerprints:
        raise AssertionError("the source run changed during post-hoc continuation")
    return output
