#!/usr/bin/env python3
"""Compute and plot train/test definition trajectories aligned to grokking."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from grokking_lab.definitions import compute_definitions, graph_from_record
from grokking_lab.io import sha256, write_json

DEFINITIONS = tuple(f"definition_{index:02d}" for index in range(1, 6))
COLORS = {"add": "#2878B5", "sub": "#D95F0E", "mul": "#2A9D6F"}
TITLES = {"add": "Addition", "sub": "Subtraction", "mul": "Multiplication"}


def _write_gzip(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as raw_handle:
        compressed = gzip.GzipFile(
            filename="", mode="wb", compresslevel=6, fileobj=raw_handle, mtime=0
        )
        with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    temporary.replace(path)


def _compute_checkpoint(task: dict[str, Any]) -> dict[str, Any]:
    source = Path(task["source"])
    output = Path(task["output"])
    rows: list[dict[str, Any]] = []
    ok = 0
    degenerate = 0
    with gzip.open(source, "rt", encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            identity = {
                "schema_version": 1,
                "run_id": task["run_id"],
                "operation": task["operation"],
                "seed": task["seed"],
                "epoch": task["epoch"],
                "relative_epoch": task["relative_epoch"],
                "example_index": raw["example_index"],
                "split": raw["split"],
                "flow_kind": raw["flow_kind"],
                "status": raw["status"],
            }
            if raw["status"] == "ok":
                identity.update(compute_definitions(graph_from_record(raw, task["num_nodes"])))
                ok += 1
            else:
                identity["error"] = raw.get("error", "degenerate raw flow")
                degenerate += 1
            rows.append(identity)
    _write_gzip(output, rows)
    return {
        "run_id": task["run_id"],
        "epoch": task["epoch"],
        "relative_epoch": task["relative_epoch"],
        "path": task["relative_output"],
        "records": len(rows),
        "ok": ok,
        "degenerate": degenerate,
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
    }


def _tasks(
    raw_root: Path, behavior_summary: Path, output: Path, window: int
) -> tuple[list[dict[str, Any]], list[str]]:
    tasks: list[dict[str, Any]] = []
    excluded: list[str] = []
    with behavior_summary.open(newline="") as handle:
        behavior = list(csv.DictReader(handle))
    for row in behavior:
        run_id = row["run_id"]
        if not row["grokking_epoch"]:
            excluded.append(run_id)
            continue
        grokking = int(row["grokking_epoch"])
        run_dir = raw_root / run_id
        manifest = json.loads((run_dir / "manifest.json").read_text())
        for file_row in json.loads((run_dir / "files.json").read_text()):
            epoch = int(file_row["epoch"])
            relative_epoch = epoch - grokking
            if not -window <= relative_epoch <= window:
                continue
            relative_output = f"per-record/{run_id}/epoch_{epoch:06d}.jsonl.gz"
            tasks.append(
                {
                    "run_id": run_id,
                    "operation": row["operation"],
                    "seed": int(row["seed"]),
                    "epoch": epoch,
                    "relative_epoch": relative_epoch,
                    "num_nodes": len(manifest["node_labels"]),
                    "source": str((run_dir / file_row["path"]).resolve()),
                    "source_sha256": file_row["sha256"],
                    "output": str((output / relative_output).resolve()),
                    "relative_output": relative_output,
                }
            )
    return sorted(tasks, key=lambda row: (row["run_id"], row["epoch"])), excluded


def _aggregate(output: Path, files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    for file_row in files:
        with gzip.open(output / file_row["path"], "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["status"] != "ok":
                    continue
                key = (
                    row["run_id"],
                    row["operation"],
                    int(row["seed"]),
                    int(row["epoch"]),
                    int(row["relative_epoch"]),
                    row["split"],
                )
                group = groups.setdefault(
                    key,
                    {"n": 0, **{definition: 0.0 for definition in DEFINITIONS}},
                )
                group["n"] += 1
                for definition in DEFINITIONS:
                    group[definition] += float(row[definition])
    rows = []
    for key, values in sorted(groups.items()):
        run_id, operation, seed, epoch, relative_epoch, split = key
        count = int(values["n"])
        rows.append(
            {
                "run_id": run_id,
                "operation": operation,
                "seed": seed,
                "epoch": epoch,
                "relative_epoch": relative_epoch,
                "split": split,
                "n": count,
                **{definition: float(values[definition]) / count for definition in DEFINITIONS},
            }
        )
    path = output / "aligned_trajectories.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _lighter(color: str, fraction: float = 0.38) -> tuple[float, float, float]:
    from matplotlib.colors import to_rgb

    rgb = np.asarray(to_rgb(color))
    return tuple(rgb + (1 - rgb) * fraction)


def _plot(rows: list[dict[str, Any]], output: Path) -> list[str]:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "legend.frameon": False,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
        }
    )
    stems = []
    for definition in DEFINITIONS:
        fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.55), sharex=True, sharey=True)
        for axis, operation in zip(axes, ("add", "sub", "mul"), strict=True):
            operation_rows = [row for row in rows if row["operation"] == operation]
            for split in ("train", "test"):
                color = COLORS[operation] if split == "train" else _lighter(COLORS[operation])
                linestyle = "-" if split == "train" else "--"
                by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for row in operation_rows:
                    if row["split"] == split:
                        by_run[str(row["run_id"])].append(row)
                for run_rows in by_run.values():
                    ordered = sorted(run_rows, key=lambda row: int(row["relative_epoch"]))
                    axis.plot(
                        [int(row["relative_epoch"]) / 1000 for row in ordered],
                        [float(row[definition]) for row in ordered],
                        color=color,
                        linestyle=linestyle,
                        alpha=0.18,
                        linewidth=0.7,
                    )

                by_epoch: dict[int, list[float]] = defaultdict(list)
                for row in operation_rows:
                    if row["split"] == split:
                        by_epoch[int(row["relative_epoch"])].append(float(row[definition]))
                x_values = np.asarray(sorted(by_epoch), dtype=float) / 1000
                means = np.asarray([np.mean(by_epoch[epoch]) for epoch in sorted(by_epoch)])
                ci95 = np.asarray(
                    [
                        0.0
                        if len(by_epoch[epoch]) < 2
                        else 1.96
                        * np.std(by_epoch[epoch], ddof=1)
                        / math.sqrt(len(by_epoch[epoch]))
                        for epoch in sorted(by_epoch)
                    ]
                )
                axis.fill_between(
                    x_values,
                    means - ci95,
                    means + ci95,
                    color=color,
                    alpha=0.14 if split == "train" else 0.10,
                    linewidth=0,
                )
                axis.plot(
                    x_values,
                    means,
                    color=color,
                    linestyle=linestyle,
                    linewidth=2.25,
                )
            axis.axvline(0, color="#20242A", linestyle="--", linewidth=1)
            axis.set_title(TITLES[operation])
            axis.set_xlabel("Epoch relative to grokking, G (thousands)")
            axis.set_xlim(-5, 5)
            axis.grid(axis="y", color="#E3E6EA", linewidth=0.6)
        axes[0].set_ylim(bottom=0)
        axes[0].set_ylabel(f"{definition.replace('_', '-').title()} value")
        legend = [
            Line2D([0], [0], color="#333333", linewidth=2.2, linestyle="-", label="Train"),
            Line2D([0], [0], color="#777777", linewidth=2.2, linestyle="--", label="Test"),
        ]
        axes[-1].legend(handles=legend, loc="lower right", fontsize=8)
        axes[0].text(
            0.03,
            0.04,
            "Thin: runs\nThick: mean\nBand: 95% mean CI",
            transform=axes[0].transAxes,
            fontsize=7.5,
            color="#555B65",
        )
        label = definition.replace("_", "-").title()
        fig.suptitle(
            f"{label} around behavioral grokking, separated by example split",
            y=1.03,
            fontweight="bold",
        )
        fig.tight_layout()
        stem = f"{definition}_aligned_train_test"
        fig.savefig(output / f"{stem}.png", bbox_inches="tight", facecolor="white")
        fig.savefig(output / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        stems.append(stem)
    return stems


def _endpoint_summary(
    rows: list[dict[str, Any]], output: Path, window: int
) -> list[dict[str, Any]]:
    result = []
    for definition in DEFINITIONS:
        for operation in ("add", "sub", "mul"):
            for split in ("train", "test"):
                row: dict[str, Any] = {
                    "definition": definition,
                    "operation": operation,
                    "split": split,
                }
                for label, relative_epoch in (("pre", -window), ("g", 0), ("post", window)):
                    values = [
                        float(candidate[definition])
                        for candidate in rows
                        if candidate["operation"] == operation
                        and candidate["split"] == split
                        and int(candidate["relative_epoch"]) == relative_epoch
                    ]
                    row[f"{label}_relative_epoch"] = relative_epoch
                    row[f"{label}_n_runs"] = len(values)
                    row[f"{label}_mean"] = float(np.mean(values)) if values else math.nan
                result.append(row)
    path = output / "endpoint_summary.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    return result


def _report(
    output: Path,
    protocol: dict[str, Any],
    files: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    plot_rows: list[dict[str, Any]],
    endpoints: list[dict[str, Any]],
    stems: list[str],
    elapsed: float,
) -> None:
    ok = sum(int(row["ok"]) for row in files)
    degenerate = sum(int(row["degenerate"]) for row in files)
    figures = "\n".join(
        f"- [{stem}.png]({stem}.png) and [{stem}.pdf]({stem}.pdf)" for stem in stems
    )
    excluded = ", ".join(f"`{run_id}`" for run_id in protocol["excluded_runs"])
    endpoint_of = {(row["definition"], row["operation"], row["split"]): row for row in endpoints}
    consistently_increasing = []
    direction_agreements = 0
    direction_comparisons = 0
    for definition in DEFINITIONS:
        all_increase = True
        for operation in ("add", "sub", "mul"):
            directions = []
            for split in ("train", "test"):
                endpoint = endpoint_of[(definition, operation, split)]
                delta = float(endpoint["post_mean"]) - float(endpoint["pre_mean"])
                directions.append(math.copysign(1, delta) if delta else 0)
            direction_comparisons += 1
            direction_agreements += int(directions[0] == directions[1])
            all_increase &= directions == [1, 1]
        if all_increase:
            consistently_increasing.append(definition.replace("_", "-").title())
    increasing_text = ", ".join(consistently_increasing) or "none"
    (output / "REPORT.md").write_text(
        f"""# Aligned numbered-definition trajectories

## Scope

This descriptive analysis uses the frozen main-study raw flows only. Runs are aligned
to the first saved checkpoint with training accuracy at least 0.99 and test accuracy
at least 0.90. The window is ±{protocol["window_epochs"]:,} epochs. It includes
{protocol["included_runs"]} grokked runs and excludes {excluded}, which did not grok
within the main 50,000-epoch horizon. Post-hoc checkpoints are not used.

## Train/test separation

Every run/checkpoint is averaged separately over the valid raw graphs selected from
training examples and test examples. Solid/darker lines represent train examples;
dashed/lighter lines represent test examples. Thin lines are individual runs, thick
lines are operation/split means, and bands are descriptive 95% mean confidence
intervals across available runs at each relative epoch. To remove initialization
transients from the aligned curves, plotted and summarized checkpoints have source
epoch strictly greater than {protocol["source_epoch_exclusive_min"]:,}.

## Scale and outputs

- Raw checkpoint files analyzed: {len(files):,}.
- Valid per-graph definition rows: {ok:,}.
- Explicit degenerate rows excluded from numeric means: {degenerate:,}.
- Aggregated run/epoch/split rows: {len(rows):,}.
- Plotted run/epoch/split rows after the source-epoch filter: {len(plot_rows):,}.
- Definitions: Definition-01 through Definition-05.
- Computation and plotting runtime: {elapsed / 60:.1f} minutes.

{figures}

## Descriptive findings

- {increasing_text} is the only candidate whose available-run mean increases from
  −{protocol["window_epochs"] // 1000}k to +{protocol["window_epochs"] // 1000}k for both
  train and test examples in all three operations.
- Train and test agree on the direction of that pre-to-post change in
  {direction_agreements}/{direction_comparisons} definition/operation comparisons.
- The other candidates are task-dependent over this window rather than showing a
  shared increasing trajectory across addition, subtraction, and multiplication.

`endpoint_summary.csv` contains the train/test means and available-run counts at
−{protocol["window_epochs"] // 1000}k, G, and +{protocol["window_epochs"] // 1000}k. Far-left
counts are smaller for fast-grokking runs whose histories do not extend the full
window before G. Confidence intervals are descriptive because runs share cell structure.

Per-graph values are stored under `per-record/`; `aligned_trajectories.csv` contains
the split-specific run/checkpoint means. This analysis does not assign semantic names
to any candidate definition.
"""
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--behavior-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--window", type=int, default=5000)
    parser.add_argument("--min-source-epoch-exclusive", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    previous_elapsed = 0.0
    previous_result_path = args.output / "RESULT.json"
    if args.resume and previous_result_path.exists():
        previous_elapsed = float(json.loads(previous_result_path.read_text())["elapsed_seconds"])
    args.output.mkdir(parents=True, exist_ok=True)
    tasks, excluded = _tasks(args.raw_root, args.behavior_summary, args.output, args.window)
    protocol = {
        "schema_version": 1,
        "analysis": "aligned_numbered_definitions_train_test",
        "definitions": list(DEFINITIONS),
        "window_epochs": args.window,
        "source_epoch_exclusive_min": args.min_source_epoch_exclusive,
        "included_runs": len({task["run_id"] for task in tasks}),
        "excluded_runs": excluded,
        "expected_checkpoint_files": len(tasks),
        "source_behavior_summary_sha256": sha256(args.behavior_summary),
        "source_raw_audit_sha256": sha256(args.raw_root / "AUDIT.json"),
    }
    protocol_path = args.output / "protocol.json"
    if protocol_path.exists():
        if not args.resume or json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("existing output protocol differs or --resume was omitted")
    elif any(args.output.iterdir()):
        raise FileExistsError(f"refusing nonempty output directory {args.output}")
    else:
        write_json(protocol_path, protocol)

    completed: dict[tuple[str, int], dict[str, Any]] = {}
    files_path = args.output / "files.json"
    if args.resume and files_path.exists():
        for row in json.loads(files_path.read_text()):
            path = args.output / row["path"]
            if path.is_file() and sha256(path) == row["sha256"]:
                completed[(row["run_id"], int(row["epoch"]))] = row
    pending = [task for task in tasks if (task["run_id"], task["epoch"]) not in completed]

    def record(row: dict[str, Any]) -> None:
        completed[(row["run_id"], int(row["epoch"]))] = row
        ordered = [completed[key] for key in sorted(completed)]
        write_json(files_path, ordered)
        if len(completed) % 25 == 0 or len(completed) == len(tasks):
            print(f"definitions {len(completed)}/{len(tasks)}", flush=True)

    if pending:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(pending))) as executor:
            futures = [executor.submit(_compute_checkpoint, task) for task in pending]
            for future in as_completed(futures):
                record(future.result())
    if len(completed) != len(tasks):
        raise AssertionError("definition computation ended with missing checkpoints")
    files = [completed[key] for key in sorted(completed)]
    rows = _aggregate(args.output, files)
    plot_rows = [row for row in rows if int(row["epoch"]) > args.min_source_epoch_exclusive]
    endpoints = _endpoint_summary(plot_rows, args.output, args.window)
    stems = _plot(plot_rows, args.output)
    elapsed = max(previous_elapsed, time.perf_counter() - started)
    result = {
        "schema_version": 1,
        "status": "passed",
        "checkpoint_files": len(files),
        "valid_records": sum(int(row["ok"]) for row in files),
        "degenerate_records": sum(int(row["degenerate"]) for row in files),
        "aggregated_rows": len(rows),
        "plotted_aggregated_rows": len(plot_rows),
        "source_epoch_exclusive_min": args.min_source_epoch_exclusive,
        "figures": stems,
        "elapsed_seconds": elapsed,
    }
    write_json(args.output / "RESULT.json", result)
    _report(args.output, protocol, files, rows, plot_rows, endpoints, stems, elapsed)
    artifact_names = [
        "protocol.json",
        "files.json",
        "RESULT.json",
        "REPORT.md",
        "aligned_trajectories.csv",
        "endpoint_summary.csv",
        *[f"{stem}.{extension}" for stem in stems for extension in ("png", "pdf")],
    ]
    write_json(
        args.output / "OUTPUT_MANIFEST.json",
        [
            {
                "path": name,
                "bytes": (args.output / name).stat().st_size,
                "sha256": sha256(args.output / name),
            }
            for name in artifact_names
        ],
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
