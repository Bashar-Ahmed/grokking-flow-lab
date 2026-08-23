#!/usr/bin/env python3
"""Compare pre-generalization grokking forecasters under grouped holdouts."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch

from grokking_lab.definitions import compute_definitions, graph_from_record
from grokking_lab.forecast import grouped_predictions, nested_candidate_predictions
from grokking_lab.io import sha256, write_json

DEFINITIONS = tuple(f"definition_{index:02d}" for index in range(1, 6))
METHODS = {
    "typical_timing": [],
    "weight_norm": [
        "parameter_l2_norm_last",
        "parameter_l2_norm_slope",
    ],
    "fourier_progress": [
        "fourier_fraction_last",
        "fourier_fraction_slope",
    ],
}
SAFE_MASS_CANDIDATES = {
    definition: [f"{definition}_last", f"{definition}_slope"] for definition in DEFINITIONS
}
FOLDS = {
    "new_seed": lambda row: str(row["run_id"]),
    "new_setting": lambda row: str(row["cell"]),
    "new_operation": lambda row: str(row["operation"]),
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _checkpoint_feature(task: dict[str, Any]) -> dict[str, Any]:
    values = {definition: [] for definition in DEFINITIONS}
    with gzip.open(task["raw_path"], "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["split"] != "train" or record["status"] != "ok":
                continue
            definitions = compute_definitions(graph_from_record(record, int(task["num_nodes"])))
            for definition in DEFINITIONS:
                values[definition].append(float(definitions[definition]))
    if not all(values.values()):
        raise ValueError(f"no usable train records in {task['raw_path']}")
    checkpoint = torch.load(task["checkpoint_path"], map_location="cpu", weights_only=True)
    norm_squared = sum(
        float(tensor.double().square().sum()) for tensor in checkpoint["model_state"].values()
    )
    return {
        "run_id": task["run_id"],
        "epoch": task["epoch"],
        "parameter_l2_norm": math.sqrt(norm_squared),
        "train_flow_records": len(values[DEFINITIONS[0]]),
        **{definition: float(np.mean(items)) for definition, items in values.items()},
    }


def _selected_runs(
    runs_root: Path,
    behavior_summary: Path,
    raw_root: Path,
    min_epoch: int,
    points: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected = []
    tasks = []
    for behavior in _read_csv(behavior_summary):
        if not behavior["grokking_epoch"] or not behavior["first_test_gt_10_epoch"]:
            continue
        run_id = behavior["run_id"]
        metrics = _read_csv(runs_root / run_id / "metrics.csv")
        cutoff = int(behavior["first_test_gt_10_epoch"])
        eligible = [
            row for row in metrics if int(row["epoch"]) > min_epoch and int(row["epoch"]) < cutoff
        ]
        if len(eligible) < points:
            continue
        window = eligible[-points:]
        cell = (
            f"{behavior['operation']}_frac{behavior['train_fraction']}_wd{behavior['weight_decay']}"
        )
        selected.append(
            {
                "run_id": run_id,
                "operation": behavior["operation"],
                "cell": cell,
                "seed": int(behavior["seed"]),
                "grokking_epoch": int(behavior["grokking_epoch"]),
                "cutoff_epoch": cutoff,
                "window": window,
            }
        )
        raw_manifest = json.loads((raw_root / run_id / "manifest.json").read_text())
        for row in window:
            epoch = int(row["epoch"])
            tasks.append(
                {
                    "run_id": run_id,
                    "epoch": epoch,
                    "num_nodes": len(raw_manifest["node_labels"]),
                    "raw_path": str(raw_root / run_id / f"epoch_{epoch:06d}.jsonl.gz"),
                    "checkpoint_path": str(
                        runs_root / run_id / "checkpoints" / f"epoch_{epoch:06d}.pt"
                    ),
                }
            )
    return selected, tasks


def _last_slope(epochs: list[int], values: list[float]) -> tuple[float, float]:
    x = np.log10(np.asarray(epochs, dtype=float))
    y = np.asarray(values, dtype=float)
    finite = np.isfinite(y)
    last = float(y[finite][-1]) if finite.any() else math.nan
    slope = float(np.polyfit(x[finite], y[finite], 1)[0]) if finite.sum() >= 2 else math.nan
    return last, slope


def _run_features(
    runs: list[dict[str, Any]], checkpoint_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    checkpoint_of = {(str(row["run_id"]), int(row["epoch"])): row for row in checkpoint_rows}
    output = []
    for run in runs:
        epochs = [int(row["epoch"]) for row in run["window"]]
        row: dict[str, Any] = {
            "run_id": run["run_id"],
            "operation": run["operation"],
            "cell": run["cell"],
            "seed": run["seed"],
            "grokking_epoch": run["grokking_epoch"],
            "target_log10_g": math.log10(run["grokking_epoch"]),
            "cutoff_epoch": run["cutoff_epoch"],
            "window_start_epoch": epochs[0],
            "window_end_epoch": epochs[-1],
        }
        metric_of = {int(item["epoch"]): item for item in run["window"]}
        series = {
            "train_loss": [float(metric_of[epoch]["train_loss"]) for epoch in epochs],
            "step_loss": [float(metric_of[epoch]["step_loss"]) for epoch in epochs],
            "fourier_fraction": [float(metric_of[epoch]["fourier_fraction"]) for epoch in epochs],
            "parameter_l2_norm": [
                float(checkpoint_of[(run["run_id"], epoch)]["parameter_l2_norm"])
                for epoch in epochs
            ],
            **{
                definition: [
                    float(checkpoint_of[(run["run_id"], epoch)][definition]) for epoch in epochs
                ]
                for definition in DEFINITIONS
            },
        }
        for name, values in series.items():
            last, slope = _last_slope(epochs, values)
            row[f"{name}_last"] = last
            row[f"{name}_slope"] = slope
        output.append(row)
    return sorted(output, key=lambda row: str(row["run_id"]))


def _evaluate(rows: list[dict[str, Any]], regularization: float) -> tuple[list, list]:
    metrics = []
    all_predictions = []
    for fold, group_of in FOLDS.items():
        for method, features in METHODS.items():
            predictions = grouped_predictions(
                rows, features, group_of, regularization=regularization
            )
            errors = [float(row["absolute_error"]) for row in predictions]
            metrics.append(
                {
                    "fold": fold,
                    "method": method,
                    "mae_log10_grokking_epoch": float(np.mean(errors)),
                    "runs": len(errors),
                    "held_out_groups": len({row["group"] for row in predictions}),
                }
            )
            all_predictions.extend({"fold": fold, "method": method, **row} for row in predictions)
        predictions = nested_candidate_predictions(
            rows, SAFE_MASS_CANDIDATES, group_of, regularization=regularization
        )
        errors = [float(row["absolute_error"]) for row in predictions]
        metrics.append(
            {
                "fold": fold,
                "method": "safe_mass_nested",
                "mae_log10_grokking_epoch": float(np.mean(errors)),
                "runs": len(errors),
                "held_out_groups": len({row["group"] for row in predictions}),
            }
        )
        all_predictions.extend(
            {"fold": fold, "method": "safe_mass_nested", **row} for row in predictions
        )
    return metrics, all_predictions


def _plot(metrics: list[dict[str, Any]], output: Path, run_count: int, points: int) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
        }
    )
    folds = ("new_seed", "new_setting", "new_operation")
    fold_labels = (
        "Same setting,\nnew seed",
        "New hyperparameter\nsetting",
        "New arithmetic\noperation",
    )
    methods = (*METHODS, "safe_mass_nested")
    method_labels = {
        "typical_timing": "Typical timing",
        "weight_norm": "Weight-norm dynamics",
        "fourier_progress": "Fourier-structure progress",
        "safe_mass_nested": "Safe-Mass (nested selection)",
    }
    colors = {
        "typical_timing": "#C9CDD3",
        "weight_norm": "#7F8792",
        "fourier_progress": "#C9A227",
        "safe_mass_nested": "#2878B5",
    }
    lookup = {
        (str(row["fold"]), str(row["method"])): float(row["mae_log10_grokking_epoch"])
        for row in metrics
    }
    figure, axis = plt.subplots(figsize=(10.2, 4.6))
    x = np.arange(len(folds))
    width = 0.19
    for index, method in enumerate(methods):
        values = [lookup[(fold, method)] for fold in folds]
        bars = axis.bar(
            x + (index - 1.5) * width,
            values,
            width,
            color=colors[method],
            label=method_labels[method],
        )
        axis.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=2, fontsize=7)
    axis.set_xticks(x, fold_labels)
    axis.set_ylabel("Mean absolute timing error (log10 epochs)  ↓")
    axis.grid(axis="y", color="#E3E6EA", linewidth=0.6)
    axis.set_axisbelow(True)
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        fontsize=8,
        ncols=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.86),
        frameon=False,
    )
    figure.suptitle(
        "Forecasting grokking from pre-generalization checkpoints",
        y=0.98,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.90,
        "Safe-Mass helps within settings, but not across operations",
        ha="center",
        fontsize=11,
    )
    figure.text(
        0.5,
        0.02,
        f"Lower is better  •  0.30 ≈ 2× epoch error  •  {run_count} runs  •  "
        f"last {points} checkpoints before test accuracy >10%",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#555B65",
    )
    figure.tight_layout(rect=(0, 0.08, 1, 0.78))
    figure.savefig(output / "forecasting_comparison.png", bbox_inches="tight", facecolor="white")
    figure.savefig(output / "forecasting_comparison.pdf", bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _report(
    output: Path,
    protocol: dict[str, Any],
    metrics: list[dict[str, Any]],
    excluded: list[str],
    elapsed: float,
) -> None:
    lookup = {
        (str(row["fold"]), str(row["method"])): float(row["mae_log10_grokking_epoch"])
        for row in metrics
    }
    table = "\n".join(
        "| {label} | {mean:.3f} | {weight:.3f} | {fourier:.3f} | {safe:.3f} |".format(
            label=label,
            mean=lookup[(fold, "typical_timing")],
            weight=lookup[(fold, "weight_norm")],
            fourier=lookup[(fold, "fourier_progress")],
            safe=lookup[(fold, "safe_mass_nested")],
        )
        for fold, label in (
            ("new_seed", "Same setting, new seed"),
            ("new_setting", "New hyperparameter setting"),
            ("new_operation", "New arithmetic operation"),
        )
    )
    excluded_text = ", ".join(f"`{name}`" for name in excluded)
    seed_gain = (
        1 - lookup[("new_seed", "safe_mass_nested")] / lookup[("new_seed", "typical_timing")]
    )
    setting_gain = (
        1 - lookup[("new_setting", "safe_mass_nested")] / lookup[("new_setting", "typical_timing")]
    )
    operation_change = (
        lookup[("new_operation", "safe_mass_nested")] / lookup[("new_operation", "typical_timing")]
        - 1
    )
    (output / "REPORT.md").write_text(
        f"""# Pre-generalization grokking-time forecasting

## Question and design

This post-hoc comparison asks whether early model signals predict the behavioral
grokking epoch. The target is `log10(grokking epoch)` and the score is held-out mean
absolute error, so lower is better; an error of 0.30 is approximately a factor-of-two
timing error. Each run contributes exactly the last
{protocol["window_checkpoints"]} saved checkpoints with source epoch strictly greater
than {protocol["source_epoch_exclusive_min"]} and strictly before test accuracy first
exceeds 10%. Safe-Mass features use training examples only.

Every learned forecaster has exactly two features and uses fixed ridge regularization
{protocol["ridge_lambda"]}:

1. **Typical timing:** the training-fold mean, with no checkpoint signals.
2. **Weight-norm dynamics:** last value and log-epoch slope of parameter L2 norm,
   motivated by established grokking optimization diagnostics.
3. **Fourier progress:** last value and slope of the operation-matched Fourier
   fraction, following mechanistic progress-measure work.
4. **Safe-Mass:** last value and slope of one numbered definition, averaged over
   train-example raw flows. The definition is selected among Definition-01--05 by
   inner grouped cross-validation using only the outer training fold.

## Results

| Held-out case | Typical timing | Weight norm | Fourier progress | Safe-Mass |
|---|---:|---:|---:|---:|
{table}

[PNG](forecasting_comparison.png) · [PDF](forecasting_comparison.pdf)

Safe-Mass reduces error relative to typical timing by {seed_gain:.1%} for a new seed
in a represented setting and {setting_gain:.1%} for a held-out hyperparameter setting.
It increases error by {operation_change:.1%} for a held-out arithmetic operation. It
beats the weight-norm and Fourier forecasters in all three comparisons, but does not
beat the no-signal timing baseline when transferring to a new operation.

## Scope and caveats

- Evaluable runs: {protocol["evaluable_runs"]} of 26 grokked main-study runs.
- Excluded for having fewer than four eligible checkpoints: {excluded_text}.
- The non-grokked main run is excluded because its target is right-censored; post-hoc
  continuation data are not used.
- “New seed” is leave-one-run-out, “new setting” holds out every available seed in one
  operation/fraction/weight-decay cell, and “new operation” holds out an arithmetic task.
- This is exploratory and post-hoc. The 10% test-accuracy boundary makes it a
  pre-generalization screening analysis, not a deployable online forecast. With only
  {protocol["evaluable_runs"]} runs and nine related cells, the bars are descriptive.
- The established references motivate predictor families rather than define a universal
  forecasting benchmark: [Power et al. (2022)](https://arxiv.org/abs/2201.02177),
  [Nanda et al. (2023)](https://openreview.net/forum?id=9XFSbDPmdW), and
  [Thilak et al. (2022)](https://arxiv.org/abs/2206.04817).

Machine-readable inputs, fold predictions, and metrics are provided beside the figure.
Computation and plotting took {elapsed:.1f} seconds.
"""
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--behavior-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--min-source-epoch-exclusive", type=int, default=100)
    parser.add_argument("--window-checkpoints", type=int, default=4)
    parser.add_argument("--ridge-lambda", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    previous_elapsed = 0.0
    previous_result = args.output / "RESULT.json"
    if args.resume and previous_result.exists():
        previous_elapsed = float(json.loads(previous_result.read_text())["elapsed_seconds"])
    runs, tasks = _selected_runs(
        args.runs_root,
        args.behavior_summary,
        args.raw_root,
        args.min_source_epoch_exclusive,
        args.window_checkpoints,
    )
    all_grokked = [row for row in _read_csv(args.behavior_summary) if row["grokking_epoch"]]
    selected_ids = {str(row["run_id"]) for row in runs}
    excluded = [row["run_id"] for row in all_grokked if row["run_id"] not in selected_ids]
    protocol = {
        "schema_version": 1,
        "analysis": "pre_generalization_grokking_forecasting",
        "target": "log10_grokking_epoch",
        "test_accuracy_cutoff_exclusive": 0.10,
        "source_epoch_exclusive_min": args.min_source_epoch_exclusive,
        "window_checkpoints": args.window_checkpoints,
        "safe_mass_example_split": "train",
        "definitions": list(DEFINITIONS),
        "ridge_lambda": args.ridge_lambda,
        "evaluable_runs": len(runs),
        "forecast_models": ("equal_complexity_last_and_slope_with_nested_safe_mass_selection"),
        "checkpoint_tasks": len(tasks),
        "source_behavior_summary_sha256": sha256(args.behavior_summary),
        "source_raw_audit_sha256": sha256(args.raw_root / "AUDIT.json"),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_path = args.output / "protocol.json"
    if protocol_path.exists():
        if not args.resume or json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("existing output protocol differs or --resume was omitted")
    elif any(args.output.iterdir()):
        raise FileExistsError(f"refusing nonempty output directory {args.output}")
    else:
        write_json(protocol_path, protocol)

    checkpoint_path = args.output / "checkpoint_features.csv"
    completed: dict[tuple[str, int], dict[str, Any]] = {}
    if args.resume and checkpoint_path.exists():
        for row in _read_csv(checkpoint_path):
            completed[(row["run_id"], int(row["epoch"]))] = row
    pending = [task for task in tasks if (task["run_id"], int(task["epoch"])) not in completed]
    if pending:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(pending))) as executor:
            futures = [executor.submit(_checkpoint_feature, task) for task in pending]
            for index, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                completed[(str(row["run_id"]), int(row["epoch"]))] = row
                if index % 10 == 0 or index == len(pending):
                    print(f"forecast features {len(completed)}/{len(tasks)}", flush=True)
    checkpoint_rows = [completed[key] for key in sorted(completed)]
    _write_csv(checkpoint_path, checkpoint_rows)
    run_rows = _run_features(runs, checkpoint_rows)
    _write_csv(args.output / "run_features.csv", run_rows)
    metrics, predictions = _evaluate(run_rows, args.ridge_lambda)
    _write_csv(args.output / "fold_metrics.csv", metrics)
    _write_csv(args.output / "held_out_predictions.csv", predictions)
    _plot(metrics, args.output, len(runs), args.window_checkpoints)
    elapsed = max(previous_elapsed, time.perf_counter() - started)
    _report(args.output, protocol, metrics, excluded, elapsed)
    result = {
        "schema_version": 1,
        "status": "passed",
        "evaluable_runs": len(runs),
        "checkpoint_features": len(checkpoint_rows),
        "elapsed_seconds": elapsed,
        "fold_metrics": metrics,
    }
    write_json(args.output / "RESULT.json", result)
    names = [
        "protocol.json",
        "RESULT.json",
        "REPORT.md",
        "checkpoint_features.csv",
        "run_features.csv",
        "fold_metrics.csv",
        "held_out_predictions.csv",
        "forecasting_comparison.png",
        "forecasting_comparison.pdf",
    ]
    write_json(
        args.output / "OUTPUT_MANIFEST.json",
        [
            {
                "path": name,
                "bytes": (args.output / name).stat().st_size,
                "sha256": sha256(args.output / name),
            }
            for name in names
        ],
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
