#!/usr/bin/env python3
"""Build a compact, reviewer-facing visual addendum from completed artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from plot_forecasting_comparison import (
    METHODS,
    _evaluate,
    _run_features,
    _selected_runs,
)

from grokking_lab.io import sha256, write_json

COLORS = {"add": "#2878B5", "sub": "#D95F0E", "mul": "#2A9D6F"}
OPERATION_NAMES = {"add": "Addition", "sub": "Subtraction", "mul": "Multiplication"}
DEFINITIONS = tuple(f"definition_{index:02d}" for index in range(1, 6))
METHOD_LABELS = {
    "typical_timing": "Typical timing",
    "weight_norm": "Weight norm",
    "fourier_progress": "Fourier structure",
    "safe_mass_nested": "Safe-Mass",
}
METHOD_COLORS = {
    "typical_timing": "#C9CDD3",
    "weight_norm": "#7F8792",
    "fourier_progress": "#C9A227",
    "safe_mass_nested": "#2878B5",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plotting():
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
        }
    )
    return plt


def _save(figure: Any, output: Path, stem: str) -> None:
    figure.savefig(output / f"{stem}.png", bbox_inches="tight", facecolor="white")
    figure.savefig(output / f"{stem}.pdf", bbox_inches="tight", facecolor="white")


def plot_coverage(behavior_path: Path, output: Path) -> list[dict[str, Any]]:
    plt = _plotting()
    source = _read_csv(behavior_path)
    rows = []
    for row in source:
        rows.append(
            {
                "run_id": row["run_id"],
                "operation": row["operation"],
                "train_fraction": float(row["train_fraction"]),
                "weight_decay": float(row["weight_decay"]),
                "seed": int(row["seed"]),
                "grokking_epoch": int(row["grokking_epoch"]) if row["grokking_epoch"] else "",
                "censored_at_epoch": "" if row["grokking_epoch"] else 50_000,
            }
        )
    cells = sorted(
        {(row["operation"], row["train_fraction"], row["weight_decay"]) for row in rows},
        key=lambda cell: (("add", "sub", "mul").index(cell[0]), cell[1], cell[2]),
    )
    y_of = {cell: index for index, cell in enumerate(cells)}
    figure, axis = plt.subplots(figsize=(9.2, 5.5))
    markers = {0: "o", 1: "s", 2: "D"}
    offsets = {0: -0.16, 1: 0.0, 2: 0.16}
    for row in rows:
        cell = (row["operation"], row["train_fraction"], row["weight_decay"])
        x = float(row["grokking_epoch"] or row["censored_at_epoch"]) / 1000
        kwargs = {}
        if not row["grokking_epoch"]:
            kwargs = {"facecolors": "none", "linewidths": 1.6}
        axis.scatter(
            x,
            y_of[cell] + offsets[int(row["seed"])],
            marker=markers[int(row["seed"])],
            s=55,
            color=COLORS[str(row["operation"])],
            edgecolors=COLORS[str(row["operation"])],
            zorder=3,
            **kwargs,
        )
        if not row["grokking_epoch"]:
            axis.annotate(
                "not by 50k",
                (x, y_of[cell] + offsets[int(row["seed"])]),
                xytext=(-6, 8),
                textcoords="offset points",
                ha="right",
                fontsize=7,
            )
    axis.axvline(30, color="#666B73", linestyle="--", linewidth=1, label="30k source horizon")
    axis.axvline(50, color="#20242A", linestyle=":", linewidth=1, label="50k main horizon")
    axis.set_yticks(
        range(len(cells)),
        [
            f"{OPERATION_NAMES[op]} · {fraction:.0%} train · wd {decay:g}"
            for op, fraction, decay in cells
        ],
    )
    axis.invert_yaxis()
    axis.set_xlim(0, 52)
    axis.set_xlabel("First grokking checkpoint (thousands of epochs)")
    axis.set_title(
        "Grokking time varies substantially across seeds and settings", fontweight="bold"
    )
    axis.grid(axis="x", color="#E3E6EA", linewidth=0.6)
    seed_handles = [
        plt.Line2D(
            [], [], marker=markers[seed], linestyle="", color="#4B515A", label=f"Seed {seed}"
        )
        for seed in (0, 1, 2)
    ]
    horizon_handles, horizon_labels = axis.get_legend_handles_labels()
    axis.legend(
        [*seed_handles, *horizon_handles],
        [*[handle.get_label() for handle in seed_handles], *horizon_labels],
        loc="lower right",
        fontsize=7.5,
        ncols=2,
    )
    figure.tight_layout()
    _save(figure, output, "figure01_experiment_coverage")
    plt.close(figure)
    _write_csv(output / "figure01_experiment_coverage.csv", rows)
    return rows


def definition_changes(aligned_path: Path) -> list[dict[str, Any]]:
    raw = _read_csv(aligned_path)
    groups: dict[tuple[str, str, str], dict[str, list[dict[str, str]]]] = defaultdict(
        lambda: {"pre": [], "post": []}
    )
    for row in raw:
        if int(row["epoch"]) <= 100:
            continue
        relative = int(row["relative_epoch"])
        phase = "pre" if -2000 <= relative <= -500 else "post" if 500 <= relative <= 2000 else None
        if phase:
            groups[(row["run_id"], row["operation"], row["split"])][phase].append(row)
    output = []
    for (run_id, operation, split), phases in sorted(groups.items()):
        if not phases["pre"] or not phases["post"]:
            continue
        for definition in DEFINITIONS:
            pre = float(np.mean([float(row[definition]) for row in phases["pre"]]))
            post = float(np.mean([float(row[definition]) for row in phases["post"]]))
            output.append(
                {
                    "run_id": run_id,
                    "operation": operation,
                    "split": split,
                    "definition": definition,
                    "pre_mean": pre,
                    "post_mean": post,
                    "change": post - pre,
                    "increased": int(post > pre),
                }
            )
    return output


def plot_definition_consistency(changes: list[dict[str, Any]], output: Path) -> None:
    plt = _plotting()
    from matplotlib.colors import TwoSlopeNorm

    summaries = []
    for split in ("train", "test"):
        for definition in DEFINITIONS:
            for operation in ("add", "sub", "mul"):
                values = [
                    row
                    for row in changes
                    if row["split"] == split
                    and row["definition"] == definition
                    and row["operation"] == operation
                ]
                summaries.append(
                    {
                        "split": split,
                        "definition": definition,
                        "operation": operation,
                        "runs_increasing": sum(int(row["increased"]) for row in values),
                        "runs_evaluable": len(values),
                        "fraction_increasing": float(np.mean([row["increased"] for row in values])),
                        "mean_change": float(np.mean([row["change"] for row in values])),
                    }
                )
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), sharey=True)
    image = None
    for axis, split in zip(axes, ("train", "test"), strict=True):
        matrix = np.asarray(
            [
                [
                    next(
                        row["fraction_increasing"]
                        for row in summaries
                        if row["split"] == split
                        and row["definition"] == definition
                        and row["operation"] == operation
                    )
                    for operation in ("add", "sub", "mul")
                ]
                for definition in DEFINITIONS
            ]
        )
        image = axis.imshow(
            matrix,
            cmap="RdYlGn",
            norm=TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1),
            aspect="auto",
        )
        axis.set_xticks(range(3), ["Addition", "Subtraction", "Multiplication"])
        axis.set_yticks(range(5), [name.replace("_", "-").title() for name in DEFINITIONS])
        axis.set_title(f"{split.title()} examples")
        for y, definition in enumerate(DEFINITIONS):
            for x, operation in enumerate(("add", "sub", "mul")):
                row = next(
                    row
                    for row in summaries
                    if row["split"] == split
                    and row["definition"] == definition
                    and row["operation"] == operation
                )
                axis.text(
                    x,
                    y,
                    f"{row['runs_increasing']}/{row['runs_evaluable']}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
    assert image is not None
    figure.subplots_adjust(left=0.12, right=0.87, top=0.84, bottom=0.16, wspace=0.12)
    color_axis = figure.add_axes([0.90, 0.22, 0.018, 0.55])
    figure.colorbar(image, cax=color_axis, label="Fraction of runs increasing")
    figure.suptitle(
        "Only Definition-03 rises consistently around grokking",
        y=0.99,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.01,
        "Cells show increasing/evaluable runs; change compares means 2,000–500 epochs "
        "before vs. 500–2,000 after G",
        ha="center",
        fontsize=7.5,
        color="#555B65",
    )
    _save(figure, output, "figure02_definition_consistency")
    plt.close(figure)
    _write_csv(output / "figure02_definition_consistency.csv", summaries)


def plot_forecast_calibration(predictions_path: Path, output: Path) -> None:
    plt = _plotting()
    methods = ("typical_timing", "safe_mass_nested")
    raw = [
        row
        for row in _read_csv(predictions_path)
        if row["fold"] == "new_seed" and row["method"] in methods
    ]
    output_rows = []
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 4.1), sharex=True, sharey=True)
    limits = (500, 60_000)
    for axis, method in zip(axes, methods, strict=True):
        rows = [row for row in raw if row["method"] == method]
        for operation in ("add", "sub", "mul"):
            selected = [row for row in rows if row["run_id"].startswith(operation)]
            actual = [10 ** float(row["target_log10_g"]) for row in selected]
            predicted = [10 ** float(row["prediction_log10_g"]) for row in selected]
            axis.scatter(
                actual,
                predicted,
                s=42,
                color=COLORS[operation],
                label=OPERATION_NAMES[operation],
                alpha=0.85,
                edgecolor="white",
                linewidth=0.5,
            )
            for row, target, estimate in zip(selected, actual, predicted, strict=True):
                output_rows.append(
                    {
                        "method": method,
                        "run_id": row["run_id"],
                        "operation": operation,
                        "actual_grokking_epoch": target,
                        "predicted_grokking_epoch": estimate,
                        "absolute_error_log10": float(row["absolute_error"]),
                    }
                )
        axis.plot(limits, limits, color="#555B65", linestyle="--", linewidth=1)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(*limits)
        axis.set_ylim(*limits)
        mae = np.mean([float(row["absolute_error"]) for row in rows])
        axis.set_title(f"{METHOD_LABELS[method]} · MAE {mae:.3f}")
        axis.grid(color="#E3E6EA", linewidth=0.5)
        axis.set_xlabel("Actual grokking epoch")
    axes[0].set_ylabel("Predicted grokking epoch")
    axes[0].legend(fontsize=7.5, loc="upper left")
    figure.suptitle(
        "Safe-Mass broadens forecasts but does not eliminate large errors", fontweight="bold"
    )
    figure.tight_layout()
    _save(figure, output, "figure03_forecast_calibration")
    plt.close(figure)
    _write_csv(output / "figure03_forecast_calibration.csv", output_rows)


def plot_train_test_agreement(changes: list[dict[str, Any]], output: Path) -> float:
    plt = _plotting()
    selected = [row for row in changes if row["definition"] == "definition_03"]
    keyed = {(row["run_id"], row["split"]): row for row in selected}
    rows = []
    for run_id in sorted({row["run_id"] for row in selected}):
        if (run_id, "train") not in keyed or (run_id, "test") not in keyed:
            continue
        train = keyed[(run_id, "train")]
        test = keyed[(run_id, "test")]
        rows.append(
            {
                "run_id": run_id,
                "operation": train["operation"],
                "train_change": train["change"],
                "test_change": test["change"],
            }
        )
    x = np.asarray([row["train_change"] for row in rows])
    y = np.asarray([row["test_change"] for row in rows])
    correlation = float(np.corrcoef(x, y)[0, 1])
    figure, axis = plt.subplots(figsize=(5.4, 4.6))
    for operation in ("add", "sub", "mul"):
        operation_rows = [row for row in rows if row["operation"] == operation]
        axis.scatter(
            [row["train_change"] for row in operation_rows],
            [row["test_change"] for row in operation_rows],
            s=48,
            color=COLORS[operation],
            label=OPERATION_NAMES[operation],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )
    low = min(float(x.min()), float(y.min()), 0)
    high = max(float(x.max()), float(y.max())) * 1.08
    axis.plot([low, high], [low, high], color="#555B65", linestyle="--", linewidth=1)
    axis.axhline(0, color="#C9CDD3", linewidth=0.7)
    axis.axvline(0, color="#C9CDD3", linewidth=0.7)
    axis.set_xlim(low, high)
    axis.set_ylim(low, high)
    axis.set_xlabel("Change on training examples")
    axis.set_ylabel("Change on test examples")
    axis.set_title("Definition-03 changes agree across example splits", fontweight="bold")
    axis.text(
        0.97,
        0.05,
        f"n={len(rows)} runs · correlation={correlation:.3f}",
        transform=axis.transAxes,
        ha="right",
        fontsize=8,
        color="#555B65",
    )
    axis.legend(fontsize=7.5, loc="upper left")
    axis.grid(color="#E3E6EA", linewidth=0.5)
    figure.tight_layout()
    _save(figure, output, "figure04_train_test_agreement")
    plt.close(figure)
    _write_csv(output / "figure04_train_test_agreement.csv", rows)
    return correlation


def plot_forecast_sensitivity(
    runs_root: Path,
    raw_root: Path,
    behavior_path: Path,
    checkpoint_features_path: Path,
    output: Path,
) -> list[dict[str, Any]]:
    plt = _plotting()
    runs, _ = _selected_runs(runs_root, behavior_path, raw_root, 100, 4)
    checkpoint_rows = _read_csv(checkpoint_features_path)
    rows = []
    for points in (2, 3, 4):
        point_runs = deepcopy(runs)
        for run in point_runs:
            run["window"] = run["window"][-points:]
        run_features = _run_features(point_runs, checkpoint_rows)
        metrics, _ = _evaluate(run_features, 1.0)
        rows.extend({"window_checkpoints": points, **row} for row in metrics)
    folds = ("new_seed", "new_setting", "new_operation")
    titles = ("New seed", "New setting", "New operation")
    figure, axes = plt.subplots(1, 3, figsize=(9.5, 3.5), sharey=True)
    for axis, fold, title in zip(axes, folds, titles, strict=True):
        for method in (*METHODS, "safe_mass_nested"):
            values = [
                next(
                    float(row["mae_log10_grokking_epoch"])
                    for row in rows
                    if row["fold"] == fold
                    and row["method"] == method
                    and row["window_checkpoints"] == points
                )
                for points in (2, 3, 4)
            ]
            axis.plot(
                (2, 3, 4),
                values,
                marker="o",
                linewidth=1.8,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
        axis.set_title(title)
        axis.set_xticks((2, 3, 4))
        axis.set_xlabel("Checkpoints used")
        axis.grid(axis="y", color="#E3E6EA", linewidth=0.6)
    axes[0].set_ylabel("Forecast MAE (log10 epochs) ↓")
    axes[-1].legend(fontsize=7, loc="upper right")
    figure.suptitle("Forecast conclusions depend on the observation window", fontweight="bold")
    figure.tight_layout()
    _save(figure, output, "figure05_forecast_window_sensitivity")
    plt.close(figure)
    _write_csv(output / "figure05_forecast_window_sensitivity.csv", rows)
    return rows


def plot_selection_stability(predictions_path: Path, output: Path) -> list[dict[str, Any]]:
    plt = _plotting()
    raw = [row for row in _read_csv(predictions_path) if row["method"] == "safe_mass_nested"]
    rows = []
    for fold in ("new_seed", "new_setting", "new_operation"):
        by_group = {row["group"]: row["selected_candidate"] for row in raw if row["fold"] == fold}
        counts = Counter(by_group.values())
        for definition in DEFINITIONS:
            rows.append(
                {
                    "fold": fold,
                    "definition": definition,
                    "held_out_groups_selecting": counts[definition],
                    "held_out_groups_total": len(by_group),
                }
            )
    figure, axis = plt.subplots(figsize=(7.0, 3.8))
    x = np.arange(3)
    bottom = np.zeros(3)
    palette = ("#BDD7E7", "#6BAED6", "#3182BD", "#756BB1", "#31A354")
    for definition, color in zip(DEFINITIONS, palette, strict=True):
        values = np.asarray(
            [
                next(
                    row["held_out_groups_selecting"]
                    for row in rows
                    if row["fold"] == fold and row["definition"] == definition
                )
                for fold in ("new_seed", "new_setting", "new_operation")
            ]
        )
        axis.bar(
            x,
            values,
            bottom=bottom,
            color=color,
            label=definition.replace("_", "-").title(),
        )
        bottom += values
    axis.set_xticks(x, ["New seed", "New setting", "New operation"])
    axis.set_ylabel("Held-out groups")
    axis.set_title(
        "Definition choice is stable within settings, unstable across tasks", fontweight="bold"
    )
    axis.legend(fontsize=7, ncols=3, loc="upper right")
    axis.set_ylim(0, float(bottom.max()) * 1.10)
    for index, total in enumerate(bottom):
        axis.text(index, total + 0.25, f"n={int(total)}", ha="center", fontsize=7)
    axis.grid(axis="y", color="#E3E6EA", linewidth=0.6)
    figure.tight_layout()
    _save(figure, output, "figure06_definition_selection_stability")
    plt.close(figure)
    _write_csv(output / "figure06_definition_selection_stability.csv", rows)
    return rows


def plot_censored_run(paths: list[Path], output: Path) -> int:
    plt = _plotting()
    by_epoch = {}
    for path in paths:
        for row in _read_csv(path):
            by_epoch[int(row["epoch"])] = row
    rows = [
        {
            "epoch": epoch,
            "train_accuracy": float(by_epoch[epoch]["train_accuracy"]),
            "test_accuracy": float(by_epoch[epoch]["test_accuracy"]),
        }
        for epoch in sorted(by_epoch)
    ]
    grokking_epoch = next(
        row["epoch"]
        for row in rows
        if row["train_accuracy"] >= 0.99 and row["test_accuracy"] >= 0.90
    )
    figure, axis = plt.subplots(figsize=(8.6, 3.8))
    axis.axvspan(50, 200, color="#F1F3F5", alpha=0.8, label="Post-hoc extension")
    axis.plot(
        [row["epoch"] / 1000 for row in rows],
        [row["test_accuracy"] for row in rows],
        color="#D95F0E",
        linewidth=1.8,
        label="Test accuracy",
    )
    axis.plot(
        [row["epoch"] / 1000 for row in rows],
        [row["train_accuracy"] for row in rows],
        color="#7F8792",
        linewidth=1,
        alpha=0.7,
        label="Train accuracy",
    )
    axis.axhline(0.90, color="#20242A", linestyle="--", linewidth=1, label="Grokking threshold")
    axis.axvline(50, color="#555B65", linestyle=":", linewidth=1)
    axis.axvline(grokking_epoch / 1000, color="#D95F0E", linestyle="--", linewidth=1)
    axis.annotate(
        f"G = {grokking_epoch // 1000}k",
        (grokking_epoch / 1000, 0.90),
        xytext=(8, -18),
        textcoords="offset points",
        fontsize=8,
    )
    axis.set_xlim(0, 200)
    axis.set_ylim(0, 1.03)
    axis.set_xlabel("Epoch (thousands)")
    axis.set_ylabel("Accuracy")
    axis.set_title(
        "The 50k-censored run groks only in the isolated post-hoc extension", fontweight="bold"
    )
    axis.legend(fontsize=7.5, ncols=2, loc="lower right")
    axis.grid(axis="y", color="#E3E6EA", linewidth=0.6)
    figure.tight_layout()
    _save(figure, output, "figure07_censored_run_posthoc")
    plt.close(figure)
    _write_csv(output / "figure07_censored_run_posthoc.csv", rows)
    return int(grokking_epoch)


def plot_pipeline(output: Path) -> None:
    plt = _plotting()
    from matplotlib.patches import FancyBboxPatch

    figure, axis = plt.subplots(figsize=(12.2, 3.1))
    axis.set_xlim(0, 12.2)
    axis.set_ylim(0, 3.1)
    axis.axis("off")
    boxes = (
        (0.2, 1.25, 1.75, 1.0, "Study design", "9 cells · 3 seeds"),
        (2.25, 1.25, 1.75, 1.0, "Behavior training", "27 runs · 50k epochs"),
        (4.3, 1.25, 1.75, 1.0, "Raw flows", "876,096 labeled graphs"),
        (6.35, 1.25, 1.75, 1.0, "Numbered candidates", "Definition-01–05"),
        (8.4, 1.25, 1.75, 1.0, "Held-out tests", "seed · setting · task"),
        (10.45, 1.25, 1.55, 1.0, "Conclusions", "signal + limits"),
    )
    colors = ("#E8EEF5", "#E8EEF5", "#E5F2EC", "#FFF1CC", "#E8EEF5", "#EDE7F6")
    for (x, y, width, height, title, subtitle), color in zip(boxes, colors, strict=True):
        axis.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.04,rounding_size=0.08",
                facecolor=color,
                edgecolor="#60656D",
                linewidth=0.9,
            )
        )
        axis.text(
            x + width / 2, y + 0.64, title, ha="center", va="center", fontweight="bold", fontsize=9
        )
        axis.text(
            x + width / 2,
            y + 0.33,
            subtitle,
            ha="center",
            va="center",
            fontsize=7.5,
            color="#555B65",
        )
    for left, right in zip(boxes[:-1], boxes[1:], strict=True):
        axis.annotate(
            "",
            xy=(right[0] - 0.06, 1.75),
            xytext=(left[0] + left[2] + 0.06, 1.75),
            arrowprops={"arrowstyle": "->", "color": "#60656D", "linewidth": 1.2},
        )
    axis.add_patch(
        FancyBboxPatch(
            (2.5, 0.15),
            2.7,
            0.6,
            boxstyle="round,pad=0.03,rounding_size=0.06",
            facecolor="#F8E7E7",
            edgecolor="#A45A52",
            linestyle="--",
            linewidth=0.9,
        )
    )
    axis.text(
        3.85,
        0.45,
        "Post-hoc 200k continuation kept isolated",
        ha="center",
        va="center",
        fontsize=8,
        color="#7A3F39",
    )
    axis.annotate(
        "",
        xy=(3.15, 1.23),
        xytext=(3.5, 0.76),
        arrowprops={"arrowstyle": "->", "color": "#A45A52", "linestyle": "--"},
    )
    axis.set_title(
        "From reproducible training to bounded forecasting claims", fontweight="bold", pad=4
    )
    figure.tight_layout()
    _save(figure, output, "figure08_research_pipeline")
    plt.close(figure)


def _report(
    output: Path, coverage: list[dict[str, Any]], correlation: float, grokking: int
) -> None:
    grokked = sum(bool(row["grokking_epoch"]) for row in coverage)
    (output / "README.md").write_text(
        f"""# MATS application visual addendum

This directory is a reviewer-facing summary generated entirely from the completed
main study and explicitly isolated post-hoc analysis. Lower-level reports and raw
artifacts remain the source of record.

## Recommended core sequence

1. [Experiment coverage](figure01_experiment_coverage.png): {grokked}/27 runs grokked
   by 50k; the dot plot exposes seed and setting variability.
2. [Definition consistency](figure02_definition_consistency.png): Definition-03 is
   the only candidate increasing for every evaluable run in every operation and split
   over the fixed pre/post windows.
3. [Forecast calibration](figure03_forecast_calibration.png): actual-versus-predicted
   timing makes forecast gains and large residual errors visible.
4. [Train/test agreement](figure04_train_test_agreement.png): Definition-03 changes
   agree across example splits (correlation {correlation:.3f}).

## Robustness and research judgment

5. [Forecast-window sensitivity](figure05_forecast_window_sensitivity.png): the
   Safe-Mass result depends on using four rather than two or three checkpoints.
6. [Definition-selection stability](figure06_definition_selection_stability.png):
   Definition-04 dominates seed/setting selection, while task transfer is unstable.
7. [Censored-run continuation](figure07_censored_run_posthoc.png): the sole censored
   main run groks at epoch {grokking:,}, but only in isolated post-hoc analysis.
8. [Research pipeline](figure08_research_pipeline.png): concise scope and provenance.

Every result figure has a PDF counterpart and a CSV containing its plotted values.
The pipeline schematic has no underlying numeric table. `MANIFEST.json` records
SHA-256 and byte size for every addendum file except itself.

## Interpretation boundary

These figures support a replicated structural transition and a modest, window-sensitive
within-setting forecast signal. They do not establish causal mechanism, universal
operation transfer, or a deployable online predictor. The application should use the
four core figures in the main work sample and keep Figures 5–8 as supporting evidence.
"""
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("application-addendum"))
    parser.add_argument(
        "--runs-root", type=Path, default=Path("runs/selected_cells_seed0_2_epoch50000")
    )
    parser.add_argument("--raw-root", type=Path, default=Path("flow-artifacts/raw-main-50k"))
    parser.add_argument(
        "--definitions-root",
        type=Path,
        default=Path("flow-artifacts/definition-aligned-plots"),
    )
    parser.add_argument(
        "--forecast-root", type=Path, default=Path("flow-artifacts/forecasting-comparison")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    behavior = args.runs_root / "behavior_summary.csv"
    coverage = plot_coverage(behavior, args.output)
    changes = definition_changes(args.definitions_root / "aligned_trajectories.csv")
    plot_definition_consistency(changes, args.output)
    plot_forecast_calibration(args.forecast_root / "held_out_predictions.csv", args.output)
    correlation = plot_train_test_agreement(changes, args.output)
    plot_forecast_sensitivity(
        args.runs_root,
        args.raw_root,
        behavior,
        args.forecast_root / "checkpoint_features.csv",
        args.output,
    )
    plot_selection_stability(args.forecast_root / "held_out_predictions.csv", args.output)
    grokking = plot_censored_run(
        [
            args.runs_root / "sub_frac0p25_wd1_seed1" / "metrics.csv",
            Path("runs/posthoc/sub_frac0p25_wd1_seed1_to100000/metrics.csv"),
            Path("runs/posthoc/sub_frac0p25_wd1_seed1_to200000/metrics.csv"),
        ],
        args.output,
    )
    plot_pipeline(args.output)
    _report(args.output, coverage, correlation, grokking)
    names = sorted(
        path.name
        for path in args.output.iterdir()
        if path.is_file() and path.name != "MANIFEST.json"
    )
    write_json(
        args.output / "MANIFEST.json",
        [
            {
                "path": name,
                "bytes": (args.output / name).stat().st_size,
                "sha256": sha256(args.output / name),
            }
            for name in names
        ],
    )
    print(json.dumps({"status": "passed", "files": len(names), "output": str(args.output)}))


if __name__ == "__main__":
    main()
