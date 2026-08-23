"""Small, leakage-aware helpers for grouped grokking forecasts."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def ridge_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    regularization: float = 1.0,
) -> np.ndarray:
    """Fit standardized ridge regression using training-fold statistics only."""

    means = np.nanmean(train_x, axis=0)
    means = np.where(np.isfinite(means), means, 0.0)
    train_x = np.where(np.isfinite(train_x), train_x, means)
    test_x = np.where(np.isfinite(test_x), test_x, means)
    center = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale == 0] = 1.0
    train_z = (train_x - center) / scale
    test_z = (test_x - center) / scale
    intercept = float(train_y.mean())
    gram = train_z.T @ train_z + regularization * np.eye(train_z.shape[1])
    weights = np.linalg.solve(gram, train_z.T @ (train_y - intercept))
    return intercept + test_z @ weights


def grouped_predictions(
    rows: list[dict[str, Any]],
    feature_names: list[str],
    group_of: Callable[[dict[str, Any]], str],
    regularization: float = 1.0,
) -> list[dict[str, Any]]:
    """Return one held-out prediction per row for leave-one-group-out folds."""

    targets = np.asarray([float(row["target_log10_g"]) for row in rows])
    features = np.asarray(
        [[float(row[name]) for name in feature_names] for row in rows], dtype=float
    )
    groups = [group_of(row) for row in rows]
    predictions = []
    for group in sorted(set(groups)):
        held = np.asarray([index for index, value in enumerate(groups) if value == group])
        train = np.asarray([index for index, value in enumerate(groups) if value != group])
        if feature_names:
            predicted = ridge_predict(
                features[train], targets[train], features[held], regularization
            )
        else:
            predicted = np.full(len(held), targets[train].mean())
        for index, value in zip(held, predicted, strict=True):
            predictions.append(
                {
                    "run_id": rows[index]["run_id"],
                    "group": group,
                    "target_log10_g": float(targets[index]),
                    "prediction_log10_g": float(value),
                    "absolute_error": abs(float(value) - float(targets[index])),
                }
            )
    return sorted(predictions, key=lambda row: str(row["run_id"]))


def nested_candidate_predictions(
    rows: list[dict[str, Any]],
    candidates: dict[str, list[str]],
    group_of: Callable[[dict[str, Any]], str],
    regularization: float = 1.0,
    inner_group_of: Callable[[dict[str, Any]], str] | None = None,
) -> list[dict[str, Any]]:
    """Choose a candidate by inner grouped CV, then predict each outer group."""

    output = []
    selection_group_of = inner_group_of or group_of
    outer_groups = [group_of(row) for row in rows]
    for outer_group in sorted(set(outer_groups)):
        train_rows = [row for row in rows if group_of(row) != outer_group]
        held_rows = [row for row in rows if group_of(row) == outer_group]
        inner_mae = {}
        for name, features in candidates.items():
            predictions = grouped_predictions(
                train_rows, features, selection_group_of, regularization
            )
            inner_mae[name] = float(np.mean([row["absolute_error"] for row in predictions]))
        selected = min(inner_mae, key=lambda name: (inner_mae[name], name))
        feature_names = candidates[selected]
        train_x = np.asarray([[float(row[name]) for name in feature_names] for row in train_rows])
        train_y = np.asarray([float(row["target_log10_g"]) for row in train_rows])
        held_x = np.asarray([[float(row[name]) for name in feature_names] for row in held_rows])
        predicted = ridge_predict(train_x, train_y, held_x, regularization=regularization)
        for row, value in zip(held_rows, predicted, strict=True):
            target = float(row["target_log10_g"])
            output.append(
                {
                    "run_id": row["run_id"],
                    "group": outer_group,
                    "selected_candidate": selected,
                    "inner_cv_mae": inner_mae[selected],
                    "target_log10_g": target,
                    "prediction_log10_g": float(value),
                    "absolute_error": abs(float(value) - target),
                }
            )
    return sorted(output, key=lambda row: str(row["run_id"]))
