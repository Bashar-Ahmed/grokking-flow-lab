import numpy as np

from grokking_lab.forecast import (
    grouped_predictions,
    nested_candidate_predictions,
    ridge_predict,
)


def test_ridge_predict_recovers_simple_relationship() -> None:
    train_x = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    train_y = np.asarray([1.0, 3.0, 5.0, 7.0])
    prediction = ridge_predict(train_x, train_y, np.asarray([[1.5]]), regularization=0.0)
    assert np.allclose(prediction, [4.0])


def test_grouped_predictions_uses_training_group_mean_for_empty_features() -> None:
    rows = [
        {"run_id": "a", "cell": "one", "target_log10_g": 1.0},
        {"run_id": "b", "cell": "two", "target_log10_g": 3.0},
        {"run_id": "c", "cell": "two", "target_log10_g": 5.0},
    ]
    predictions = grouped_predictions(rows, [], lambda row: str(row["cell"]))
    by_run = {row["run_id"]: row for row in predictions}
    assert by_run["a"]["prediction_log10_g"] == 4.0
    assert by_run["b"]["prediction_log10_g"] == 1.0
    assert by_run["c"]["prediction_log10_g"] == 1.0


def test_nested_candidate_selection_does_not_use_outer_group() -> None:
    rows = [
        {
            "run_id": f"run-{index}",
            "group": str(index),
            "target_log10_g": float(index),
            "useful": float(index),
            "outer_only": 0.0 if index < 4 else 100.0,
        }
        for index in range(5)
    ]
    predictions = nested_candidate_predictions(
        rows,
        {"useful": ["useful"], "outer_only": ["outer_only"]},
        lambda row: str(row["group"]),
    )
    assert {row["selected_candidate"] for row in predictions} == {"useful"}
