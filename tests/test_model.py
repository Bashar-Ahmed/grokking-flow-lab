from __future__ import annotations

import pytest
import torch

from grokking_lab.model import ModelConfig, Transformer, fourier_fraction, make_dataset


def config(operation: str = "add") -> ModelConfig:
    return ModelConfig(operation, 7, 0.3, 8, 2, 16, 4)


@pytest.mark.parametrize("operation", ["add", "sub", "mul"])
def test_labels(operation: str) -> None:
    data = make_dataset(config(operation))
    for tokens, label in zip(data.tokens.tolist(), data.labels.tolist(), strict=True):
        a, b, equals = tokens
        expected = {
            "add": (a + b) % 7,
            "sub": (a - b) % 7,
            "mul": (a * b) % 7,
        }[operation]
        assert equals == 7
        assert label == expected


def test_multiplication_uses_nonzero_units() -> None:
    data = make_dataset(config("mul"))
    assert data.tokens.shape[0] == 36
    assert int(data.tokens[:, :2].min()) == 1
    assert int(data.labels.min()) == 1


def test_forward_shape_and_cache() -> None:
    model = Transformer(config())
    tokens = make_dataset(config()).tokens[:3]
    logits, cache = model(tokens, return_cache=True)
    assert logits.shape == (3, 3, 8)
    assert cache.head_output.shape == (3, 2, 3, 8)


def perfect_logits(operation: str, p: int = 7) -> torch.Tensor:
    values = range(1, p) if operation == "mul" else range(p)
    rows = []
    for a in values:
        for b in values:
            answer = {
                "add": (a + b) % p,
                "sub": (a - b) % p,
                "mul": (a * b) % p,
            }[operation]
            row = torch.zeros(p + 1)
            row[answer] = 10
            rows.append(row)
    return torch.stack(rows)


@pytest.mark.parametrize("operation", ["add", "sub", "mul"])
def test_operation_matched_fourier_fraction(operation: str) -> None:
    assert fourier_fraction(perfect_logits(operation), 7, operation) > 0.99


def test_split_is_seed_deterministic() -> None:
    left = make_dataset(config())
    right = make_dataset(config())
    assert torch.equal(left.train_indices, right.train_indices)
