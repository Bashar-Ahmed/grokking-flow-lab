"""The model, modular datasets, and behavior-only measurements."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

OPERATIONS = ("add", "sub", "mul")


@dataclass(frozen=True)
class ModelConfig:
    operation: str
    p: int
    train_fraction: float
    d_model: int
    num_heads: int
    d_mlp: int
    seed: int

    def __post_init__(self) -> None:
        if self.operation not in OPERATIONS:
            raise ValueError(f"operation must be one of {OPERATIONS}")
        if self.p < 3:
            raise ValueError("p must be at least 3")
        if not 0 < self.train_fraction < 1:
            raise ValueError("train_fraction must be strictly between 0 and 1")
        if min(self.d_model, self.num_heads, self.d_mlp) <= 0:
            raise ValueError("model dimensions must be positive")
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")

    @property
    def d_head(self) -> int:
        return self.d_model // self.num_heads

    @property
    def vocabulary_size(self) -> int:
        return self.p + 1


@dataclass(frozen=True)
class Dataset:
    tokens: Tensor
    labels: Tensor
    train_indices: Tensor
    test_indices: Tensor

    def split(self, name: str) -> tuple[Tensor, Tensor]:
        indices = self.train_indices if name == "train" else self.test_indices
        return self.tokens[indices], self.labels[indices]

    def to(self, device: torch.device | str) -> Dataset:
        return Dataset(*(value.to(device) for value in self.__dict__.values()))


@dataclass(frozen=True)
class ForwardCache:
    token_embedding: Tensor
    position_embedding: Tensor
    residual_pre: Tensor
    attention: Tensor
    head_output: Tensor
    residual_mid: Tensor
    mlp_post: Tensor


class Transformer(nn.Module):
    """One causal block, no layer norm, and an untied unembedding."""

    sequence_length = 3

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        scale = math.sqrt(config.d_model)
        h, r, d = config.num_heads, config.d_head, config.d_model

        self.W_E = nn.Parameter(torch.randn(d, config.vocabulary_size) / scale)
        self.W_pos = nn.Parameter(torch.randn(self.sequence_length, d) / scale)
        self.W_Q = nn.Parameter(torch.randn(h, r, d) / scale)
        self.W_K = nn.Parameter(torch.randn(h, r, d) / scale)
        self.W_V = nn.Parameter(torch.randn(h, r, d) / scale)
        self.W_O = nn.Parameter(torch.randn(d, h, r) / scale)
        self.W_in = nn.Parameter(torch.randn(config.d_mlp, d) / scale)
        self.b_in = nn.Parameter(torch.zeros(config.d_mlp))
        self.W_out = nn.Parameter(torch.randn(d, config.d_mlp) / scale)
        self.b_out = nn.Parameter(torch.zeros(d))
        self.W_U = nn.Parameter(
            torch.randn(d, config.vocabulary_size) / math.sqrt(config.vocabulary_size)
        )
        self.register_buffer("causal_mask", torch.tril(torch.ones(3, 3, dtype=torch.bool)))

    def forward(
        self, tokens: Tensor, *, return_cache: bool = False
    ) -> Tensor | tuple[Tensor, ForwardCache]:
        if tokens.ndim != 2 or tokens.shape[1] != self.sequence_length:
            raise ValueError("tokens must have shape [batch, 3]")

        token_embedding = F.embedding(tokens, self.W_E.T)
        position_embedding = self.W_pos.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        residual_pre = token_embedding + position_embedding

        keys = torch.einsum("hrd,bpd->bhpr", self.W_K, residual_pre)
        queries = torch.einsum("hrd,bpd->bhpr", self.W_Q, residual_pre)
        values = torch.einsum("hrd,bpd->bhpr", self.W_V, residual_pre)
        scores = torch.einsum("bhpr,bhqr->bhqp", keys, queries) / math.sqrt(self.config.d_head)
        scores = scores.masked_fill(~self.causal_mask[None, None], -1e10)
        attention = scores.softmax(dim=-1)
        mixed_values = torch.einsum("bhpr,bhqp->bhqr", values, attention)
        head_output = torch.einsum("dhr,bhqr->bhqd", self.W_O, mixed_values)
        residual_mid = residual_pre + head_output.sum(dim=1)

        mlp_pre = torch.einsum("md,bpd->bpm", self.W_in, residual_mid) + self.b_in
        mlp_post = F.relu(mlp_pre)
        mlp_output = torch.einsum("dm,bpm->bpd", self.W_out, mlp_post) + self.b_out
        logits = torch.einsum("bpd,dv->bpv", residual_mid + mlp_output, self.W_U)
        if not return_cache:
            return logits
        return logits, ForwardCache(
            token_embedding=token_embedding,
            position_embedding=position_embedding,
            residual_pre=residual_pre,
            attention=attention,
            head_output=head_output,
            residual_mid=residual_mid,
            mlp_post=mlp_post,
        )


def make_dataset(config: ModelConfig, device: torch.device | str = "cpu") -> Dataset:
    """Enumerate every ordered pair and make a seed-fixed train/test split."""

    values = range(1, config.p) if config.operation == "mul" else range(config.p)
    pairs = [(a, b) for a in values for b in values]
    tokens = torch.tensor([(a, b, config.p) for a, b in pairs], dtype=torch.long)
    if config.operation == "add":
        answers = [(a + b) % config.p for a, b in pairs]
    elif config.operation == "sub":
        answers = [(a - b) % config.p for a, b in pairs]
    else:
        answers = [(a * b) % config.p for a, b in pairs]
    labels = torch.tensor(answers, dtype=torch.long)

    indices = list(range(len(pairs)))
    random.Random(config.seed).shuffle(indices)
    split_at = int(config.train_fraction * len(indices))
    dataset = Dataset(
        tokens=tokens,
        labels=labels,
        train_indices=torch.tensor(indices[:split_at], dtype=torch.long),
        test_indices=torch.tensor(indices[split_at:], dtype=torch.long),
    )
    return dataset.to(device)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


@torch.inference_mode()
def metrics(
    model: Transformer, tokens: Tensor, labels: Tensor, batch_size: int
) -> dict[str, float]:
    total_loss = 0.0
    total_correct = 0
    for start in range(0, len(tokens), batch_size):
        logits = model(tokens[start : start + batch_size])
        assert isinstance(logits, Tensor)
        logits = logits[:, -1]
        batch_labels = labels[start : start + batch_size]
        total_loss += float(F.cross_entropy(logits, batch_labels, reduction="sum"))
        total_correct += int((logits.argmax(-1) == batch_labels).sum())
    return {"loss": total_loss / len(tokens), "accuracy": total_correct / len(tokens)}


@torch.inference_mode()
def all_logits(model: Transformer, tokens: Tensor, batch_size: int) -> Tensor:
    outputs = []
    for start in range(0, len(tokens), batch_size):
        logits = model(tokens[start : start + batch_size])
        assert isinstance(logits, Tensor)
        outputs.append(logits[:, -1].cpu())
    return torch.cat(outputs)


def primitive_root(p: int) -> int:
    factors: set[int] = set()
    remainder, divisor = p - 1, 2
    while divisor * divisor <= remainder:
        while remainder % divisor == 0:
            factors.add(divisor)
            remainder //= divisor
        divisor += 1
    if remainder > 1:
        factors.add(remainder)
    for candidate in range(2, p):
        if all(pow(candidate, (p - 1) // factor, p) != 1 for factor in factors):
            return candidate
    raise ValueError(f"{p} has no primitive root")


def fourier_fraction(logits: Tensor, p: int, operation: str) -> float:
    """Operation-matched share of nonconstant 3D Fourier power."""

    if operation in ("add", "sub"):
        if logits.shape[0] != p * p:
            raise ValueError("logit row count does not match p^2")
        cube = logits[:, :p].double().reshape(p, p, p)
        period = p
    else:
        period = p - 1
        if logits.shape[0] != period * period:
            raise ValueError("multiplication logit row count does not match (p-1)^2")
        root = primitive_root(p)
        powers = [pow(root, exponent, p) for exponent in range(period)]
        row_of = {value: index for index, value in enumerate(range(1, p))}
        rows = [
            row_of[powers[a]] * period + row_of[powers[b]]
            for a in range(period)
            for b in range(period)
        ]
        cube = logits.double()[rows][:, powers].reshape(period, period, period)

    power = torch.fft.fftn(cube).abs().square()
    power[0, 0, 0] = 0
    denominator = float(power.sum())
    if denominator == 0:
        return 0.0
    if operation == "sub":
        numerator = sum(float(power[-k % period, k, k]) for k in range(1, period))
    else:
        numerator = sum(float(power[k, k, -k % period]) for k in range(1, period))
    return numerator / denominator


def attention_input_parts(model: Transformer, cache: ForwardCache) -> Tensor:
    """Per-head, per-source-position token/position value-path outputs."""

    parts = torch.stack((cache.token_embedding, cache.position_embedding), dim=2)
    values = torch.einsum("hrd,bpkd->bhpkr", model.W_V, parts)
    weighted = values * cache.attention[:, :, -1, :, None, None]
    return torch.einsum("dhr,bhpkr->bhpkd", model.W_O, weighted)


def parameter_count(config: ModelConfig) -> int:
    seed_everything(config.seed)
    return sum(parameter.numel() for parameter in Transformer(config).parameters())
