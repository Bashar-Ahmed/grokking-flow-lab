from __future__ import annotations

import pytest

from grokking_lab.flow import FLOW_KINDS, DegenerateFlowError, build_raw_flow
from grokking_lab.model import ModelConfig, Transformer, make_dataset, seed_everything


@pytest.mark.parametrize("kind", FLOW_KINDS)
def test_raw_flow_is_unit_normalized_and_conserved(kind: str) -> None:
    model_config = ModelConfig("add", 7, 0.3, 8, 2, 16, 2)
    seed_everything(model_config.seed)
    model = Transformer(model_config)
    data = make_dataset(model_config)
    try:
        flow = build_raw_flow(model, data.tokens[0], int(data.labels[0]), kind)
    except DegenerateFlowError:
        pytest.skip("this random initialization has no positive mass for the requested polarity")
    assert sum(weight for _, weight in flow.paths) == pytest.approx(1.0, abs=1e-7)
    assert flow.conservation_error() < 1e-7
    assert all(value >= 0 for _, _, value in flow.edges)
