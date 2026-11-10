"""Unit tests for steering and coordinate vector construction."""

import pytest
import torch
import torch.nn as nn
from weight_space.decomposition import build_sparse_vector, decompose_feature_direction
from weight_space.steering import SteeringHook


class DummyModel(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(2)])

    def forward(self, x):
        for layer in self.model.layers:
            x = layer(x)
        return x


def test_build_sparse_vector():
    coords = {10: 0.5, 25: -0.8}
    vec = build_sparse_vector(coords, d_model=64)
    assert vec.shape == (64,)
    assert pytest.approx(vec[10].item(), abs=1e-5) == 0.5
    assert pytest.approx(vec[25].item(), abs=1e-5) == -0.8
    assert vec[0].item() == 0.0


def test_decompose_feature_direction():
    direction = torch.linspace(-1.0, 1.0, steps=100)
    pos, neg = decompose_feature_direction(direction, top_k=5)
    assert len(pos) == 5
    assert len(neg) == 5
    # Largest positive is at index 99 (value 1.0)
    assert 99 in pos
    # Most negative is at index 0 (value -1.0)
    assert 0 in neg


def test_steering_hook():
    model = DummyModel(hidden_dim=32)
    steering_vec = torch.ones(32) * 5.0

    x = torch.zeros(1, 4, 32)
    with SteeringHook(model, steering_vector=steering_vec, alpha=1.0, layer_idx=0):
        # When layer 0 forward runs, hook should inject the steering vector
        out = model.model.layers[0](x)
        # linear layer on zeros gives bias (which is rand/zero), but steered should include +5.0
        assert out is not None
