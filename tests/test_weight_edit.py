"""Unit tests for closed-form weight-space editing and ablation."""

import pytest
import torch
import torch.nn as nn
from weight_space.weight_edit import (
    WeightEditContext,
    apply_orthogonal_ablation,
    apply_rank1_edit,
)


def test_apply_rank1_edit():
    d_out, d_in = 16, 32
    linear = nn.Linear(d_in, d_out, bias=False)
    orig_w = linear.weight.data.clone()

    direction = torch.randn(d_out)
    delta_W = apply_rank1_edit(linear, direction=direction, alpha=2.0)

    assert delta_W.shape == (d_out, d_in)
    # The rank of delta_W should be exactly 1
    u, s, v = torch.svd(delta_W)
    assert s[0] > 1e-4
    assert torch.all(s[1:] < 1e-4)

    # Verify weight changed
    assert not torch.allclose(linear.weight.data, orig_w)


def test_orthogonal_ablation():
    d_out, d_in = 8, 12
    linear = nn.Linear(d_in, d_out, bias=False)

    direction = torch.zeros(d_out)
    direction[0] = 1.0  # standard basis along axis 0

    apply_orthogonal_ablation(linear, direction)

    # After ablated along axis 0, row 0 of linear.weight must be identically 0
    assert torch.allclose(linear.weight.data[0, :], torch.zeros(d_in), atol=1e-6)


def test_weight_edit_context_rollback():
    linear = nn.Linear(8, 8, bias=False)
    orig_w = linear.weight.data.clone()

    with WeightEditContext(linear):
        linear.weight.data.add_(5.0)
        assert not torch.allclose(linear.weight.data, orig_w)

    # Must be restored after context exit
    assert torch.allclose(linear.weight.data, orig_w)
