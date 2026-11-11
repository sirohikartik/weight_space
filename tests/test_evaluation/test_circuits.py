"""Unit tests for multi-layer circuit steering and dynamic token scheduling."""

import pytest
import torch

from weight_space.evaluation.circuits.token_scheduler import DynamicTokenScheduler
from weight_space.evaluation.circuits.layer_sweeper import (
    calculate_sweep_layers,
    distribute_circuit_energy,
    identify_optimal_layer,
    LayerSweepResult,
)


def test_token_scheduler_constant():
    scheduler = DynamicTokenScheduler(strategy="constant", alpha_0=30.0)
    for step in [0, 5, 20, 50]:
        assert scheduler.get_scale(step) == 30.0


def test_token_scheduler_linear_decay():
    decay_steps = 10
    scheduler = DynamicTokenScheduler(strategy="linear_decay", alpha_0=20.0, decay_steps=decay_steps)
    assert scheduler.get_scale(0) == 20.0
    assert pytest.approx(scheduler.get_scale(5), rel=1e-5) == 10.0
    assert scheduler.get_scale(10) == 0.0
    assert scheduler.get_scale(15) == 0.0

    curve = scheduler.generate_schedule_curve(num_tokens=11)
    assert len(curve) == 11
    assert curve[0] == 20.0
    assert curve[-1] == 0.0


def test_token_scheduler_exponential_decay():
    scheduler = DynamicTokenScheduler(strategy="exponential_decay", alpha_0=40.0, gamma=0.1)
    assert scheduler.get_scale(0) == 40.0
    s1 = scheduler.get_scale(1)
    s2 = scheduler.get_scale(2)
    assert s1 > s2 > 0.0


def test_calculate_sweep_layers():
    # 28 total layers, L/4 = 7, 3L/4 = 21, step = 2
    layers = calculate_sweep_layers(total_layers=28, min_frac=0.25, max_frac=0.75, step=2)
    assert layers[0] == 7
    assert layers[-1] == 21
    assert all(7 <= l <= 21 for l in layers)


def test_distribute_circuit_energy():
    torch.manual_seed(42)
    d_model = 32
    target_c = 10.0
    circuit_vectors = {
        10: torch.randn(d_model),
        12: torch.randn(d_model),
        14: torch.randn(d_model),
    }

    scaled_circuit = distribute_circuit_energy(
        vectors_by_layer=circuit_vectors,
        target_total_norm=target_c,
    )

    # Compute joint Euclidean energy: sqrt(sum_l ||Delta x_l||_2^2)
    sum_sq_norms = sum(torch.sum(v ** 2).item() for v in scaled_circuit.values())
    joint_norm = sum_sq_norms ** 0.5
    assert pytest.approx(joint_norm, rel=1e-4) == target_c


def test_identify_optimal_layer():
    results = [
        LayerSweepResult(layer_idx=10, adherence_pct=65.0, adherence_ci=(60, 70), collapse_pct=5.0, collapse_ci=(2, 8), corpus_perplexity=2.01, refusal_overlap_cosine=0.01),
        LayerSweepResult(layer_idx=14, adherence_pct=88.0, adherence_ci=(83, 92), collapse_pct=0.0, collapse_ci=(0, 3.7), corpus_perplexity=1.99, refusal_overlap_cosine=0.02),
        LayerSweepResult(layer_idx=18, adherence_pct=72.0, adherence_ci=(67, 76), collapse_pct=8.0, collapse_ci=(4, 12), corpus_perplexity=2.05, refusal_overlap_cosine=0.05),
    ]

    best = identify_optimal_layer(results)
    assert best is not None
    assert best.layer_idx == 14
