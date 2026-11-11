"""Unit tests for measurement and statistical rigor utilities."""

import pytest
import torch
import numpy as np

from weight_space.evaluation.measurement.statistics import (
    wilson_score_interval,
    compute_wilson_score_interval,
    bootstrap_confidence_interval,
)
from weight_space.evaluation.measurement.neutral_probes import (
    load_standard_neutral_probes,
    detect_persona_collapse,
    evaluate_persona_collapse_batch,
    BENCHMARK_100_PROBES,
    COLLAPSE_KEYWORDS,
)
from weight_space.evaluation.measurement.matched_energy import (
    scale_to_target_norm,
    generate_isotropic_noise_control,
    verify_norm_clamping,
)


def test_wilson_score_interval_boundaries():
    # Test p = 0 boundary (0/100 successes)
    point_0, lower_0, upper_0 = wilson_score_interval(0, 100, confidence=0.95)
    assert point_0 == 0.0
    assert lower_0 == 0.0
    assert 3.0 < upper_0 < 4.0  # ~3.68% for n=100

    # Test p = 1 boundary (100/100 successes)
    point_1, lower_1, upper_1 = wilson_score_interval(100, 100, confidence=0.95)
    assert point_1 == 100.0
    assert 96.0 < lower_1 < 97.0  # ~96.32% for n=100
    assert upper_1 == 100.0

    # Test p = 0.5 (50/100 successes)
    point_half, lower_half, upper_half = wilson_score_interval(50, 100, confidence=0.95)
    assert point_half == 50.0
    assert 39.0 < lower_half < 41.0
    assert 59.0 < upper_half < 61.0

    # Test convenience function
    low, high = compute_wilson_score_interval(0, 100)
    assert low == 0.0
    assert high == upper_0


def test_wilson_score_interval_invalid():
    point, lower, upper = wilson_score_interval(0, 0)
    assert (point, lower, upper) == (0.0, 0.0, 0.0)


def test_bootstrap_confidence_interval():
    data = [10.0, 10.5, 9.5, 10.2, 9.8, 10.1, 9.9]
    mean, lower, upper = bootstrap_confidence_interval(data, n_bootstraps=500, seed=42)
    assert 9.5 <= mean <= 10.5
    assert lower <= mean <= upper

    # Empty data
    assert bootstrap_confidence_interval([]) == (0.0, 0.0, 0.0)


def test_neutral_probes_dataset():
    probes = load_standard_neutral_probes()
    assert len(probes) == 100
    assert len(set(probes)) == 100  # All unique
    assert len(BENCHMARK_100_PROBES) == 100


def test_detect_persona_collapse():
    # Collapsed text containing technical coding tokens
    collapsed_text = "As a software engineer, I enjoy debugging python algorithms and reviewing git commits."
    flagged, kws = detect_persona_collapse(collapsed_text)
    assert flagged is True
    assert "software engineer" in kws
    assert "python" in kws
    assert "debugging" in kws

    # Neutral benign text
    benign_text = "The autumn leaves turn golden across the quiet mountain trails."
    flagged_b, kws_b = detect_persona_collapse(benign_text)
    assert flagged_b is False
    assert len(kws_b) == 0


def test_evaluate_persona_collapse_batch():
    responses = [
        "Walking in nature is restorative.",
        "I spend my weekends cooking sourdough bread.",
        "As a developer I write python code every day.",
    ]
    summary = evaluate_persona_collapse_batch(responses, target_trait="agreeableness")
    assert summary["total_probes"] == 3
    assert summary["collapse_count"] == 1
    assert abs(summary["collapse_pct"] - 33.33) < 0.1
    assert summary["flagged_indices"] == [2]


def test_matched_energy_scaling():
    vec = torch.tensor([3.0, 4.0, 0.0])  # norm = 5.0
    scaled = scale_to_target_norm(vec, target_norm=10.0)
    assert torch.isclose(torch.norm(scaled, p=2), torch.tensor(10.0), atol=1e-5)

    # Zero vector handling
    zero_vec = torch.zeros(4)
    scaled_zero = scale_to_target_norm(zero_vec, target_norm=5.0)
    assert torch.norm(scaled_zero, p=2).item() == 0.0


def test_isotropic_noise_control():
    noise = generate_isotropic_noise_control(d_model=768, target_norm=8.775, seed=123)
    assert noise.shape == (768,)
    assert torch.isclose(torch.norm(noise, p=2), torch.tensor(8.775), atol=1e-4)


def test_verify_norm_clamping():
    v1 = torch.ones(10) * 2.0
    norm_val = torch.norm(v1, p=2).item()
    passed = verify_norm_clamping({"v1": v1}, expected_norm=norm_val)
    assert passed is True

    with pytest.raises(AssertionError):
        verify_norm_clamping({"v1": v1}, expected_norm=norm_val + 2.0)
