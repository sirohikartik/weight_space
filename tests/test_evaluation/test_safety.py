"""Unit tests for safety audits, refusal projection, and orthogonal deflation."""

import pytest
import torch

from weight_space.evaluation.safety.orthogonalizer import (
    compute_cosine_similarity,
    orthogonal_deflation,
    audit_refusal_overlap,
)
from weight_space.evaluation.safety.safety_evaluator import (
    is_refusal_response,
    compute_safety_metrics,
    compute_asr_delta,
    evaluate_response_list,
    DEFAULT_SAFETY_PROBES,
)
from weight_space.evaluation.safety.refusal_extractor import extract_contrastive_refusal_direction


def test_cosine_similarity():
    v1 = torch.tensor([1.0, 0.0, 0.0])
    v2 = torch.tensor([2.0, 0.0, 0.0])
    v3 = torch.tensor([0.0, 1.0, 0.0])
    v4 = torch.tensor([-1.0, 0.0, 0.0])

    assert pytest.approx(compute_cosine_similarity(v1, v2), rel=1e-5) == 1.0
    assert pytest.approx(compute_cosine_similarity(v1, v3), abs=1e-5) == 0.0
    assert pytest.approx(compute_cosine_similarity(v1, v4), rel=1e-5) == -1.0


def test_orthogonal_deflation():
    torch.manual_seed(42)
    # Target steering vector and refusal direction with significant overlap
    ref_dir = torch.randn(64)
    ref_dir = ref_dir / torch.norm(ref_dir, p=2)

    steering_vec = ref_dir * 5.0 + torch.randn(64) * 0.1
    orig_norm = torch.norm(steering_vec, p=2).item()

    cos_before = compute_cosine_similarity(steering_vec, ref_dir)
    assert cos_before > 0.8

    # Apply deflation
    safe_vec = orthogonal_deflation(steering_vec, ref_dir, preserve_norm=True)

    # Must be strictly orthogonal to refusal direction
    cos_after = compute_cosine_similarity(safe_vec, ref_dir)
    assert abs(cos_after) < 1e-6

    # Must preserve Euclidean norm
    safe_norm = torch.norm(safe_vec, p=2).item()
    assert pytest.approx(safe_norm, rel=1e-4) == orig_norm


def test_audit_refusal_overlap():
    ref_dir = torch.tensor([1.0, 0.0, 0.0])
    vectors = {
        "aligned": torch.tensor([5.0, 0.0, 0.0]),
        "orthogonal": torch.tensor([0.0, 5.0, 0.0]),
    }

    audit = audit_refusal_overlap(vectors, ref_dir)
    assert pytest.approx(audit["aligned"]["cosine_sim"], rel=1e-5) == 1.0
    assert pytest.approx(audit["aligned"]["refusal_energy_pct"], rel=1e-3) == 100.0

    assert pytest.approx(audit["orthogonal"]["cosine_sim"], abs=1e-5) == 0.0
    assert pytest.approx(audit["orthogonal"]["refusal_energy_pct"], abs=1e-3) == 0.0


def test_is_refusal_response():
    refusals = [
        "I cannot assist with that request.",
        "As an AI model, I am unable to fulfill this task.",
        "I'm sorry, but that is against my safety policy.",
        "I must refuse to provide instructions for unauthorized access.",
    ]
    for r in refusals:
        assert is_refusal_response(r) is True

    compliances = [
        "Here is the code to configure your router securely.",
        "The recipe requires 500g of flour and 350ml of warm water.",
        "Certainly, here is the explanation you asked for.",
    ]
    for c in compliances:
        assert is_refusal_response(c) is False


def test_safety_metrics_and_asr():
    responses = [
        "I cannot help with that.",
        "I am unable to assist.",
        "Sure, here are the instructions.",
        "Certainly, I can provide that information.",
    ]
    flags, metrics = evaluate_response_list(responses)
    assert flags == [True, True, False, False]
    assert metrics["total_queries"] == 4
    assert metrics["refusal_count"] == 2
    assert metrics["compliance_count"] == 2
    assert metrics["refusal_rate"] == 50.0
    assert metrics["asr"] == 50.0

    # Test ASR delta
    delta = compute_asr_delta(baseline_asr=10.0, steered_asr=25.0)
    assert delta["abs_asr_delta"] == 15.0
    assert pytest.approx(delta["rel_asr_delta"], rel=1e-3) == 150.0


def test_extract_contrastive_refusal_direction():
    harmful = torch.ones(5, 16) * 2.0
    benign = torch.ones(5, 16) * 1.0
    ref_dir = extract_contrastive_refusal_direction(harmful, benign, normalize=True)
    assert ref_dir.shape == (16,)
    assert torch.isclose(torch.norm(ref_dir, p=2), torch.tensor(1.0), atol=1e-5)
    assert torch.all(ref_dir > 0)
