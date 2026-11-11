"""Unit tests for frontier baseline algorithms and unified suite construction."""

import pytest
import torch

from weight_space.evaluation.frontier.caa import compute_caa_vector_from_activations
from weight_space.evaluation.frontier.sae_ts import compute_sae_ts_vector
from weight_space.evaluation.frontier.sas import compute_sas_vector
from weight_space.evaluation.frontier.sign_balanced import (
    decompose_coordinates,
    build_sign_balanced_vector,
    build_positive_only_vector,
    build_negative_only_vector,
    build_magnitude_only_vector,
)
from weight_space.evaluation.frontier.frontier_suite import build_frontier_steering_suite


def test_caa_vector_computation():
    torch.manual_seed(42)
    # 5 pairs, seq_len 4, d_model 16
    pos_acts = torch.randn(5, 4, 16) + 1.0
    neg_acts = torch.randn(5, 4, 16) - 1.0

    caa_vec = compute_caa_vector_from_activations(pos_acts, neg_acts, normalize=True)
    assert caa_vec.shape == (16,)
    assert torch.isclose(torch.norm(caa_vec, p=2), torch.tensor(1.0), atol=1e-5)
    # The mean difference should be positive in direction
    assert torch.mean(caa_vec).item() > 0


def test_sign_balanced_decomposition():
    # Construct known vector with 5 pos and 5 neg values
    v = torch.tensor([10.0, -10.0, 5.0, -5.0, 1.0, -1.0, 0.1, -0.1, 0.0, 0.0])
    top_pos, top_neg, top_abs = decompose_coordinates(v, k_pos=2, k_neg=2)

    assert top_pos == [0, 2]  # 10.0, 5.0
    assert top_neg == [1, 3]  # -10.0, -5.0
    assert set(top_abs) == {0, 1, 2, 3}

    v_bal = build_sign_balanced_vector(v, top_pos, top_neg)
    assert v_bal[0] == 10.0
    assert v_bal[1] == -10.0
    assert v_bal[2] == 5.0
    assert v_bal[3] == -5.0
    assert v_bal[4] == 0.0

    v_pos = build_positive_only_vector(v, top_pos)
    assert v_pos[0] == 10.0
    assert v_pos[1] == 0.0
    assert v_pos[2] == 5.0

    v_neg = build_negative_only_vector(v, top_neg)
    assert v_neg[1] == -10.0
    assert v_neg[3] == -5.0
    assert v_neg[0] == 0.0

    v_abs = build_magnitude_only_vector(v, top_abs)
    assert torch.equal(v_abs, v_bal)


def test_sas_vector_computation():
    torch.manual_seed(42)
    d_model = 32
    d_sae = 64
    sae_decoder = torch.randn(d_model, d_sae)
    z_pos = torch.zeros(d_sae)
    z_neg = torch.zeros(d_sae)
    z_pos[10] = 5.0
    z_neg[20] = 3.0

    sas_vec = compute_sas_vector(sae_decoder, z_pos, z_neg, k_sparse_features=2)
    assert sas_vec.shape == (d_model,)
    assert torch.norm(sas_vec, p=2).item() > 0


def test_sae_ts_optimization():
    torch.manual_seed(42)
    d_model = 32
    d_sae = 20
    sae_decoder = torch.randn(d_model, d_sae)
    sae_decoder = sae_decoder / torch.norm(sae_decoder, dim=0, keepdim=True)
    target_idx = 5
    target_dir = sae_decoder[:, target_idx]

    sae_ts_vec = compute_sae_ts_vector(
        target_feature_dir=target_dir,
        sae_decoder_weights=sae_decoder,
        target_latent_idx=target_idx,
        reg_lambda=0.5,
        max_iter=25,
        lr=0.05,
    )
    assert sae_ts_vec.shape == (d_model,)
    # Should maintain high cosine similarity with target
    cos_with_target = torch.dot(sae_ts_vec / torch.norm(sae_ts_vec), target_dir).item()
    assert cos_with_target > 0.8


def test_build_frontier_steering_suite_energy_conservation():
    torch.manual_seed(42)
    d_model = 64
    d_sae = 128
    sae_decoder = torch.randn(d_model, d_sae)
    sae_decoder = sae_decoder / torch.norm(sae_decoder, dim=0, keepdim=True)
    target_latent = 12
    target_dir = sae_decoder[:, target_latent]

    z_pos = torch.rand(d_sae)
    z_neg = torch.rand(d_sae)
    caa_dir = torch.randn(d_model)

    conditions, target_c = build_frontier_steering_suite(
        target_sae_dir=target_dir,
        sae_decoder_matrix=sae_decoder,
        target_latent_idx=target_latent,
        z_pos_mean=z_pos,
        z_neg_mean=z_neg,
        caa_raw_dir=caa_dir,
        alpha=20.0,
        k_coords_each=5,
    )

    expected_baselines = [
        "CAA",
        "Dense_SAE",
        "SAE_TS",
        "SAS",
        "Pos_Only",
        "Top_Abs",
        "NeuronHarness_Bal",
        "Random_Norm_Ctrl",
    ]
    for b in expected_baselines:
        assert b in conditions
        norm = torch.norm(conditions[b], p=2).item()
        # Verify strict Euclidean norm matching
        assert pytest.approx(norm, rel=1e-4) == target_c
