"""Unified factory for frontier steering baselines with strict norm matching."""

from typing import Dict, List, Optional, Tuple
import torch

from weight_space.evaluation.measurement.matched_energy import scale_to_target_norm, generate_isotropic_noise_control
from .caa import compute_caa_vector
from .sae_ts import compute_sae_ts_vector
from .sas import compute_sas_vector
from .sign_balanced import decompose_coordinates, build_sign_balanced_vector, build_positive_only_vector, build_magnitude_only_vector


def build_frontier_steering_suite(
    target_sae_dir: torch.Tensor,
    sae_decoder_matrix: torch.Tensor,
    target_latent_idx: int,
    z_pos_mean: torch.Tensor,
    z_neg_mean: torch.Tensor,
    caa_raw_dir: torch.Tensor,
    alpha: float = 35.0,
    k_coords_each: int = 10,
    seed: int = 42,
) -> Tuple[Dict[str, torch.Tensor], float]:
    """Constructs all frontier steering baselines, strictly clamped to target Euclidean energy c.

    The target energy c is defined by the norm of the balanced 20-coordinate recipe (alpha * ||v_bal||₂).

    Baselines included:
    1. CAA: Contrastive Activation Addition (Rimsky et al., 2023)
    2. Dense_SAE: Full unconstrained decoder feature vector
    3. SAE_TS: SAE-Targeted Steering (Chalnev et al., 2024)
    4. SAS: Sparse Activation Steering (Bayat et al., 2025)
    5. Pos_Only: Positive excitation alone
    6. Top_Abs: Magnitude selection without sign balance
    7. NeuronHarness_Bal: Proposed balanced coordinates
    8. Random_Norm_Ctrl: Isotropic Gaussian random noise control (Korznikov et al., 2025)

    Args:
        target_sae_dir: 1D tensor `(d_model,)` decoder column for target latent.
        sae_decoder_matrix: 2D tensor `(d_model, d_sae)` decoder weights.
        target_latent_idx: Target feature index.
        z_pos_mean: Mean latent activation vector for positive exemplars.
        z_neg_mean: Mean latent activation vector for negative exemplars.
        caa_raw_dir: Difference-in-means vector from contrastive pairs.
        alpha: Scaling coefficient multiplier.
        k_coords_each: Number of positive and negative coordinates (default 10 each = 20 total).
        seed: Random seed for controls.

    Returns:
        Tuple of `(dict_of_conditions, target_norm_c)`.
    """
    d_model = target_sae_dir.shape[0]
    pos_idx, neg_idx, abs_idx = decompose_coordinates(target_sae_dir, k_pos=k_coords_each, k_neg=k_coords_each)

    # Reference condition defines the target norm c
    v_bal_raw = build_sign_balanced_vector(target_sae_dir, pos_idx, neg_idx)
    target_norm_c = float(alpha * torch.norm(v_bal_raw, p=2).item())

    # 1. NeuronHarness Balanced (already scaled)
    v_bal = alpha * v_bal_raw

    # 2. CAA
    v_caa = scale_to_target_norm(caa_raw_dir, target_norm_c)

    # 3. Dense SAE decoder
    v_dense = scale_to_target_norm(target_sae_dir.clone(), target_norm_c)

    # 4. SAE-Targeted Steering (Chalnev et al., 2024)
    v_sae_ts_raw = compute_sae_ts_vector(
        target_sae_dir, sae_decoder_matrix, target_latent_idx, top_interfering_latents=20
    )
    v_sae_ts = scale_to_target_norm(v_sae_ts_raw, target_norm_c)

    # 5. Sparse Activation Steering (Bayat et al., 2025)
    v_sas_raw = compute_sas_vector(sae_decoder_matrix, z_pos_mean, z_neg_mean, k_sparse_features=5)
    v_sas = scale_to_target_norm(v_sas_raw, target_norm_c)

    # 6. Positive-Only Coordinates
    v_pos_raw = build_positive_only_vector(target_sae_dir, pos_idx)
    v_pos = scale_to_target_norm(v_pos_raw, target_norm_c)

    # 7. Magnitude-Only (Top|w| without sign balance)
    v_abs_raw = build_magnitude_only_vector(target_sae_dir, abs_idx)
    v_abs = scale_to_target_norm(v_abs_raw, target_norm_c)

    # 8. Isotropic Gaussian Random Control (Korznikov et al., 2025)
    v_noise = generate_isotropic_noise_control(
        d_model, target_norm_c, seed=seed, device=target_sae_dir.device, dtype=target_sae_dir.dtype
    )

    suite = {
        "CAA": v_caa,
        "Dense_SAE": v_dense,
        "SAE_TS": v_sae_ts,
        "SAS": v_sas,
        "Pos_Only": v_pos,
        "Top_Abs": v_abs,
        "NeuronHarness_Bal": v_bal,
        "Random_Norm_Ctrl": v_noise,
    }

    return suite, target_norm_c
