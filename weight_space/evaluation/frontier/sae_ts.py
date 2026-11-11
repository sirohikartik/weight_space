"""SAE-Targeted Steering (SAE-TS) baseline (Chalnev et al., 2024).

Optimizes / filters steering coordinates to maximize projection along the target SAE feature
while explicitly minimizing interference with collateral off-target dictionary features:
min_v ||v - w_k||_2^2 + lambda * sum_{m != k} |<v, w_m>|^2
"""

from typing import List, Optional
import torch


def compute_sae_ts_vector(
    target_feature_dir: torch.Tensor,
    sae_decoder_weights: torch.Tensor,
    target_latent_idx: int,
    top_interfering_latents: int = 20,
    reg_lambda: float = 0.5,
    max_iter: int = 100,
    lr: float = 1e-2,
) -> torch.Tensor:
    """Computes an SAE-Targeted Steering (SAE-TS) vector that suppresses collateral features.

    Args:
        target_feature_dir: 1D tensor `(d_model,)` corresponding to target latent k.
        sae_decoder_weights: 2D tensor `(d_model, d_sae)` of unit-norm decoder columns.
        target_latent_idx: Integer index k of the target feature.
        top_interfering_latents: Number of top interfering collateral features to penalize.
        reg_lambda: Regularization penalty strength for off-target overlap.
        max_iter: Number of optimization gradient steps.
        lr: Optimization learning rate.

    Returns:
        1D PyTorch tensor `(d_model,)` optimized for targeted feature alignment.
    """
    device = target_feature_dir.device
    dtype = target_feature_dir.dtype
    d_model, d_sae = sae_decoder_weights.shape

    # Identify top interfering latent features that have highest cosine similarity with target
    with torch.no_grad():
        w_k_norm = target_feature_dir / (torch.norm(target_feature_dir, p=2) + 1e-12)
        overlaps = torch.abs(torch.matmul(w_k_norm, sae_decoder_weights))  # shape (d_sae,)
        overlaps[target_latent_idx] = -1.0  # exclude target itself
        interfering_indices = torch.topk(overlaps, k=min(top_interfering_latents, d_sae - 1)).indices
        interfering_cols = sae_decoder_weights[:, interfering_indices].detach()  # (d_model, k_interf)

    # Initialize v with target feature direction
    v = target_feature_dir.clone().detach().requires_grad_(True)
    optimizer = torch.optim.AdamW([v], lr=lr)

    for _ in range(max_iter):
        optimizer.zero_grad()
        # Fidelity loss: stay aligned with target feature
        fidelity_loss = torch.norm(v - target_feature_dir, p=2) ** 2
        # Collateral penalty: minimize dot product with top interfering features
        collateral_projections = torch.matmul(v, interfering_cols)  # (k_interf,)
        collateral_loss = torch.sum(collateral_projections ** 2)

        total_loss = fidelity_loss + reg_lambda * collateral_loss
        total_loss.backward()
        optimizer.step()

    return v.detach().to(dtype=dtype, device=device)
