"""Sparse Activation Steering (SAS) baseline (Bayat et al., 2025).

Implements sparse-space steering by identifying differential feature activations in SAE latent
space and projecting only the top positive and contrastive feature directions into residual space.
"""

from typing import Dict, List, Optional, Tuple
import torch


def compute_sas_vector(
    sae_decoder_weights: torch.Tensor,
    z_pos_mean: torch.Tensor,
    z_neg_mean: torch.Tensor,
    k_sparse_features: int = 5,
) -> torch.Tensor:
    """Computes Sparse Activation Steering (SAS) vector from differential SAE latents.

    Delta_x_SAS = sum_{i in TopK(z_diff)} (z_pos_i - z_neg_i) * w_i

    Args:
        sae_decoder_weights: 2D tensor `(d_model, d_sae)` containing unit-norm decoder columns.
        z_pos_mean: 1D tensor `(d_sae,)` mean latent activation on positive exemplars.
        z_neg_mean: 1D tensor `(d_sae,)` mean latent activation on negative exemplars.
        k_sparse_features: Number of top differential latent features to include.

    Returns:
        1D PyTorch tensor `(d_model,)` corresponding to sparse dictionary steering.
    """
    device = sae_decoder_weights.device
    dtype = sae_decoder_weights.dtype
    d_model, d_sae = sae_decoder_weights.shape

    z_diff = z_pos_mean - z_neg_mean
    top_latent_indices = torch.topk(z_diff, k=min(k_sparse_features, d_sae)).indices

    sas_vec = torch.zeros(d_model, device=device, dtype=dtype)
    for idx in top_latent_indices:
        coeff = z_diff[idx].item()
        w_i = sae_decoder_weights[:, idx]
        sas_vec += coeff * w_i

    return sas_vec
