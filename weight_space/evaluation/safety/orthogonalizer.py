"""Subspace projection and orthogonal deflation against the model refusal direction."""

from typing import Dict, Tuple
import torch


def compute_cosine_similarity(v1: torch.Tensor, v2: torch.Tensor, eps: float = 1e-8) -> float:
    """Computes cosine similarity between two 1D vectors."""
    v1_flat = v1.flatten().float()
    v2_flat = v2.flatten().float()
    norm1 = torch.norm(v1_flat, p=2)
    norm2 = torch.norm(v2_flat, p=2)
    if norm1 < eps or norm2 < eps:
        return 0.0
    cos = torch.dot(v1_flat, v2_flat) / (norm1 * norm2)
    return float(cos.item())


def orthogonal_deflation(
    steering_vector: torch.Tensor,
    refusal_direction: torch.Tensor,
    preserve_norm: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Deflates a steering vector by removing its projection along the refusal subspace.

    Formula:
        v_proj = (<Delta_x, v_refusal> / ||v_refusal||_2^2) * v_refusal
        Delta_x_safe = Delta_x - v_proj

    If preserve_norm is True:
        Delta_x_safe = ||Delta_x||_2 * (Delta_x_safe / ||Delta_x_safe||_2)

    Args:
        steering_vector: 1D tensor `(d_model,)` representing raw steering vector.
        refusal_direction: 1D tensor `(d_model,)` representing refusal direction.
        preserve_norm: Whether to re-scale the deflated vector to match original norm.
        eps: Small epsilon to avoid division by zero.

    Returns:
        Deflated steering vector `(d_model,)` orthogonal to refusal_direction.
    """
    orig_norm = torch.norm(steering_vector, p=2)
    ref_norm_sq = torch.sum(refusal_direction ** 2)

    if ref_norm_sq < eps or orig_norm < eps:
        return steering_vector.clone()

    proj_scalar = torch.dot(steering_vector, refusal_direction) / ref_norm_sq
    v_proj = proj_scalar * refusal_direction
    deflated = steering_vector - v_proj

    deflated_norm = torch.norm(deflated, p=2)
    if preserve_norm and deflated_norm > eps:
        deflated = deflated * (orig_norm / deflated_norm)

    return deflated


def audit_refusal_overlap(
    vectors: Dict[str, torch.Tensor],
    refusal_direction: torch.Tensor,
) -> Dict[str, Dict[str, float]]:
    """Audits a collection of steering vectors for overlap with the refusal subspace.

    Args:
        vectors: Dictionary mapping condition names to steering vectors.
        refusal_direction: 1D tensor `(d_model,)` for refusal direction.

    Returns:
        Dictionary mapping condition name to metrics:
            - `cosine_sim`: Cosine similarity cos(Delta_x, v_refusal).
            - `dot_product`: Raw dot product <Delta_x, v_refusal>.
            - `proj_norm`: L2 norm of the projected refusal component.
            - `refusal_energy_pct`: Percentage of vector energy in refusal subspace.
    """
    ref_unit = refusal_direction / (torch.norm(refusal_direction, p=2) + 1e-8)
    audit_results: Dict[str, Dict[str, float]] = {}

    for name, vec in vectors.items():
        vec_norm = torch.norm(vec, p=2).item()
        cos_sim = compute_cosine_similarity(vec, refusal_direction)
        dot_prod = float(torch.dot(vec, ref_unit).item())
        proj_norm = abs(dot_prod)
        energy_pct = (proj_norm ** 2 / (vec_norm ** 2 + 1e-8)) * 100.0 if vec_norm > 0 else 0.0

        audit_results[name] = {
            "cosine_sim": cos_sim,
            "dot_product": dot_prod,
            "proj_norm": proj_norm,
            "refusal_energy_pct": energy_pct,
        }

    return audit_results
