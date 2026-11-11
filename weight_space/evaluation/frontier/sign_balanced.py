"""NeuronHarness sign-balanced physical coordinate formulations."""

from typing import Dict, List, Tuple
import torch


def decompose_coordinates(
    direction: torch.Tensor, k_pos: int = 10, k_neg: int = 10
) -> Tuple[List[int], List[int], List[int]]:
    """Extracts top positive, top negative, and top magnitude physical coordinate indices.

    Args:
        direction: 1D tensor `(d_model,)` corresponding to SAE decoder column.
        k_pos: Number of top positive coordinates.
        k_neg: Number of top negative coordinates.

    Returns:
        Tuple of `(top_pos_indices, top_neg_indices, top_abs_indices)`.
    """
    sorted_asc = torch.argsort(direction, descending=False).cpu().tolist()
    sorted_desc = torch.argsort(direction, descending=True).cpu().tolist()
    sorted_abs = torch.argsort(torch.abs(direction), descending=True).cpu().tolist()

    top_pos = sorted_desc[:k_pos]
    top_neg = sorted_asc[:k_neg]
    top_abs = sorted_abs[:(k_pos + k_neg)]

    return top_pos, top_neg, top_abs


def build_sign_balanced_vector(
    direction: torch.Tensor,
    pos_indices: List[int],
    neg_indices: List[int],
) -> torch.Tensor:
    """Builds the balanced physical coordinate vector Delta_x = sum_{P} w_j e_j + sum_{N} w_j e_j."""
    v = torch.zeros_like(direction)
    for idx in pos_indices:
        v[idx] = direction[idx]
    for idx in neg_indices:
        v[idx] = direction[idx]
    return v


def build_positive_only_vector(
    direction: torch.Tensor,
    pos_indices: List[int],
) -> torch.Tensor:
    """Builds positive-only excitation vector Delta_x = sum_{P} w_j e_j."""
    v = torch.zeros_like(direction)
    for idx in pos_indices:
        v[idx] = direction[idx]
    return v


def build_negative_only_vector(
    direction: torch.Tensor,
    neg_indices: List[int],
) -> torch.Tensor:
    """Builds negative-only coordinate vector Delta_x = sum_{N} w_j e_j."""
    v = torch.zeros_like(direction)
    for idx in neg_indices:
        v[idx] = direction[idx]
    return v


def build_magnitude_only_vector(
    direction: torch.Tensor,
    abs_indices: List[int],
) -> torch.Tensor:
    """Builds magnitude-only vector without sign balance Delta_x = sum_{Top|w|} w_j e_j."""
    v = torch.zeros_like(direction)
    for idx in abs_indices:
        v[idx] = direction[idx]
    return v
