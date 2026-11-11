"""Euclidean energy clamping and isotropic baseline controls.

Ensures all experimental conditions hold ||Δx||₂ strictly constant at target energy c,
eliminating injection magnitude as a confounding variable.
"""

from typing import Dict, Tuple
import torch


def scale_to_target_norm(vec: torch.Tensor, target_norm: float, eps: float = 1e-12) -> torch.Tensor:
    """Clamps or projects a vector to have exactly target Euclidean norm c.

    Args:
        vec: 1D PyTorch tensor of shape `(d_model,)`.
        target_norm: Target L2 norm c > 0.
        eps: Numerical stability constant.

    Returns:
        Rescaled tensor with ||v||₂ == target_norm.
    """
    current_norm = float(torch.norm(vec, p=2).item())
    if current_norm < eps:
        return torch.zeros_like(vec)
    return (vec / current_norm) * target_norm


def generate_isotropic_noise_control(
    d_model: int,
    target_norm: float,
    seed: int = 42,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generates an isotropic Gaussian random vector clamped to target norm c.

    Serves as the empirical random-vector floor (Korznikov et al., 2025), testing whether
    steering gains exceed non-directional perturbation energy.

    Args:
        d_model: Hidden dimension size.
        target_norm: Target L2 norm c.
        seed: Random seed.
        device: PyTorch device.
        dtype: PyTorch dtype.

    Returns:
        Random Gaussian vector with ||v||₂ == target_norm.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(d_model, generator=g, dtype=dtype)
    return scale_to_target_norm(noise.to(device), target_norm)


def verify_norm_clamping(vector_dict: Dict[str, torch.Tensor], expected_norm: float, tolerance: float = 1e-3) -> bool:
    """Asserts that all steered conditions match expected norm within numerical tolerance."""
    for name, v in vector_dict.items():
        if v is None:
            continue
        actual = float(torch.norm(v, p=2).item())
        assert abs(actual - expected_norm) <= tolerance, (
            f"Norm check failed for condition '{name}': expected {expected_norm:.4f}, got {actual:.4f}"
        )
    return True
