"""Closed-form weight-space compilation and parameter modification algorithms."""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import PreTrainedModel
except ImportError:
    PreTrainedModel = nn.Module  # type: ignore


def get_mlp_down_proj(model: PreTrainedModel, layer_idx: int = 12) -> nn.Linear:
    """Retrieves the MLP down-projection linear module for a given transformer layer."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layer = model.model.layers[layer_idx]
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layer = model.transformer.h[layer_idx]
    else:
        raise AttributeError(f"Unable to locate layer {layer_idx}")

    if hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj"):
        return layer.mlp.down_proj
    elif hasattr(layer, "mlp") and hasattr(layer.mlp, "c_proj"):
        return layer.mlp.c_proj
    raise AttributeError(f"Unable to locate down_proj in layer {layer_idx}")


def apply_rank1_edit(
    down_proj: nn.Linear,
    direction: torch.Tensor,
    alpha: float = 2.5,
    u_vector: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compiles a latent feature direction directly into W_down via a closed-form rank-1 update.

    Equation:
        W_{down}^{new} = W_{down} + alpha * d * u^T

    Args:
        down_proj: The MLP down-projection module (weight shape: `(d_out, d_in)`).
        direction: Steering feature direction in residual space `(d_out,)`.
        alpha: Scaling magnitude for the edit.
        u_vector: Intermediate activation weighting vector `(d_in,)`.
            If None, defaults to the normalized all-ones vector.

    Returns:
        delta_W: The applied rank-1 update matrix of shape `(d_out, d_in)`.
    """
    device = down_proj.weight.device
    dtype = down_proj.weight.dtype
    d_out, d_in = down_proj.weight.shape

    d_col = direction.to(device=device, dtype=dtype).view(d_out, 1)

    if u_vector is None:
        u_row = F.normalize(torch.ones(1, d_in, device=device, dtype=dtype), dim=-1)
    else:
        u_row = F.normalize(u_vector.to(device=device, dtype=dtype).view(1, d_in), dim=-1)

    delta_W = alpha * torch.matmul(d_col, u_row)
    with torch.no_grad():
        down_proj.weight.data.add_(delta_W)

    return delta_W


def apply_orthogonal_ablation(down_proj: nn.Linear, direction: torch.Tensor) -> None:
    """Projects W_down onto the subspace orthogonal to direction.

    Equation:
        W_{down}^{new} = (I - d d^T) W_{down}

    Args:
        down_proj: The MLP down-projection module.
        direction: Vector to ablate `(d_out,)`.
    """
    device = down_proj.weight.device
    dtype = down_proj.weight.dtype
    d_out = down_proj.weight.shape[0]

    d = F.normalize(direction.to(device=device, dtype=dtype), dim=0).view(d_out, 1)
    proj_matrix = torch.eye(d_out, device=device, dtype=dtype) - torch.matmul(d, d.T)

    with torch.no_grad():
        down_proj.weight.data = torch.matmul(proj_matrix, down_proj.weight.data)


class WeightEditContext:
    """Context manager for safely applying and rolling back weight edits."""

    def __init__(self, down_proj: nn.Linear):
        self.down_proj = down_proj
        self.original_weight = None

    def __enter__(self):
        self.original_weight = self.down_proj.weight.data.clone()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.original_weight is not None:
            self.down_proj.weight.data.copy_(self.original_weight)
            self.original_weight = None
