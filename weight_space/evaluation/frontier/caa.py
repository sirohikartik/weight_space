"""Contrastive Activation Addition (CAA) baseline (Rimsky et al., 2023).

Extracts the canonical difference-in-means activation direction across positive and negative
contrastive pairs without requiring dictionary learning or autoencoders.
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer


def compute_caa_vector_from_activations(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    normalize: bool = False,
) -> torch.Tensor:
    """Computes difference-in-means vector directly from pre-extracted activation tensors.

    Args:
        pos_activations: Tensor of positive activations `(batch, seq, d_model)` or `(batch, d_model)`.
        neg_activations: Tensor of negative activations `(batch, seq, d_model)` or `(batch, d_model)`.
        normalize: Whether to unit normalize the resulting difference vector.

    Returns:
        1D tensor `(d_model,)`.
    """
    if pos_activations.dim() == 3:
        pos_mean = pos_activations.mean(dim=(0, 1))
    else:
        pos_mean = pos_activations.mean(dim=0)

    if neg_activations.dim() == 3:
        neg_mean = neg_activations.mean(dim=(0, 1))
    else:
        neg_mean = neg_activations.mean(dim=0)

    diff = pos_mean - neg_mean
    if normalize:
        diff = diff / (torch.norm(diff, p=2) + 1e-12)
    return diff


def compute_caa_vector(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    contrastive_pairs: List[Tuple[str, str]],
    layer_idx: int = 14,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Computes the Contrastive Activation Addition (CAA) steering direction.

    v_CAA = Mean(h_pos) - Mean(h_neg)

    Args:
        model: Causal LM.
        tokenizer: Pretrained tokenizer.
        contrastive_pairs: List of tuples `(pos_prompt, neg_prompt)`.
        layer_idx: Layer to extract activations from.
        device: Device to place tensors on.

    Returns:
        1D PyTorch tensor of shape `(d_model,)`.
    """
    if device is None:
        device = next(model.parameters()).device

    pos_activations: List[torch.Tensor] = []
    neg_activations: List[torch.Tensor] = []
    captured_acts: List[torch.Tensor] = []

    def hook_fn(module, inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        captured_acts.append(hidden.detach().to(torch.float32))

    # Access target layer
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layer = model.model.layers[layer_idx]
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layer = model.transformer.h[layer_idx]
    else:
        raise AttributeError(f"Unable to locate layer {layer_idx}")

    handle = layer.register_forward_hook(hook_fn)
    d_model = model.config.hidden_size

    with torch.no_grad():
        for pos_text, neg_text in contrastive_pairs:
            # Positive forward pass
            captured_acts.clear()
            inp_pos = tokenizer(pos_text, return_tensors="pt").to(device)
            model(**inp_pos)
            pos_activations.append(captured_acts[-1].view(-1, d_model).mean(dim=0))

            # Negative forward pass
            captured_acts.clear()
            inp_neg = tokenizer(neg_text, return_tensors="pt").to(device)
            model(**inp_neg)
            neg_activations.append(captured_acts[-1].view(-1, d_model).mean(dim=0))

    handle.remove()

    mean_pos = torch.stack(pos_activations).mean(dim=0)
    mean_neg = torch.stack(neg_activations).mean(dim=0)
    caa_dir = mean_pos - mean_neg
    return caa_dir.to(device)
