"""Utilities for capturing residual stream activations from Transformer models."""

from typing import List, Tuple
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer


def collect_residual_activations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    texts: List[str],
    layer_idx: int = 12,
    device: str = "cuda",
) -> torch.Tensor:
    """Capture residual stream activations from a specified transformer layer.

    Args:
        model: HuggingFace causal language model (e.g. Qwen2.5).
        tokenizer: Pretrained tokenizer corresponding to the model.
        texts: List of strings to run through the model.
        layer_idx: Target transformer block index.
        device: Device to place tensors on.

    Returns:
        Tensor of shape `(total_tokens, hidden_dim)` containing un-embedded
        residual stream vectors.
    """
    captured: List[torch.Tensor] = []

    def hook_fn(module: nn.Module, input_args: Tuple, output: Tuple) -> None:
        out = output[0] if isinstance(output, tuple) else output
        captured.append(out.detach())

    # Compatible with Llama, Qwen, Mistral architectures (model.model.layers)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        target_layer = model.model.layers[layer_idx]
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        target_layer = model.transformer.h[layer_idx]
    else:
        raise AttributeError(f"Unable to locate layer {layer_idx} on model {type(model)}")

    handle = target_layer.register_forward_hook(hook_fn)

    model.eval()
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt").to(device)
            model(**inputs)

    handle.remove()

    hidden_dim = model.config.hidden_size
    stacked = torch.cat([act.view(-1, hidden_dim) for act in captured], dim=0)
    return stacked
