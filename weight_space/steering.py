"""Activation steering hooks and steered generation interfaces."""

from typing import Optional
import torch
import torch.nn as nn
try:
    from transformers import PreTrainedModel, PreTrainedTokenizer
except ImportError:
    PreTrainedModel = nn.Module  # type: ignore
    PreTrainedTokenizer = object  # type: ignore


class SteeringHook:
    """Context manager for non-destructive activation steering on a Transformer layer."""

    def __init__(
        self,
        model: PreTrainedModel,
        steering_vector: torch.Tensor,
        alpha: float = 1.0,
        layer_idx: int = 12,
    ):
        self.model = model
        self.steering_vector = steering_vector
        self.alpha = alpha
        self.layer_idx = layer_idx
        self.handle = None

    def _get_target_layer(self) -> nn.Module:
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers[self.layer_idx]
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h[self.layer_idx]
        raise AttributeError(f"Unable to locate layer {self.layer_idx}")

    def __enter__(self):
        if self.steering_vector is not None and self.alpha != 0.0:
            target_layer = self._get_target_layer()
            vec = self.steering_vector.view(1, 1, -1)

            def hook_fn(module, input_args, output):
                hidden = output[0] if isinstance(output, tuple) else output
                steered = hidden + self.alpha * vec.to(hidden.device)
                if isinstance(output, tuple):
                    return (steered,) + output[1:]
                return steered

            self.handle = target_layer.register_forward_hook(hook_fn)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def generate_with_steering(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    steering_vector: Optional[torch.Tensor] = None,
    alpha: float = 14.0,
    layer_idx: int = 12,
    max_new_tokens: int = 85,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: Optional[torch.device] = None,
) -> str:
    """Generates autoregressive text while applying activation steering.

    Args:
        model: HuggingFace causal LM.
        tokenizer: Corresponding tokenizer.
        prompt: User input string.
        steering_vector: 1D steering vector of shape `(d_model,)`.
        alpha: Steering coefficient multiplier.
        layer_idx: Layer to steer.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling probability.
        device: Device to place tensors on.

    Returns:
        Generated continuation string with the prompt stripped.
    """
    if device is None:
        device = next(model.parameters()).device

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with SteeringHook(model, steering_vector=steering_vector, alpha=alpha, layer_idx=layer_idx):
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
            )

    full_text = tokenizer.decode(out[0], skip_special_tokens=True)
    if full_text.startswith(prompt):
        return full_text[len(prompt) :].strip()
    return full_text.strip()
