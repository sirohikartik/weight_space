"""Dynamic token scheduling for activation steering (Xie et al., 2025)."""

from typing import List, Optional
import math
import torch


class DynamicTokenScheduler:
    """Manages token-wise scaling coefficients for activation steering during generation.

    Supported strategies:
    - `constant`: Uniform injection at every generation step (alpha_t = alpha_0).
    - `linear_decay`: Linear ramp-down across decay_steps (alpha_t = alpha_0 * max(0, 1 - t / T)).
    - `exponential_decay`: Exponential attenuation (alpha_t = alpha_0 * exp(-gamma * t)).
    - `cosine_decay`: Half-cosine decay to zero over decay_steps.
    """

    def __init__(
        self,
        strategy: str = "constant",
        alpha_0: float = 35.0,
        decay_steps: int = 32,
        gamma: float = 0.05,
    ) -> None:
        valid_strategies = {"constant", "linear_decay", "exponential_decay", "cosine_decay"}
        if strategy not in valid_strategies:
            raise ValueError(f"Unknown scheduling strategy: '{strategy}'. Choose from {valid_strategies}")

        self.strategy = strategy
        self.alpha_0 = alpha_0
        self.decay_steps = max(1, decay_steps)
        self.gamma = gamma

    def get_scale(self, step: int) -> float:
        """Returns the scalar multiplier alpha_t at the given token generation step (0-indexed)."""
        if step < 0:
            return float(self.alpha_0)

        if self.strategy == "constant":
            return float(self.alpha_0)

        elif self.strategy == "linear_decay":
            progress = min(1.0, step / float(self.decay_steps))
            return float(self.alpha_0 * max(0.0, 1.0 - progress))

        elif self.strategy == "exponential_decay":
            return float(self.alpha_0 * math.exp(-self.gamma * step))

        elif self.strategy == "cosine_decay":
            if step >= self.decay_steps:
                return 0.0
            cos_factor = 0.5 * (1.0 + math.cos(math.pi * step / self.decay_steps))
            return float(self.alpha_0 * cos_factor)

        return float(self.alpha_0)

    def generate_schedule_curve(self, num_tokens: int) -> List[float]:
        """Generates list of alpha_t values for a trajectory of length num_tokens."""
        return [self.get_scale(t) for t in range(num_tokens)]

    def apply_schedule(self, vector: torch.Tensor, step: int) -> torch.Tensor:
        """Scales the steering vector according to the schedule at step."""
        scale = self.get_scale(step)
        # Assuming vector is normalized to unit norm or pre-scaled direction
        return vector * (scale / (self.alpha_0 + 1e-8)) if self.alpha_0 != 0 else torch.zeros_like(vector)
