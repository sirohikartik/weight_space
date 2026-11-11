"""Multi-layer circuit steering and intermediate layer depth sweeps (Saadatinia et al., 2026)."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch


@dataclass
class LayerSweepResult:
    """Evaluation summary for a single layer intervention."""
    layer_idx: int
    adherence_pct: float
    adherence_ci: Tuple[float, float]
    collapse_pct: float
    collapse_ci: Tuple[float, float]
    corpus_perplexity: float
    refusal_overlap_cosine: float


def calculate_sweep_layers(
    total_layers: int,
    min_frac: float = 0.25,
    max_frac: float = 0.75,
    step: int = 2,
) -> List[int]:
    """Generates the intermediate layer indices to sweep across the residual stream.

    Args:
        total_layers: Total number of transformer layers L in the model.
        min_frac: Lower boundary fraction (default 0.25 = L/4).
        max_frac: Upper boundary fraction (default 0.75 = 3L/4).
        step: Stride between evaluated layers.

    Returns:
        Sorted list of integer layer indices.
    """
    start_idx = max(0, int(total_layers * min_frac))
    end_idx = min(total_layers - 1, int(total_layers * max_frac))
    layers = list(range(start_idx, end_idx + 1, max(1, step)))
    if end_idx not in layers and end_idx >= start_idx:
        layers.append(end_idx)
    return sorted(list(set(layers)))


def distribute_circuit_energy(
    vectors_by_layer: Dict[int, torch.Tensor],
    layer_weights: Optional[Dict[int, float]] = None,
    target_total_norm: float = 8.775,
    eps: float = 1e-8,
) -> Dict[int, torch.Tensor]:
    """Distributes steering energy across a circuit of multiple layers with strict total norm conservation.

    The total Euclidean energy across all layers satisfies:
        sqrt( sum_{l} ||Delta x_l||_2^2 ) == target_total_norm

    Args:
        vectors_by_layer: Dict mapping layer index l to unscaled candidate steering vector `(d_model,)`.
        layer_weights: Optional relative weighting per layer (e.g. Gaussian bell curve over mid-layers).
                       If None, energy is distributed equally.
        target_total_norm: Fixed target joint norm c.
        eps: Epsilon to prevent division by zero.

    Returns:
        Dict mapping layer index l to scaled steering vector Delta x_l.
    """
    if not vectors_by_layer:
        return {}

    layers = sorted(vectors_by_layer.keys())
    if layer_weights is None:
        weights = {l: 1.0 / len(layers) for l in layers}
    else:
        total_w = sum(layer_weights.get(l, 1.0) for l in layers)
        weights = {l: layer_weights.get(l, 1.0) / (total_w + eps) for l in layers}

    # Normalize individual vectors to unit L2 norm first
    unit_vectors: Dict[int, torch.Tensor] = {}
    for l in layers:
        v = vectors_by_layer[l]
        v_norm = torch.norm(v, p=2)
        unit_vectors[l] = v / (v_norm + eps)

    # Allocate squared norm according to normalized weights:
    # ||Delta x_l||_2^2 = weights[l] * (target_total_norm^2)
    # ||Delta x_l||_2   = sqrt(weights[l]) * target_total_norm
    scaled_circuit: Dict[int, torch.Tensor] = {}
    for l in layers:
        target_l_norm = (weights[l] ** 0.5) * target_total_norm
        scaled_circuit[l] = unit_vectors[l] * target_l_norm

    return scaled_circuit


def identify_optimal_layer(results: List[LayerSweepResult]) -> Optional[LayerSweepResult]:
    """Finds the layer maximizing target adherence while penalizing persona collapse and perplexity degradation.

    Score heuristic = Adherence - (2.0 * Collapse) - (10.0 * max(0, PPL - baseline_floor))
    """
    if not results:
        return None

    best_res = None
    best_score = -float("inf")

    for res in results:
        # Penalize collapse heavily and favor adherence
        score = res.adherence_pct - (2.0 * res.collapse_pct)
        if score > best_score:
            best_score = score
            best_res = res

    return best_res
