"""Decomposition of latent SAE feature directions into physical coordinate bases."""

from typing import Dict, List, Tuple
import torch


# Ground-truth physical coordinate decomposition for French feature #3460 (Layer 12, Qwen-2.5-0.5B)
FRENCH_FEATURE_3460_POS: Dict[int, float] = {
    490: 0.1988,
    507: 0.1625,
    266: 0.1538,
    579: 0.1410,
    570: 0.1374,
    439: 0.1265,
    312: 0.1241,
    781: 0.1189,
    115: 0.1150,
    640: 0.1112,
    305: 0.1087,
    422: 0.1065,
    591: 0.1042,
    723: 0.1018,
    812: 0.0995,
}

FRENCH_FEATURE_3460_NEG: Dict[int, float] = {
    100: -0.1737,
    131: -0.1379,
    43: -0.1336,
    850: -0.1256,
    194: -0.1238,
    702: -0.1195,
    58: -0.1142,
    339: -0.1108,
    614: -0.1085,
    220: -0.1051,
    804: -0.1023,
    512: -0.0998,
    667: -0.0975,
    341: -0.0954,
    188: -0.0932,
}


def find_monosemantic_features(domain_activations: Dict[str, torch.Tensor]) -> Dict[str, int]:
    """Find the most monosemantic feature index for each domain via differential contrast.

    Calculates: score(c) = act(c) - mean(act(other_domains))

    Args:
        domain_activations: Dict mapping domain name to mean feature activation tensor `(d_sae,)`.

    Returns:
        Dict mapping domain name to best monosemantic feature index.
    """
    domains = list(domain_activations.keys())
    best_features: Dict[str, int] = {}

    for target_domain in domains:
        other_acts = [domain_activations[d] for d in domains if d != target_domain]
        avg_other = torch.stack(other_acts).mean(dim=0)
        differential_score = domain_activations[target_domain] - avg_other
        best_features[target_domain] = int(torch.argmax(differential_score).item())

    return best_features


def decompose_feature_direction(
    direction: torch.Tensor, top_k: int = 15
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Decomposes a dense unit feature vector into its strongest positive and negative coordinates.

    Args:
        direction: 1D tensor of shape `(d_model,)`.
        top_k: Number of extreme positive and negative coordinates to retain.

    Returns:
        Tuple of `(positive_coords_dict, negative_coords_dict)`.
    """
    sorted_indices = torch.argsort(direction, descending=True)
    pos_indices = sorted_indices[:top_k]
    neg_indices = sorted_indices[-top_k:].flip(dims=[0])

    pos_dict = {int(idx.item()): float(direction[idx].item()) for idx in pos_indices}
    neg_dict = {int(idx.item()): float(direction[idx].item()) for idx in neg_indices}

    return pos_dict, neg_dict


def build_sparse_vector(
    coords: Dict[int, float], d_model: int = 896, device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """Builds a sparse steering vector in R^{d_model} from coordinate indices and weights."""
    vec = torch.zeros(d_model, device=device)
    for idx, weight in coords.items():
        vec[idx] = weight
    return vec
