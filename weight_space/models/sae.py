"""Top-K Sparse Autoencoder (SAE) implementation.

Sparse Autoencoders decompose dense residual representations into sparse,
interpretable overcomplete feature directions by enforcing hard Top-K sparsity.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKSAE(nn.Module):
    """Top-K Sparse Autoencoder for residual stream decomposition.

    Args:
        d_in: Dimensionality of the model activation space (residual stream).
        d_sae: Dimensionality of the overcomplete feature dictionary (expansion).
        k: Number of active features retained per token vector.
    """

    def __init__(self, d_in: int = 896, d_sae: int = 4096, k: int = 32):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.k = k

        self.encoder = nn.Linear(d_in, d_sae, bias=True)
        self.decoder = nn.Linear(d_sae, d_in, bias=False)
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        # Initialize decoder columns to unit norm
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(torch.randn(d_in, d_sae), dim=0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode activations into sparse Top-K feature coefficients.

        Args:
            x: Input tensor of shape `(..., d_in)`.

        Returns:
            Sparse feature activations `z` of shape `(..., d_sae)` where exactly
            `k` non-zero elements are kept per token.
        """
        x_cent = x - self.b_dec
        pre_acts = F.relu(self.encoder(x_cent))
        topk_vals, topk_indices = torch.topk(pre_acts, self.k, dim=-1)
        z = torch.zeros_like(pre_acts)
        z.scatter_(-1, topk_indices, topk_vals)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct residual activations from sparse feature activations.

        Args:
            z: Sparse feature tensor of shape `(..., d_sae)`.

        Returns:
            Reconstructed activations of shape `(..., d_in)`.
        """
        return self.decoder(z) + self.b_dec

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the SAE.

        Args:
            x: Residual activations `(..., d_in)`.

        Returns:
            Tuple of `(x_reconstructed, z)` where `z` is sparse feature activations.
        """
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        return x_reconstructed, z

    def get_feature_direction(self, feature_idx: int) -> torch.Tensor:
        """Return normalized decoder direction for a specific feature index.

        Args:
            feature_idx: Index of feature in `[0, d_sae - 1]`.

        Returns:
            Unit vector of shape `(d_in,)`.
        """
        w = self.decoder.weight[:, feature_idx]
        return F.normalize(w, dim=0)
