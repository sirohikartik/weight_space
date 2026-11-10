"""Unit tests for TopKSAE."""

import pytest
import torch
from weight_space.models.sae import TopKSAE


def test_sae_initialization():
    sae = TopKSAE(d_in=128, d_sae=512, k=8)
    assert sae.encoder.weight.shape == (512, 128)
    assert sae.decoder.weight.shape == (128, 512)
    assert sae.b_dec.shape == (128,)


def test_sae_sparsity_enforcement():
    d_in = 64
    d_sae = 256
    k = 10
    sae = TopKSAE(d_in=d_in, d_sae=d_sae, k=k)

    x = torch.randn(8, d_in)
    x_rec, z = sae(x)

    assert x_rec.shape == x.shape
    assert z.shape == (8, d_sae)

    # Exactly k non-zeros per sample
    non_zeros = (z > 0).sum(dim=-1)
    assert torch.all(non_zeros == k)


def test_sae_feature_direction_normalized():
    sae = TopKSAE(d_in=64, d_sae=128, k=4)
    direction = sae.get_feature_direction(0)
    assert direction.shape == (64,)
    assert torch.isclose(torch.norm(direction), torch.tensor(1.0), atol=1e-5)
