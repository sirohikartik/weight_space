"""Training routines for Top-K Sparse Autoencoders."""

from typing import Dict, Optional
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from weight_space.models.sae import TopKSAE


def train_sae(
    activations: torch.Tensor,
    d_sae: int = 4096,
    k: int = 32,
    lr: float = 2e-3,
    weight_decay: float = 1e-5,
    epochs: int = 150,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> TopKSAE:
    """Train a Top-K Sparse Autoencoder on residual activations.

    Args:
        activations: Activation tensor of shape `(N, d_model)`.
        d_sae: Feature dictionary expansion size (e.g. 4096).
        k: Top-K active features per token.
        lr: Learning rate for AdamW optimizer.
        weight_decay: L2 weight decay.
        epochs: Number of training epochs.
        batch_size: Minibatch size.
        device: Target execution device. If None, inferred from activations.
        verbose: Whether to log training progress.

    Returns:
        Trained TopKSAE instance in eval mode.
    """
    if device is None:
        device = activations.device

    d_in = activations.shape[-1]
    sae = TopKSAE(d_in=d_in, d_sae=d_sae, k=k).to(device)

    dataset = TensorDataset(activations.to(device))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(sae.parameters(), lr=lr, weight_decay=weight_decay)

    sae.train()
    iterator = range(epochs)
    if verbose:
        iterator = tqdm(iterator, desc="Training Top-K SAE")

    for epoch in iterator:
        total_loss = 0.0
        for (batch_x,) in dataloader:
            optimizer.zero_grad()
            recon_x, z = sae(batch_x)
            loss = F.mse_loss(recon_x, batch_x)
            loss.backward()
            optimizer.step()

            # Maintain unit-norm decoder directions
            with torch.no_grad():
                sae.decoder.weight.data = F.normalize(sae.decoder.weight.data, dim=0)

            total_loss += loss.item()

        if verbose and (epoch + 1) % 50 == 0:
            avg_loss = total_loss / len(dataloader)
            sparsity = (z == 0).float().mean().item() * 100.0
            tqdm.write(f"Epoch {epoch+1:3d}/{epochs} | MSE Loss: {avg_loss:.5f} | Sparsity: {sparsity:.2f}% zeros")

    sae.eval()
    return sae
